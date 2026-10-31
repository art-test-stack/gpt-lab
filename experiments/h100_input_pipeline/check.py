"""Focused offline fixture checks: no Modal calls, downloads, compilation or GPU."""
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from cache import capture, Replay, verify, owned, digest_batch
from fixture import factory
from recipe import SHA, train_args, BATCH_SIZES, ORDER, cpu_copy, resolve_cpu_batch, batch_config, combine_reports
from runner import restore, run_updates
from tracing import summarize, ranges, FetchRange


@pytest.mark.parametrize("device_batch_size,expected_accumulation", [(16,16),(32,8)])
def test_plan_and_literal_transfer_tail(device_batch_size,expected_accumulation):
    from gpt_lab.cli.parser import build_parser
    from gpt_lab.data.loader import DistDataLoader
    args = build_parser().parse_args(train_args(device_batch_size))
    assert (args.num_steps,args.warmup_steps,args.n_acc_steps,args.device_batch_size,args.max_seq_len) == (2317,300,-1,device_batch_size,2048)
    resolved = resolve_cpu_batch(device_batch_size)
    assert resolved == dict(device_batch_size=device_batch_size,seq_len=2048,world_size=1,
                            n_acc_steps=expected_accumulation,total_batch_size=524288)
    assert args.total_batch_size == 524288 and args.target_param_data_ratio == 8
    assert args.act_func == "swiglu" and args.attn_impl == "fa3" and not args.fp8_ffn_forward
    assert (args.save_every,args.eval_bpb_every,args.eval_core_every,args.sample_every) == (0,0,0,0)
    source = inspect.getsource(DistDataLoader.__next__)
    replay = inspect.getsource(Replay.__next__)
    for line in ('self.gpu.copy_(self.cpu, non_blocking=False)',
                 'data = self.gpu.view(B, T + 1)',
                 'self.inputs.copy_(data[:, :-1])', 'self.targets.copy_(data[:, 1:])'):
        assert line in source
        assert line.replace('view(B, T + 1)','view(self.B, self.T + 1)') in replay
    result = subprocess.run([sys.executable,str(HERE/"launch.py"),"--run-name","offline-check","--dry-run"],
                            capture_output=True,text=True,check=True)
    plan = json.loads(result.stdout)
    assert plan["production_sha"] == SHA and plan["order"] == ORDER
    assert (plan["warmup"],plan["measured"],plan["workers"]) == (25,50,1)
    assert plan["cpu_preflight_first"]
    assert plan["device_batch_sizes"] == list(BATCH_SIZES)
    for size in BATCH_SIZES:
        planned = build_parser().parse_args(plan["training"][str(size)])
        assert planned.n_acc_steps == -1 and planned.device_batch_size == size


def test_real_file_cache_ownership_and_strides(tmp_path):
    make = factory(tmp_path)
    loader = make()
    packed,batches,hashes = capture(loader,9)
    before = [tensor.clone() for tensor in (packed[0],*batches[0][:2])]
    for _ in range(20):
        next(loader)
    for actual,expected in zip((packed[0],*batches[0][:2]),before):
        verify(actual,expected)
    pointers = [t.data_ptr() for batch in batches for t in batch[:2]]
    assert len(set(pointers)) == len(pointers)
    for gpu in (False,True):
        replay = Replay(packed,batches,"cpu",gpu=gpu)
        assert [digest_batch(next(replay)) for _ in batches] == hashes
        with pytest.raises(StopIteration):
            next(replay)
    # The ownership helper also preserves noncontiguous layouts.
    view = torch.arange(24).view(4,6)[:,::2]
    clone = owned(view,"cpu")
    verify(clone,view)
    view.zero_()
    assert clone.sum() > 0


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(300,4)
        self.head = torch.nn.Linear(4,300)

    def training_loss(self,x,y):
        return torch.nn.functional.cross_entropy(self.head(self.embed(x)).flatten(0,1),y.flatten())


@pytest.mark.parametrize("device_batch_size", [2,4])
def test_restore_and_complete_production_updates_across_paths(tmp_path,device_batch_size):
    from gpt_lab.model.checkpoint import capture_rng_state
    from gpt_lab.train.trainer import Trainer
    from gpt_lab.utils.distributed import ComputeContext
    from gpt_lab.utils.schemas import TrainerConfig
    from gpt_lab.optim.factory import OptimizerFactory
    from gpt_lab.model.auto import AutoGPTConfig
    auto = AutoGPTConfig(run_name="fixture",device_batch_size=device_batch_size,max_seq_len=8,
                         total_batch_size=64,n_acc_steps=-1,
                         comp_info=ComputeContext(torch.device("cpu"),compute_dtype=torch.float32))
    auto.n_acc_steps = auto.compute_accumulation_steps(auto.total_batch_size)
    resolved = batch_config(auto,auto.max_seq_len)
    make = factory(tmp_path/"data",batch_size=resolved["device_batch_size"],seq_len=resolved["seq_len"])
    packed,batches,_ = capture(make(),resolved["n_acc_steps"]*3+1)
    model = TinyLM()
    optimizer = OptimizerFactory([dict(params=list(model.parameters()),opt="adamw",lr=.001,
                                      betas=(.9,.99),eps=1e-8,weight_decay=.01)],
                                 comp_info=ComputeContext(torch.device("cpu")))
    config = TrainerConfig(n_steps=10,n_acc_steps=resolved["n_acc_steps"],total_batch_size=resolved["total_batch_size"],
                           device_batch_size=resolved["device_batch_size"],n_flops_per_token=100,
                           comp_info=ComputeContext(torch.device("cpu"),compute_dtype=torch.float32))
    trainer = Trainer(model,None,make(),None,config=config,optimizer=optimizer,
                      checkpoint_manager=SimpleNamespace(source_dir=tmp_path/"model"))
    trainer.compiled_model = model
    trainer.compiled_training_loss = model.training_loss
    weights,optim,state,rng = cpu_copy(model.state_dict()),cpu_copy(optimizer.state_dict()),copy.deepcopy(trainer.state),capture_rng_state()
    results = []
    random_samples = []
    original_methods = (torch.Tensor.backward,torch.Tensor.copy_)
    for kind in ORDER:
        if kind.startswith("production"):
            selected = make
        else:
            selected = lambda:Replay(packed,batches,"cpu",gpu=kind=="gpu-cache")
        iterator,batch = restore(trainer,weights,optim,state,rng,selected)
        assert trainer.state.step == state.step and not optimizer.state
        random_samples.append(torch.rand(3))
        assert all(torch.equal(model.state_dict()[key],value) for key,value in weights.items())
        # Exercise the same instrumented path too; it must restore Tensor methods.
        context = ranges(trainer,iterator) if kind.startswith("production") else __import__("contextlib").nullcontext()
        with torch.compiler.set_stance("force_eager"), context:
            actual_iterator = FetchRange(iterator) if kind.startswith("production") else iterator
            batch,seconds,losses,schedules = run_updates(trainer,actual_iterator,batch,0,3)
        assert (torch.Tensor.backward,torch.Tensor.copy_) == original_methods
        assert trainer.state.completed_steps == 3 and optimizer.state
        assert trainer.state.n_tokens == 3*resolved["total_batch_size"]
        assert not all(torch.equal(model.state_dict()[key],value) for key,value in weights.items())
        results.append((cpu_copy(model.state_dict()),losses,schedules,digest_batch((*batch[0],batch[1]))))
    assert all(torch.equal(random_samples[0],sample) for sample in random_samples[1:])
    for result in results[1:]:
        assert result[1:] == results[0][1:]
        for key,value in result[0].items():
            torch.testing.assert_close(value,results[0][0][key],rtol=0,atol=0)


@pytest.mark.parametrize("device_batch_size", BATCH_SIZES)
def test_idle_gaps_use_union_and_report_temporal_overlap(tmp_path,device_batch_size):
    def span(name,start,duration,cat="user_annotation"):
        return dict(name=name,ts=start,dur=duration,cat=cat,ph="X")
    resolved = resolve_cpu_batch(device_batch_size)
    events = [span("diagnostic/update/25",0,100),span("kernel",0,20,"kernel"),
              span("kernel",10,20,"kernel"),span("copy",50,10,"gpu_memcpy"),span("kernel",70,30,"kernel")]
    events += [span("compute/optimizer",99,1)]
    for _ in range(resolved["n_acc_steps"]):
        events += [span("input/fetch",10,50),span("input/transfer",40,20),
                   span("compute/forward",1,1),span("compute/backward",2,1)]
    path = tmp_path/"trace.json"
    path.write_text(json.dumps(dict(traceEvents=events)))
    result = summarize(path,expected_updates=1,expected_microsteps=resolved["n_acc_steps"])["updates"][0]
    assert result["gpu_idle_ms"] == pytest.approx(.03)
    assert result["idle_overlapping_fetch_ms"] == pytest.approx(.02)
    assert result["idle_overlapping_transfer_ms"] == pytest.approx(.01)


def test_summary_separates_batch_and_input_effects():
    reports = {}
    for size,accumulation in ((16,16),(32,8)):
        legs = {name:dict(global_tokens_per_second=float(size),timing=dict(median_ms=100/size,p95_ms=120/size),
                          schedules=[dict(lrm=.1)],peak_allocated_bytes=size*100,peak_reserved_bytes=size*200)
                for name in ORDER}
        reports[str(size)] = dict(complete=True,initial_model_sha256="same",initial_optimizer_groups=[],model={},
                                 parameter_storage_dtypes={},first_row_sha256="same",trainer=dict(n_steps=2317,
                                 n_acc_steps=accumulation,device_batch_size=size),
                                 resolved_batch=dict(n_acc_steps=accumulation),legs=legs)
    result = combine_reports(reports)
    assert len(result["configurations"]) == len(BATCH_SIZES)*len(ORDER)
    assert result["batch32_vs_batch16_same_path"]["production-before"]["throughput_ratio"] == 2
    reports["32"]["initial_model_sha256"] = "different"
    with pytest.raises(AssertionError,match="initial_model"):
        combine_reports(reports)
