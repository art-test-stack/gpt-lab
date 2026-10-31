"""Offline harness checks. No Modal calls, compilation, downloads or GPU work."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

# The same sibling imports are used by the uploaded, standalone worker directory.
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze import trace_report
from common import SHA, TRAIN_ARGS, compare_aba, partition_intervals, union_length
from worker import GraphCapture, cpu_copy, restore_training_state, runtime_gemms, validate_dispatch
sys.path.pop(0)


def test_recipe_and_dry_run():
    from gpt_lab.cli.parser import build_parser
    from gpt_lab.model.auto import AutoGPTConfig
    args=build_parser().parse_args(TRAIN_ARGS)
    assert (args.num_steps,args.warmup_steps,args.n_acc_steps,args.device_batch_size)==(2317,300,16,16)
    assert args.total_batch_size==524288
    assert args.attn_impl=="fa3" and args.act_func=="swiglu"
    assert not args.fp8_ffn_forward
    assert (args.eval_core_every,args.eval_bpb_every,args.sample_every,args.save_every)==(0,0,0,0)
    cfg=AutoGPTConfig(depth=12,aspect_ratio=64,d_head=128,act_func="swiglu",attn_impl="fa3").build_transformer_config(12,50257)
    assert (cfg.d_model,cfg.d_ffn)==(768,3072)
    assert 8*cfg.d_model==6144
    result=subprocess.run([sys.executable,str(HERE/"launch.py"),"--run-name","offline-check","--dry-run"],capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan["sha"]==SHA and plan["gpu"]=="H100:1" and plan["workers"]==1
    assert plan["environment"]=="main"
    assert plan["warmup"]==20 and plan["measured"]==50


def test_snapshot_and_variation():
    import torch
    source={"model":torch.tensor([1.]),"state":{"x":[torch.tensor([2.])]}}
    snapshot=cpu_copy(source)
    source["model"].zero_()
    source["state"]["x"][0].zero_()
    assert snapshot["model"].item()==1 and snapshot["state"]["x"][0].item()==2
    assert compare_aba([100,101,99],[90,91,89],[100,101,99])["gain_exceeds_observed_variation"]
    assert not compare_aba([100,110,90],[99,109,89],[98,108,88])["gain_exceeds_observed_variation"]
    assert not compare_aba([100,101,99],[96,97,95],[90,91,89])["gain_exceeds_observed_variation"]


def test_interval_accounting_never_double_counts():
    assert union_length([(0,10),(5,15),(17,20)])==18
    result=partition_intervals([(0,10,"ffn"),(5,15,"attention"),(6,9,"ffn")],0,20)
    assert result=={"gpu_idle":5,"ffn":5,"overlap":5,"attention":5}
    assert sum(result.values())==20


def test_trace_correlation_and_complete_microsteps(tmp_path):
    def span(name,ts,dur,cat,tid=1,args=None):
        return dict(name=name,ts=ts,dur=dur,cat=cat,tid=tid,ph="X",args=args or {})
    events=[span("diagnostic/update/20",0,1000,"user_annotation"),
            span("diagnostic/gemm/ffn/forward/gemm_fp8|_scaled_mm",100,10,"user_annotation"),
            span("aten::_scaled_mm",101,8,"cpu_op",args={"External id":3}),
            span("cudaLaunchKernel",102,2,"cuda_runtime",args={"External id":3,"correlation":42}),
            span("sm90_fp8_gemm",120,100,"kernel",tid=7,args={"correlation":42}),
            span("another_stream_kernel",150,100,"kernel",tid=8),
            span("diagnostic/optimizer",300,200,"user_annotation"),
            span("diagnostic/gemm/other_gemm|mm",330,20,"user_annotation"),
            span("cudaLaunchKernel",335,2,"cuda_runtime",args={"correlation":43}),
            span("optimizer_gemm",360,80,"kernel",tid=7,args={"correlation":43})]
    events += [span(f"diagnostic/microstep/{i}",i*50,5,"user_annotation") for i in range(16)]
    path=tmp_path/"trace.json"
    path.write_text(json.dumps({"traceEvents":events}))
    report=trace_report(path,{})
    assert report["fp8_cuda_evidence"][0]["kernel"]=="sm90_fp8_gemm"
    update=report["updates"][0]
    assert sum(update["exclusive_ms"].values())==pytest.approx(update["wall_ms"])
    assert update["exclusive_ms"]["overlap"]==.07
    assert update["exclusive_ms"]["optimizer"]==.08
    events.pop()
    path.write_text(json.dumps({"traceEvents":events}))
    with pytest.raises(RuntimeError,match="incomplete"):
        trace_report(path,{})


def test_dispatch_requires_runtime_dtypes_and_compiled_path():
    forward=dict(category="ffn/forward/gemm_fp8",operator="_scaled_mm",
                 compiled_inductor=True,direction="forward",dtypes=["torch.float8_e4m3fn"]*2,
                 shapes=[[32768,768],[768,6144]])
    backward=dict(category="ffn/backward/gemm_bf16",operator="mm",
                  compiled_inductor=True,direction="backward",dtypes=["torch.bfloat16"]*2)
    second=forward | {"shapes":[[32768,3072],[3072,768]]}
    records=[forward.copy() for _ in range(16*12)]+[second.copy() for _ in range(16*12)]+[backward.copy() for _ in range(16*12*4)]
    validate_dispatch(records,True,updates=1)
    records[0]["compiled_inductor"]=False
    with pytest.raises(RuntimeError,match="not proven"):
        validate_dispatch(records,True,updates=1)
    records[0]["compiled_inductor"]=True
    records[0]["dtypes"]=["torch.bfloat16"]*2
    with pytest.raises(RuntimeError,match="operand dtypes"):
        validate_dispatch(records,True,updates=1)
    records[0]["dtypes"]=["torch.float8_e4m3fn"]*2
    with pytest.raises(RuntimeError,match="outside FFN"):
        validate_dispatch([*records,forward | {"category":"head_loss/forward"}],True,updates=1)


def test_compiled_runtime_observer_on_cpu(tmp_path):
    import torch
    from torch._inductor import config
    layer=torch.nn.Linear(16,16,bias=False)
    inputs=torch.randn(4,16,requires_grad=True)
    capture=GraphCapture(tmp_path)
    with config.patch({"fx_graph_cache":False,"trace.enabled":True,
                       "trace.debug_dir":str(tmp_path/"compiler"),"trace.provenance_tracking_level":1}):
        compiled=torch.compile(layer,backend=capture.backend,dynamic=False)
        compiled(inputs).sum().backward()
    layer.zero_grad(set_to_none=True)
    records=[]
    with runtime_gemms(records):
        compiled(inputs).sum().backward()
    assert records and all(item["compiled_inductor"] for item in records)
    assert {item["direction"] for item in records}=={"forward","backward"}
    assert capture.graphs and (tmp_path/"compiled_graphs.json").exists()


def test_restores_parameters_optimizer_rng_and_loader_position():
    import torch
    from types import SimpleNamespace
    from gpt_lab.model.checkpoint import capture_rng_state
    from gpt_lab.optim.factory import OptimizerFactory
    from gpt_lab.utils.distributed import ComputeContext
    model=torch.nn.Linear(4,4,bias=False)
    parameter=model.weight
    optimizer=OptimizerFactory([dict(params=[parameter],opt="adamw",lr=.01,
                                     betas=(.9,.99),eps=1e-8,weight_decay=0.)],
                               comp_info=ComputeContext(torch.device("cpu")))
    optimizer.state[parameter]={"step":2,"exp_avg":torch.ones_like(parameter)}
    trainer=SimpleNamespace(model=model,optimizer=optimizer,state=SimpleNamespace(step=17),prepare_batch=lambda batch:batch)
    weights=cpu_copy(model.state_dict())
    optim=cpu_copy(optimizer.state_dict())
    state=cpu_copy(trainer.state)
    rng=capture_rng_state()
    calls=[]
    def loader():
        calls.append(True)
        return iter([torch.rand(3),torch.rand(3)])
    batches=[]
    for _ in range(2):
        with torch.no_grad():
            parameter.zero_()
        optimizer.state[parameter]["exp_avg"].zero_()
        optimizer.param_groups[0]["lr"]=.9
        trainer.state.step=999
        iterator,batch=restore_training_state(trainer,weights,optim,state,rng,loader)
        batches.append((batch,next(iterator)))
        assert trainer.state.step==17 and optimizer.param_groups[0]["lr"]==.01
        assert model.weight is parameter
        torch.testing.assert_close(parameter,weights["weight"],rtol=0,atol=0)
        assert torch.equal(optimizer.state[parameter]["exp_avg"],torch.ones_like(parameter))
    assert len(calls)==2
    assert all(torch.equal(a,b) for a,b in zip(*batches))
