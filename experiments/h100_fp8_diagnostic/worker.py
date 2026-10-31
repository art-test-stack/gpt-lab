"""GPU-only harness. Production train_step, optimizer, loader and scalar loss."""

import argparse
from contextlib import contextmanager, nullcontext
import copy
import gc
import hashlib
import inspect
import json
from pathlib import Path
import platform
import re
import subprocess
import time
import traceback

from analyze import generated_kernel_categories, node_category, trace_report
from common import SHA, TRAIN_ARGS, compare_aba, distribution


def write(path, value):
    path.write_text(json.dumps(value,indent=2,default=str)+"\n")


def hardware_report(path):
    try:
        result=subprocess.run(["nvidia-smi","-q","-x"],capture_output=True,text=True)
        path.write_text(result.stdout+result.stderr)
    except OSError as error:
        path.write_text(f"Optional NVIDIA-SMI telemetry unavailable: {error}\n")


def cpu_copy(value):
    import torch
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {key:cpu_copy(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name,tensor in tensors:
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_dtypes(value, prefix=""):
    import torch
    if isinstance(value,torch.Tensor):
        return {prefix:str(value.dtype)}
    if isinstance(value,dict):
        return {name:dtype for key,item in value.items() for name,dtype in tensor_dtypes(item,f"{prefix}/{key}").items()}
    if isinstance(value,(tuple,list)):
        return {name:dtype for key,item in enumerate(value) for name,dtype in tensor_dtypes(item,f"{prefix}/{key}").items()}
    return {}


def restore_training_state(trainer, weights, optimizer, state, rng, loader_factory):
    from gpt_lab.model.checkpoint import set_rng_state
    trainer.model.zero_grad(set_to_none=True)
    trainer.model.load_state_dict(weights,strict=True,assign=False)
    trainer.optimizer.load_state_dict(copy.deepcopy(optimizer))
    trainer.state=copy.deepcopy(state)
    set_rng_state(rng)
    train_iter=iter(loader_factory())
    return train_iter,trainer.prepare_batch(next(train_iter))


class GraphCapture:
    def __init__(self, output):
        self.output, self.graphs = output, []
        self.runtime_kernels=[]

    def backend(self, gm, inputs):
        from torch._inductor.compile_fx import compile_fx, compile_fx_inner
        def inner(graph, values, **kwargs):
            nodes=[]
            for node in graph.graph.nodes:
                nodes.append(dict(name=node.name,target=str(node.target),
                                  modules=str(node.meta.get("nn_module_stack",node.meta.get("fwd_nn_module_stack",""))),
                                  stack=str(node.meta.get("stack_trace","")),
                                  origins=str(node.meta.get("from_node",""))))
            result = compile_fx_inner(graph,values,**kwargs)
            backward=bool(kwargs.get("is_backward"))
            mapping=json.loads(getattr(result,"inductor_provenance_mapping_str",None) or "{}")
            scope={node["name"]:node_category(node,backward) for node in nodes}
            function=inspect.unwrap(result.current_callable) if result.current_callable else None
            for name,kernel in getattr(function,"__globals__",{}).items():
                if name.startswith("triton_") and hasattr(kernel,"run"):
                    categories={scope[node] for node in mapping.get("cppCodeToPost",{}).get(name,[]) if node in scope}
                    category=next(iter(categories)) if len(categories)==1 else "mixed/"+"+".join(sorted(categories)) if categories else "unattributed"
                    self.runtime_kernels.append((kernel,name,category))
            self.graphs.append(dict(backward=bool(kwargs.get("is_backward")),nodes=nodes,
                                    source_code=getattr(result,"source_code",""),
                                    provenance=mapping,
                                    cache_key=getattr(result,"cache_key","")))
            write(self.output/"compiled_graphs.json",self.graphs)
            return result
        return compile_fx(gm,inputs,inner_compile=inner)


@contextmanager
def runtime_gemms(records, runtime_kernels=()):
    """Observe real extern GEMM operands after compilation; never cast them."""
    import torch
    from torch._inductor.select_algorithm import extern_kernels
    originals={}
    original_launchers=[]
    source_cache={}
    for name in ("_scaled_mm","mm","addmm","bmm"):
        if not hasattr(extern_kernels,name):
            continue
        original=getattr(extern_kernels,name)
        originals[name]=original
        def wrapped(*args,_name=name,_original=original,**kwargs):
            frame=inspect.currentframe().f_back
            filename=frame.f_code.co_filename
            if filename not in source_cache:
                source_cache[filename]=Path(filename).read_text() if Path(filename).is_file() else ""
            source=source_cache[filename]
            compiled="AOT ID:" in source
            direction="backward" if re.search(r"AOT ID:.*backward",source) else "forward"
            tensors=[arg for arg in args if isinstance(arg,torch.Tensor) and arg.ndim==2]
            shapes=[list(arg.shape) for arg in tensors[:2]]
            dtype=[str(arg.dtype) for arg in tensors[:2]]
            ffn=any(6144 in shape or 3072 in shape for shape in shapes) and any(32768 in shape for shape in shapes)
            family=(f"ffn/{direction}/gemm_"+("fp8" if _name=="_scaled_mm" else "bf16") if ffn else
                    f"head_loss/{direction}" if any(50304 in shape for shape in shapes) else "other_gemm")
            records.append(dict(operator=_name,shapes=shapes,dtypes=dtype,
                                compiled_inductor=compiled,direction=direction,category=family,
                                source=filename,line=frame.f_lineno,
                                use_fast_accum=kwargs.get("use_fast_accum")))
            with torch.profiler.record_function(f"diagnostic/gemm/{family}|{_name}"):
                return _original(*args,**kwargs)
        setattr(extern_kernels,name,wrapped)
    seen=set()
    for kernel,name,category in runtime_kernels:
        if id(kernel) in seen:
            continue
        seen.add(id(kernel))
        original=kernel.run
        def launch(*args,_original=original,_name=name,_category=category,**kwargs):
            with torch.profiler.record_function(f"diagnostic/kernel/{_category}|{_name}"):
                return _original(*args,**kwargs)
        original_launchers.append((kernel,original))
        kernel.run=launch
    try:
        yield
    finally:
        for name,original in originals.items():
            setattr(extern_kernels,name,original)
        for kernel,original in original_launchers:
            kernel.run=original


def validate_dispatch(records, fp8, updates=2):
    forward=[item for item in records if item["category"].startswith("ffn/forward/gemm")]
    expected=updates*16*12*2
    if len(forward)!=expected or not all(item["compiled_inductor"] for item in forward):
        raise RuntimeError(f"Compiled FFN dispatch not proven: {len(forward)} observed, expected {expected}. Inspect raw trace and generated code; no acceleration claim.")
    target="torch.float8_e4m3fn" if fp8 else "torch.bfloat16"
    if any(item["dtypes"]!=[target,target] for item in forward):
        raise RuntimeError("Unexpected real FFN GEMM operand dtypes")
    if any(item["operator"]=="_scaled_mm" and item["direction"]=="backward" for item in records):
        raise RuntimeError("Unexpected FP8 backward GEMM")
    if any(item["operator"]=="_scaled_mm" and item["category"]!="ffn/forward/gemm_fp8" for item in records):
        raise RuntimeError("Unexpected FP8 GEMM outside FFN forward")
    backward=[item for item in records if item["category"].startswith("ffn/backward/gemm")]
    if len(backward)!=expected*2 or any(item["dtypes"]!=["torch.bfloat16"]*2 for item in backward):
        raise RuntimeError("BF16 FFN backward dispatch not proven")
    expected_shapes={(32768,768,6144):updates*16*12,(32768,3072,768):updates*16*12}
    actual_shapes={shape:0 for shape in expected_shapes}
    for item in forward:
        shape=(item["shapes"][0][0],item["shapes"][0][1],item["shapes"][1][1])
        if shape not in actual_shapes:
            raise RuntimeError(f"Wrong production FFN GEMM shape: {shape}")
        actual_shapes[shape]+=1
    if actual_shapes!=expected_shapes:
        raise RuntimeError("Missing production projection shapes")


def diagnostic(trainer, loader_factory, output, warmup, measured):
    import torch
    import torchao
    from gpt_lab.model.checkpoint import capture_rng_state
    from gpt_lab.model.fp8 import DisableFP8, check_fp8_support, enable_fp8_ffn_forward
    from gpt_lab.model.flash_attn import require_flash_attn3
    check_fp8_support(trainer.device,trainer.dtype)
    cfg=trainer.model.config
    assert (cfg.d_model,cfg.d_ffn,cfg.max_context,cfg.n_layers,cfg.act_func,cfg.attn_impl)==(768,3072,2048,12,"swiglu","fa3")
    assert (trainer.config.n_steps,trainer.config.lr_warmup_steps,trainer.config.n_acc_steps,
            trainer.config.device_batch_size,trainer.config.total_batch_size)==(2317,300,16,16,524288)
    assert trainer.config.comp_info.world_size==1 and trainer.scaler is None
    model=trainer.model
    initial_weights=cpu_copy(model.state_dict())
    initial_optimizer=cpu_copy(trainer.optimizer.state_dict())
    initial_state=copy.deepcopy(trainer.state)
    initial_rng=capture_rng_state()
    identities={name:id(parameter) for name,parameter in model.named_parameters(remove_duplicate=False)}
    stored_dtypes={name:str(parameter.dtype) for name,parameter in model.named_parameters()}
    storage_before=stored_dtypes.copy()
    initial_hash=tensor_hash(initial_weights.items())
    enable_fp8_ffn_forward(model,torch.bfloat16)
    assert identities=={name:id(parameter) for name,parameter in model.named_parameters(remove_duplicate=False)}
    report=dict(sha=SHA,model=cfg.model_dump(),trainer=trainer.config.model_dump(mode="json"),
                torch=torch.__version__,torchao=torchao.__version__,python=platform.python_version(),
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
                gpu_properties=str(torch.cuda.get_device_properties(0)),
                fa3_module_path=getattr(require_flash_attn3(),"__file__",None),
                initial_model_sha256=initial_hash,parameter_storage_dtypes=stored_dtypes,
                parameter_identity_preserved=True,optimizer_initial_tensor_dtypes=tensor_dtypes(initial_optimizer),
                full_vocabulary_objective=True,precision_recipe=str(model.blocks[0].ffn.w_1.config),
                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                useful_flops_per_token=model.estimate_flops(),
                mfu_denominator="989e12 dense BF16 FLOP/s; BF16-peak-normalized utilization for both precision paths",
                warmup_updates=warmup,measured_updates=measured,legs={})
    write(output/"report.json",report)

    def reset():
        train_iter,batch=restore_training_state(trainer,initial_weights,initial_optimizer,initial_state,initial_rng,loader_factory)
        assert stored_dtypes=={name:str(parameter.dtype) for name,parameter in model.named_parameters()}
        assert identities=={name:id(parameter) for name,parameter in model.named_parameters(remove_duplicate=False)}
        return train_iter,batch

    def execute_leg(name,fp8,profile=False):
        print(f"Starting {name}",flush=True)
        leg_dir=output/name
        leg_dir.mkdir()
        toggle=nullcontext() if fp8 else DisableFP8(model)
        with toggle:
            train_iter,batch=reset()
            first_hash=tensor_hash(zip(("x","y"),batch[0]))
            model.train()
            trainer.compiled_model=None
            trainer.compiled_training_loss=None
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            graphs=GraphCapture(leg_dir)
            from torch._inductor import config as inductor_config
            patches={"trace.enabled":True,"trace.debug_dir":str(leg_dir/"compiler"),
                     "trace.provenance_tracking_level":1,"fx_graph_cache":False} if profile else {}
            with inductor_config.patch(patches):
                backend=graphs.backend if profile else "inductor"
                trainer.compiled_model=torch.compile(model,backend=backend,dynamic=False)
                trainer.compiled_training_loss=torch.compile(model.training_loss,backend=backend,dynamic=False)
                warm_times=[]
                total_dt=[]
                start=time.perf_counter()
                for step in range(warmup):
                    trainer.state.step=step
                    batch,metrics=trainer.train_step(train_iter,batch,False)
                    warm_times.append(metrics["step_time"]*1000)
                    trainer.update_metrics(metrics,total_dt)
                    trainer.state.completed_steps=step+1
                    if step==0:
                        gc.disable()
                startup=time.perf_counter()-start
                torch.cuda.reset_peak_memory_stats()
                losses=[]
                schedules=[]
                times=[]
                records=[]
                gradient_dtypes={}
                original_optimizer_step=trainer.optimizer_step
                original_compute_loss=trainer.compute_loss
                microstep=0
                def optimizer_step(log):
                    for key,parameter in model.named_parameters():
                        if parameter.grad is not None:
                            gradient_dtypes[key]=str(parameter.grad.dtype)
                            assert parameter.grad.dtype==parameter.dtype
                    with torch.profiler.record_function("diagnostic/optimizer"):
                        return original_optimizer_step(log)
                def compute_loss(payload):
                    nonlocal microstep
                    with torch.profiler.record_function(f"diagnostic/microstep/{microstep}"):
                        value=original_compute_loss(payload)
                    microstep+=1
                    return value
                activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
                # Runtime wrappers already capture exact GEMM shapes and source callsites.
                profiler=torch.profiler.profile(activities=activities) if profile else nullcontext()
                try:
                    if profile:
                        trainer.optimizer_step=optimizer_step
                        trainer.compute_loss=compute_loss
                    with profiler as trace, runtime_gemms(records,graphs.runtime_kernels) if profile else nullcontext():
                        for step in range(warmup,warmup+(2 if profile else measured)):
                            trainer.state.step=step
                            marker=torch.profiler.record_function(f"diagnostic/update/{step}") if profile else nullcontext()
                            with marker:
                                batch,metrics=trainer.train_step(train_iter,batch,False)
                            assert metrics["tokens"]==524288
                            losses.append(float(metrics["loss"]))
                            times.append(metrics["step_time"]*1000)
                            schedules.append({key:metrics[key] for key in ("lrm","muon_momentum","weight_decay")})
                            trainer.update_metrics(metrics,total_dt)
                            trainer.state.completed_steps=step+1
                finally:
                    trainer.optimizer_step=original_optimizer_step
                    trainer.compute_loss=original_compute_loss
                result=dict(timing=distribution(times),losses=losses,schedules=schedules,fp8_ffn_forward=fp8,
                            first_batch_sha256=first_hash,last_prefetched_batch_sha256=tensor_hash(zip(("x","y"),batch[0])),
                            last_loader_state=trainer.state.train_loader_state.model_dump() if trainer.state.train_loader_state else None,
                            initial_model_sha256=initial_hash,initial_schedule_step=initial_state.step,
                            measured_schedule_steps=list(range(warmup,warmup+len(times))),
                            startup_and_warmup_seconds=startup,warmup_update_ms=warm_times,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            optimizer_tensor_dtypes=tensor_dtypes(trainer.optimizer.state_dict()))
                result["useful_flops_per_second"]=model.estimate_flops()*result["timing"]["tokens_per_second"]
                result["bf16_peak_normalized_utilization_percent"]=100*result["useful_flops_per_second"]/989e12
                if profile:
                    trace.export_chrome_trace(str(leg_dir/"trace.json"))
                    write(leg_dir/"gemm_operands.json",records)
                    write(leg_dir/"gradient_storage_dtypes.json",gradient_dtypes)
                    categories=generated_kernel_categories(graphs.graphs,leg_dir/"compiler")
                    write(leg_dir/"kernel_provenance.json",categories)
                    result["profile"]=trace_report(leg_dir/"trace.json",categories)
                    write(leg_dir/"trace_summary.json",result["profile"])
                    # Persist partial evidence even if validation fails.
                    write(leg_dir/"result.json",result)
                    validate_dispatch(records,fp8)
                    if fp8 and sum(item["gemm_named"] for item in result["profile"]["fp8_cuda_evidence"])<2*16*12*2:
                        raise RuntimeError("Insufficient CUDA GEMM kernels correlated with actual compiled FP8 FFN calls; inspect raw trace")
                write(leg_dir/"result.json",result)
                report["legs"][name]=result
                write(output/"report.json",report)
                print(f"{name}: {result['timing']['median_ms']:.3f} ms",flush=True)
        trainer.compiled_model=None
        trainer.compiled_training_loss=None
        return result

    # Prove dispatch before spending time on the longer unprofiled comparison.
    execute_leg("profile-bf16",False,profile=True)
    execute_leg("profile-fp8-forward",True,profile=True)
    for key in ("first_batch_sha256","last_prefetched_batch_sha256","last_loader_state","schedules","optimizer_tensor_dtypes"):
        assert report["legs"]["profile-bf16"][key]==report["legs"]["profile-fp8-forward"][key],f"Unmatched profile controls: {key}"
    profiles={}
    for name in ("profile-bf16","profile-fp8-forward"):
        windows=report["legs"][name]["profile"]["updates"]
        keys={key for window in windows for key in window["exclusive_ms"]}
        profiles[name]={key:sum(window["exclusive_ms"].get(key,0) for window in windows)/len(windows) for key in keys}
    report["profile_comparison_mean_exclusive_ms_per_update"]=profiles
    report["profile_ffn_forward_gemm_saving_ms"]=(profiles["profile-bf16"].get("ffn/forward/gemm_bf16",0)-
                                                  profiles["profile-fp8-forward"].get("ffn/forward/gemm_fp8",0))
    report["profile_interpretation"]="Compare FFN GEMM saving with quantization_casts_or_fused, mixed_fused_work, other component times and exposed idle. Mixed fused work is not a pure quantization measurement. Profiling overhead is excluded from the ABA comparison."
    write(output/"report.json",report)
    # No profiler, hooks, runtime GEMM wrappers or debug compilation in ABA timing.
    before=execute_leg("bf16-before",False)
    fp8=execute_leg("fp8-forward",True)
    after=execute_leg("bf16-after",False)
    for key in ("first_batch_sha256","last_prefetched_batch_sha256","last_loader_state","schedules","optimizer_tensor_dtypes"):
        assert before[key]==fp8[key]==after[key],f"Unmatched data: {key}"
    report["comparison"]=compare_aba(*(item["timing"]["samples_ms"] for item in (before,fp8,after)))
    report["comparison"]["max_bf16_repeat_loss_difference"]=max(abs(a-b) for a,b in zip(before["losses"],after["losses"]))
    report["comparison"]["max_fp8_paired_loss_difference"]=max(abs(a-b) for a,b in zip(before["losses"],fp8["losses"]))
    write(output/"report.json",report)
    assert storage_before=={name:str(parameter.dtype) for name,parameter in model.named_parameters()}
    report["complete"]=True
    report["limitations"]="Two profiled updates are diagnostic, not unprofiled timings. Mixed/unattributed fused kernels are explicit. No quality equivalence or hardware acceleration claim unless runtime dispatch and CUDA correlation gates pass. No FP8 backward, caches, head or QKV conversion."
    write(output/"report.json",report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--measured",type=int,default=50)
    args=parser.parse_args()
    if args.warmup<20 or args.measured<50 or args.warmup+args.measured>2317:
        parser.error("Require >=20 warmup, >=50 measured updates, within horizon 2317")
    assert subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()==SHA
    import torch
    if torch.cuda.device_count()!=1 or "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("This diagnostic requires exactly one H100")
    from gpt_lab.model.flash_attn import require_flash_attn3
    require_flash_attn3()
    from gpt_lab.data import loader
    from gpt_lab.train.trainer import Trainer
    from gpt_lab.model.checkpoint import CheckpointManager
    build=loader.build_dataloader
    factories=[]
    def remember_loader(**kwargs):
        if kwargs["split"]=="train":
            factories.append(lambda:build(**kwargs))
        return build(**kwargs)
    loader.build_dataloader=remember_loader
    Trainer.train=lambda self:diagnostic(self,factories[0],args.output,args.warmup,args.measured)
    Trainer.save_checkpoint=lambda self:None
    CheckpointManager.save_training_config=lambda self,config:None
    from gpt_lab.cli import main as cli
    scratch=Path(".tmp/codex/h100-fp8-diagnostic")
    manifest={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")}
    write(args.output/"harness_manifest.json",manifest)
    hardware_report(args.output/"nvidia-smi-start.xml")
    try:
        cli([*TRAIN_ARGS,"--run-name",args.output.name,"--model-dir",str(scratch/"models"),
             "--board-dir",str(scratch/"board"),"--stop-after-step",str(args.warmup+args.measured)])
    except BaseException:
        write(args.output/"failure.json",dict(traceback=traceback.format_exc()))
        raise
    finally:
        hardware_report(args.output/"nvidia-smi-end.xml")


if __name__=="__main__":
    main()
