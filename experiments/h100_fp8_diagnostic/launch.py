"""User-launched diagnostic: one worker, one H100, fixed production SHA."""

import argparse
import json
from pathlib import Path
import re

from common import REPOSITORY, SHA, TRAIN_ARGS


def preflight(run_name):
    from pathlib import Path
    import json
    cache = Path("/cache/gpt_lab")
    data = cache / "data/climbmix-base"
    shards = sorted(data.glob("shard_*.parquet"))
    if len(shards)<5 or [p.name for p in shards] != [f"shard_{i:05d}.parquet" for i in range(len(shards))]:
        raise RuntimeError("Need existing contiguous climbmix-base shards; downloads are disabled")
    if any(p.stat().st_size==0 for p in shards):
        raise RuntimeError("Empty cached shard")
    if not any(p.is_file() for p in (cache/"tokenizers/gpt2").rglob("*")):
        raise RuntimeError("Need existing cached GPT2 tokenizer; downloads are disabled")
    metadata = json.loads((data/"meta.json").read_text()) if (data/"meta.json").exists() else {}
    index = metadata.get("max_shards") or len(shards)-1
    if type(index) is not int or not 4<=index<len(shards):
        raise RuntimeError("Invalid cached validation-shard boundary")
    expected=dict(name="climbmix-base",base_url="",column_name="text",max_shards=index)
    if metadata and any(metadata.get(key)!=value for key,value in expected.items()):
        raise RuntimeError("Dataset metadata differs from the production loader convention")
    if (Path("/runs")/run_name).exists():
        raise FileExistsError("Choose a fresh run name")
    return dict(shards=len(shards), validation_shard=index)


def worker(run_name, warmup, measured, runs):
    import os
    from pathlib import Path
    import subprocess
    root = Path("/runs")/run_name
    root.mkdir(exist_ok=False)
    environment = os.environ | {
        "GPTLAB_CACHE_DIR":"/cache/gpt_lab", "GPTLAB_DTYPE":"bfloat16",
        "PYTHONUNBUFFERED":"1", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
        "PYTORCH_ALLOC_CONF":"expandable_segments:True",
        "TORCHINDUCTOR_CACHE_DIR":"/opt/gpt-lab/.tmp/codex/h100-fp8-diagnostic/compiler_cache",
        "TORCH_COMPILE_DEBUG_DIR":str(root/"compiler_debug"),
        "TORCHINDUCTOR_UNIQUE_KERNEL_NAMES":"1",
    }
    command = ["/opt/gpt-lab/.venv/bin/python", "/opt/diagnostic/worker.py",
               "--output",str(root),"--warmup",str(warmup),"--measured",str(measured)]
    try:
        with (root/"worker.log").open("w",buffering=1) as log:
            with subprocess.Popen(command,cwd="/opt/gpt-lab",env=environment,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as process:
                for line in process.stdout:
                    print(line,end="",flush=True)
                    log.write(line)
                code = process.wait()
        if code:
            raise RuntimeError(f"Diagnostic failed ({code}); inspect {run_name}/worker.log and partial reports")
    finally:
        runs.commit()
    return str(root/"report.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--measured",type=int,default=50)
    parser.add_argument("--detach",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}",args.run_name):
        parser.error("run-name must contain 1–100 letters/digits/_/-")
    if args.warmup<20 or args.measured<50 or args.warmup+args.measured>2317:
        parser.error("Require >=20 warmup, >=50 measured updates, within horizon 2317")
    print(json.dumps(dict(sha=SHA, repository=REPOSITORY,environment="main",gpu="H100:1",workers=1,
                         order=["profile-bf16","profile-fp8-forward","bf16-before","fp8-forward","bf16-after"],
                         warmup=args.warmup,measured=args.measured,training=TRAIN_ARGS),indent=2))
    if args.dry_run:
        return
    import modal
    app = modal.App("gpt-lab-h100-fp8-diagnostic")
    cache = modal.Volume.from_name("gpt-lab-cache",environment_name="main")
    runs = modal.Volume.from_name("gpt-lab-runs",environment_name="main")
    image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("git","build-essential","ca-certificates")
             .pip_install("uv==0.13.6")
             .run_commands(f"git init /opt/gpt-lab",
                           f"git -C /opt/gpt-lab remote add origin {REPOSITORY}",
                           f"git -C /opt/gpt-lab fetch --depth 1 origin {SHA}",
                           f"git -C /opt/gpt-lab checkout --detach {SHA}",
                           "git -C /opt/gpt-lab -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive",
                           "uv sync --locked --default-index https://pypi.org/simple --project /opt/gpt-lab --extra gpu --extra fp8 --group dev")
             .add_local_dir(str(Path(__file__).resolve().parent),"/opt/diagnostic",ignore=["**/__pycache__/**","**/*.pyc"]))
    check = app.function(image=image,cpu=1,memory=1024,timeout=120,
                         volumes={"/cache/gpt_lab":cache.with_mount_options(read_only=True),
                                  "/runs":runs.with_mount_options(read_only=True)},serialized=True)(preflight)
    run = app.function(image=image,gpu="H100:1",cpu=16,memory=65536,timeout=3600,
                       max_containers=1,retries=0,
                       volumes={"/cache/gpt_lab":cache,"/runs":runs},
                       secrets=[modal.Secret.from_name("gpt-lab",environment_name="main",required_keys=["HF_TOKEN"])],
                       serialized=True)(worker)
    with modal.enable_output(), app.run(detach=args.detach,environment_name="main"):
        print(check.remote(args.run_name),flush=True)
        if args.detach:
            call = run.spawn(args.run_name,args.warmup,args.measured,runs)
            print(f"Submitted {call.object_id}; reports: gpt-lab-runs/{args.run_name}")
        else:
            print(run.remote(args.run_name,args.warmup,args.measured,runs))


if __name__=="__main__":
    main()
