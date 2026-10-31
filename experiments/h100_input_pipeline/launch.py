"""User launch only: CPU preflight first, then one worker with one H100."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from recipe import SHA, REPOSITORY, train_args, BATCH_SIZES, ORDER


def gpu_worker(run_name, warmup, measured, runs, harness_sha, batch_sizes):
    import os
    import sys
    import json
    from pathlib import Path
    import subprocess
    sys.path.insert(0,"/opt/diagnostic/h100_input_pipeline")
    from recipe import BATCH_SIZES, combine_reports
    root = Path("/runs")/run_name
    if not (root/"cpu-preflight.json").is_file():
        raise RuntimeError("CPU preflight must complete before GPU work")
    if (root/"worker.log").exists():
        raise FileExistsError("Refusing to overwrite an existing GPU run")
    environment = os.environ | {"GPTLAB_CACHE_DIR":"/cache/gpt_lab", "GPTLAB_DTYPE":"bfloat16",
        "PYTHONUNBUFFERED":"1", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
        "PYTORCH_ALLOC_CONF":"expandable_segments:True", "INPUT_HARNESS_SHA":harness_sha}
    try:
        with (root/"worker.log").open("x",buffering=1) as log:
            reports = {}
            for batch_size in batch_sizes:
                output = root/f"batch-{batch_size}"
                output.mkdir(exist_ok=False)
                environment["TORCHINDUCTOR_CACHE_DIR"] = f"/opt/gpt-lab/.tmp/codex/h100-input-pipeline/compiler_cache/batch-{batch_size}"
                command = ["/opt/gpt-lab/.venv/bin/python","/opt/diagnostic/h100_input_pipeline/runner.py",
                           "--output",str(output),"--preflight",str(root/"cpu-preflight.json"),
                           "--device-batch-size",str(batch_size),"--warmup",str(warmup),"--measured",str(measured)]
                with subprocess.Popen(command,cwd="/opt/gpt-lab",env=environment,
                                      stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as process:
                    for line in process.stdout:
                        print(line,end="",flush=True)
                        log.write(line)
                    code = process.wait()
                if code:
                    raise RuntimeError(f"Batch {batch_size} failed ({code}); inspect {run_name}/worker.log and partial reports")
                reports[str(batch_size)] = json.loads((output/"report.json").read_text())
                summary = combine_reports(reports) if len(reports) == len(BATCH_SIZES) else {}
                (root/"report.json").write_text(json.dumps(dict(batches=reports,complete=len(reports)==len(batch_sizes),device_batch_sizes=batch_sizes,**summary),indent=2)+"\n")
    finally:
        runs.commit()
    return str(root/"report.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-batch-size",type=int,choices=BATCH_SIZES,help="Run only this shape (default: both)")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--measured", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="Run only the real-cache CPU preflight")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}",args.run_name):
        parser.error("run-name must contain 1–100 letters/digits/_/-")
    if args.warmup<25 or args.measured<50 or args.warmup+args.measured>2317:
        parser.error("Need >=25 warmup, >=50 measured updates within horizon 2317")
    batch_sizes = [args.device_batch_size] if args.device_batch_size is not None else list(BATCH_SIZES)
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    harness_sha = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    print(json.dumps(dict(production_sha=SHA,harness_sha=harness_sha,repository=REPOSITORY,
                         gpu="H100:1",workers=1,cpu_preflight_first=True,cpu_only=args.cpu_only,
                         warmup=args.warmup,measured=args.measured,order=ORDER,
                         training={str(size):train_args(size) for size in batch_sizes},
                         device_batch_sizes=batch_sizes,environment="main"),indent=2),flush=True)
    if args.dry_run:
        return
    dirty = subprocess.check_output(["git","-C",str(root),"status","--porcelain","--",
                                    "experiments/h100_input_pipeline","experiments/h100_fp8_diagnostic"],text=True)
    if dirty:
        parser.error("Commit diagnostic files before launching so harness_sha identifies uploaded code")
    import modal
    app = modal.App("gpt-lab-h100-input-pipeline")
    cache = modal.Volume.from_name("gpt-lab-cache",environment_name="main")
    runs = modal.Volume.from_name("gpt-lab-runs",environment_name="main")
    image = (modal.Image.debian_slim(python_version="3.12")
        .apt_install("git","build-essential","ca-certificates")
        .pip_install("uv==0.13.6")
        .env({"GPTLAB_CACHE_DIR":"/cache/gpt_lab", "GPTLAB_DTYPE":"bfloat16",
              "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "TIKTOKEN_CACHE_DIR":"/opt/tiktoken-cache"})
        .run_commands("git init /opt/gpt-lab",
                      f"git -C /opt/gpt-lab remote add origin {REPOSITORY}",
                      f"git -C /opt/gpt-lab fetch --depth 1 origin {SHA}",
                      f"git -C /opt/gpt-lab checkout --detach {SHA}",
                      "git -C /opt/gpt-lab -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive",
                      "uv sync --locked --default-index https://pypi.org/simple --project /opt/gpt-lab --extra gpu --group dev",
                      # Tokenizer assets only, never download or replace production dataset shards.
                      "/opt/gpt-lab/.venv/bin/python -c 'import tiktoken; tiktoken.get_encoding(\"gpt2\")'")
        .add_local_dir(str(here),"/opt/diagnostic/h100_input_pipeline",ignore=["**/__pycache__/**","**/*.pyc"])
        .add_local_dir(str(here.parent/"h100_fp8_diagnostic"),"/opt/diagnostic/h100_fp8_diagnostic",ignore=["**/__pycache__/**","**/*.pyc"]))
    # Use the uv environment for both functions, including CPU-side production imports.
    def cpu_entry(run_name, warmup, measured, runs, batch_sizes):
        import subprocess
        import json
        from pathlib import Path
        script = "/opt/diagnostic/h100_input_pipeline/preflight_entry.py"
        subprocess.run(["/opt/gpt-lab/.venv/bin/python",script,"--run-name",run_name,
                        "--warmup",str(warmup),"--measured",str(measured),
                        "--device-batch-sizes",*map(str,batch_sizes)],check=True,cwd="/opt/gpt-lab")
        runs.commit()
        return json.loads((Path("/runs")/run_name/"cpu-preflight.json").read_text())
    check = app.function(image=image,cpu=16,memory=65536,timeout=1800,
                         volumes={"/cache/gpt_lab":cache.with_mount_options(read_only=True),"/runs":runs},
                         serialized=True)(cpu_entry)
    run = app.function(image=image,gpu="H100:1",cpu=16,memory=65536,timeout=3600,
                       min_containers=0,buffer_containers=0,max_containers=1,retries=0,
                       volumes={"/cache/gpt_lab":cache.with_mount_options(read_only=True),"/runs":runs},
                       serialized=True)(gpu_worker)
    with modal.enable_output(), app.run(environment_name="main"):
        result = check.remote(args.run_name,args.warmup,args.measured,runs,batch_sizes)
        print(json.dumps(result,indent=2),flush=True)
        if not args.cpu_only:
            print(run.remote(args.run_name,args.warmup,args.measured,runs,harness_sha,batch_sizes))


if __name__ == "__main__":
    main()
