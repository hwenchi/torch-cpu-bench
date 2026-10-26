import argparse
import json
import os
import subprocess
import sys
import time

THREADS = [1, 2, 4, 8]
WARMUP = 1
REPEATS = 5


def worker(name, threads):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from suite import SUITE

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")

    entry = next(m for m in SUITE if m["name"] == name)
    torch.manual_seed(0)
    model = entry["build"]().eval()
    x = entry["input"]()
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert x.device.type == "cpu"

    times_ms = []
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)
        for _ in range(REPEATS):
            start = time.perf_counter()
            model(x)
            times_ms.append((time.perf_counter() - start) * 1000)

        with FlopCounterMode(display=False) as counter:
            model(x)

    print(json.dumps({
        "mean_ms": sum(times_ms) / len(times_ms),
        "median_ms": sorted(times_ms)[len(times_ms) // 2],
        "times_ms": times_ms,
        "params": sum(p.numel() for p in model.parameters()),
        "gflops": counter.get_total_flops() / 1e9,
        "input_shape": list(x.shape),
        "torch_threads": torch.get_num_threads(),
    }))


def main():
    from suite import SUITE

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="subset of model names (default: all)")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per run")
    parser.add_argument("--out", default="out/results.json")
    args = parser.parse_args()

    cpus = len(os.sched_getaffinity(0))
    if cpus < max(THREADS):
        sys.exit(f"need at least {max(THREADS)} CPUs, only {cpus} available (run docker with --cpuset-cpus=0-{max(THREADS) - 1})")

    models = [m for m in SUITE if not args.models or m["name"] in args.models]
    results = {
        "config": {"threads": THREADS, "warmup": WARMUP, "repeats": REPEATS, "timeout_s": args.timeout},
        "runs": [],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total = len(models) * len(THREADS)
    i = 0
    for m in models:
        for t in THREADS:
            i += 1
            row = {"model": m["name"], "family": m["family"], "domain": m["domain"], "threads": t}
            env = dict(os.environ, OMP_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t))
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, __file__, "--worker", m["name"], str(t)],
                    env=env, capture_output=True, text=True, timeout=args.timeout,
                )
                lines = proc.stdout.strip().splitlines()
                if proc.returncode == 0 and lines:
                    row.update(status="ok", **json.loads(lines[-1]))
                else:
                    stderr = proc.stderr.strip()
                    row.update(status="error", error=stderr.splitlines()[-1] if stderr else f"exit code {proc.returncode}")
            except subprocess.TimeoutExpired:
                row.update(status="timeout", error=f"exceeded {args.timeout:g}s")
            row["wall_s"] = round(time.perf_counter() - start, 2)

            results["runs"].append(row)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            detail = f"{row['mean_ms']:9.1f} ms" if row["status"] == "ok" else f"{row['status']}: {row['error']}"
            print(f"[{i}/{total}] {m['name']:16s} threads={t}  {detail}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        main()
