import argparse
import datetime
import glob
import json
import os
import platform
import subprocess
import sys
import time

THREADS = [1, 2, 4]
WARMUP = 1
REPEATS = 20

# CPU flags that decide which fast paths the math libraries can use.
VECTOR_FLAGS = [
    "sse4_2", "avx", "avx2", "fma", "f16c",
    "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512_bf16", "avx512_fp16",
    "amx_tile", "amx_bf16", "amx_int8",
    "asimd", "asimddp", "sve", "sve2", "bf16", "i8mm",
]


def worker(name, threads):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from models import MODELS

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")

    entry = next(m for m in MODELS if m["name"] == name)
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


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def hardware():
    import torch

    cpuinfo = (read("/proc/cpuinfo") or "").splitlines()
    meminfo = (read("/proc/meminfo") or "").splitlines()

    def cpuinfo_field(key):
        return next((l.split(":", 1)[1].strip() for l in cpuinfo if l.split(":", 1)[0].strip() == key), None)

    flags = set((cpuinfo_field("flags") or cpuinfo_field("Features") or "").split())
    mem_kb = next((int(l.split()[1]) for l in meminfo if l.startswith("MemTotal")), None)

    # cgroup v2 limits: "max 100000" means unlimited, "400000 100000" means 4 CPUs.
    cpu_max = read("/sys/fs/cgroup/cpu.max")
    cpu_limit = None
    if cpu_max and not cpu_max.startswith("max"):
        quota, period = cpu_max.split()
        cpu_limit = int(quota) / int(period)
    mem_max = read("/sys/fs/cgroup/memory.max")
    mem_limit_gb = round(int(mem_max) / 2**30, 1) if mem_max and mem_max != "max" else None

    caches = {}
    for index in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cache/index*")):
        level, kind, size = read(f"{index}/level"), read(f"{index}/type"), read(f"{index}/size")
        if level and kind and size:
            caches[f"L{level} {kind.lower()}"] = size

    build = torch.__config__.show()
    return {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "node": os.environ.get("NODE_NAME"),
        "machine": platform.machine(),
        "kernel": platform.release(),
        # x86 reports a model name; ARM reports only a part number.
        "cpu_model": cpuinfo_field("model name") or (cpuinfo_field("CPU part") and f"ARM part {cpuinfo_field('CPU part')}"),
        "cpu_vendor": cpuinfo_field("vendor_id") or cpuinfo_field("CPU implementer"),
        "cpu_mhz": cpuinfo_field("cpu MHz"),
        "hypervisor": "hypervisor" in flags,
        "vector_flags": [f for f in VECTOR_FLAGS if f in flags],
        "caches": caches,
        "cpus_visible": os.cpu_count(),
        "cpus_affinity": len(os.sched_getaffinity(0)),
        "cpu_limit": cpu_limit,
        "mem_total_gb": round(mem_kb / 2**20, 1) if mem_kb else None,
        "mem_limit_gb": mem_limit_gb,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "blas": build.split("BLAS_INFO=")[1].split(",")[0] if "BLAS_INFO=" in build else None,
        "mkldnn": torch.backends.mkldnn.is_available(),
    }


def main():
    from models import MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="subset of model names (default: all)")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per run")
    parser.add_argument("--out", default="out/results.json")
    args = parser.parse_args()

    cpus = len(os.sched_getaffinity(0))
    if cpus < max(THREADS):
        sys.exit(f"need at least {max(THREADS)} CPUs, only {cpus} available (run docker with --cpuset-cpus=0-{max(THREADS) - 1})")

    selected = [m for m in MODELS if not args.models or m["name"] in args.models]
    results = {
        "hardware": hardware(),
        "config": {"threads": THREADS, "warmup": WARMUP, "repeats": REPEATS, "timeout_s": args.timeout},
        "runs": [],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total = len(selected) * len(THREADS)
    i = 0
    for m in selected:
        for t in THREADS:
            i += 1
            row = {"model": m["name"], "family": m["family"], "domain": m["domain"], "task": m["task"], "threads": t}
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
