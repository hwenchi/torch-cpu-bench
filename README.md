# torch-cpu-bench

Times PyTorch model inference on CPU at 1, 2 and 4 threads.

## Run

```
docker build -t torch-cpu-bench .
docker run --rm --cpuset-cpus=0-3 -v "$PWD/out:/bench/out" torch-cpu-bench python run.py
```

On Kubernetes:

```
kubectl apply -f k8s/job.yaml
```

Results go to the `torch-cpu-bench-results` volume.

## Report

```
uv run python report.py
```

Writes `out/report.html` from `out/results.json`.
