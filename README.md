# torch-cpu-bench

Times PyTorch inference on CPU for a suite of models at 1, 2 and 4 threads, and builds an HTML report from the results.

## Run

Locally, in Docker:

```
docker build -t torch-cpu-bench .
docker run --rm --cpuset-cpus=0-3 -v "$PWD/out:/bench/out" torch-cpu-bench python run.py
```

`--models a b` runs a subset; `--timeout` sets seconds per run.

On Kubernetes, using the image built by CI (`ghcr.io/hwenchi/torch-cpu-bench:latest`):

```
kubectl apply -f k8s/job.yaml
kubectl -n torch-cpu-bench logs -f job/torch-cpu-bench
```

Results are written to the `torch-cpu-bench-results` volume. To copy them out:

```
kubectl -n torch-cpu-bench run fetch --image=busybox --restart=Never --overrides='{"spec":{"containers":[{"name":"fetch","image":"busybox","command":["sleep","300"],"volumeMounts":[{"name":"r","mountPath":"/out"}]}],"volumes":[{"name":"r","persistentVolumeClaim":{"claimName":"torch-cpu-bench-results"}}]}}'
kubectl -n torch-cpu-bench cp fetch:/out/results.json out/results.json
kubectl -n torch-cpu-bench delete pod fetch
```

## Report

```
uv run python report.py
```

Reads `out/results.json` and writes `out/report.html`.
