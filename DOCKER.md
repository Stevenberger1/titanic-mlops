# Running the Titanic API in Docker

This document is the step-by-step runbook for building and running the
containerised API on any machine with a working Docker daemon.

The dev machine that *wrote* this code may or may not have Docker
installed. The image is designed to be built and run anywhere — local
Linux/macOS, a CI runner, a cloud VM, a colleague's laptop — without
any other tooling on the host.

## Prerequisites on the host

- **Docker** ≥ 24.0 (`docker --version` to check).
- A copy of this repository.
- Nothing else. No Python, no `uv`, no MLflow.

If `docker` is not on the host yet, install Docker Engine (Linux) or
Docker Desktop (macOS / Windows). Docker Desktop on Windows requires
CPU virtualisation enabled in BIOS plus WSL2.

## One-time prep on the dev machine (before pushing the image)

The image bakes in the model that was tagged `@production` in the MLflow
registry at build time. That model has to be exported from the dev
machine *before* `docker build`:

```bash
uv run python scripts/export_production_model.py
```

This populates `models/exported/` with a relocatable MLmodel folder plus
a `metadata.json` describing the version, alias, and run id. The
`Dockerfile` copies that folder into the image.

Without this step, `docker build` will fail at the
`COPY models/exported/ ./models/exported/` line because the source
directory won't exist.

## Build

From the project root:

```bash
docker build -t titanic-api:latest .
```

What the flags mean:
- `-t titanic-api:latest` — tag the resulting image as
  `titanic-api:latest`. Without `-t`, the image is anonymous and only
  addressable by its sha256 digest. Tags are how humans refer to images.
- `.` — the **build context**. Docker tarballs this directory (minus
  `.dockerignore` entries) and ships it to the daemon. The `Dockerfile`
  is read from the root of the context unless `-f` overrides it.

Expected first-time build duration: 2–5 minutes (dependency install
dominates). Subsequent rebuilds after editing only Python files: ~10
seconds (the dep layer is cached).

## Run

```bash
docker run --rm -p 8000:8000 --name titanic-api titanic-api:latest
```

Flag breakdown:
- `--rm` — automatically remove the container when it exits. Without
  this, stopped containers linger and `docker ps -a` fills up over
  time.
- `-p 8000:8000` — publish container port 8000 to host port 8000.
  Format: `HOST:CONTAINER`. If 8000 is taken on the host, use
  `-p 8080:8000` and visit `http://localhost:8080` instead.
- `--name titanic-api` — friendly name for `docker stop`, `docker logs`,
  etc. Without it, Docker generates a random name like
  `eager_einstein`.

You should see uvicorn startup output. The API is reachable at:

- `http://localhost:8000/docs` — Swagger UI
- `http://localhost:8000/health`
- `http://localhost:8000/model-info`
- `http://localhost:8000/predict` (POST)

## Verify it's working

In a second terminal:

```bash
curl http://localhost:8000/health
# expected: {"status":"ok"}

curl http://localhost:8000/model-info
# expected: {"name":"titanic-classifier","version":"1","alias":"production","loaded_at":"..."}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"}'
# expected: {"survived":0,"survival_probability":0.0xxx}
```

## Stop

In the terminal running `docker run`, press `Ctrl+C`. Because of
`--rm`, the container is removed automatically.

If you ran it detached (`-d`), stop and remove with:

```bash
docker stop titanic-api
```

(`--rm` from the run command still handles removal.)

## Running in the background

For a longer-lived session add `-d` (detached):

```bash
docker run -d --rm -p 8000:8000 --name titanic-api titanic-api:latest

# Inspect:
docker ps
docker logs -f titanic-api      # follow the logs
docker exec -it titanic-api sh  # open a shell inside the container

# Stop:
docker stop titanic-api
```

## Updating to a new model version

The model is baked in at build time, so deploying a newer registry
version is a rebuild:

1. On the dev machine, promote the new version to `@production` in the
   MLflow UI (move the alias from v1 → v2, for example).
2. Re-run `uv run python scripts/export_production_model.py`. This
   overwrites `models/exported/` with the new version's artifacts.
3. Rebuild: `docker build -t titanic-api:latest .`
4. Stop the old container, run the new image.

In Phase 7 (cloud deployment) this becomes a CI-driven pipeline rather
than manual steps.

## Inspecting the image

Useful commands when something looks off:

```bash
docker images titanic-api       # list image tags + sizes
docker history titanic-api:latest  # per-layer view (handy for size debugging)
docker inspect titanic-api:latest  # full JSON metadata
```

## Cleaning up

```bash
docker rmi titanic-api:latest   # remove the image
docker system prune             # remove unused images, networks, containers
docker system prune -a --volumes  # nuclear option: remove everything unused
```

## Why the image doesn't talk to MLflow at runtime

The local MLflow store (`mlruns/`) uses file-based storage with
absolute paths from the dev machine. Those paths don't exist inside a
Linux container, so a live registry lookup would fail.

Instead we resolve `@production` once on the dev machine (via
`scripts/export_production_model.py`), copy the resulting relocatable
model folder into the image, and have `app/main.py` load from that
local path when the `MODEL_PATH` environment variable is set.

In a more elaborate setup the registry would live on a remote MLflow
tracking server, and the container would resolve aliases over HTTP at
startup. That's the right pattern for production-scale deployments but
not necessary for this learning project.
