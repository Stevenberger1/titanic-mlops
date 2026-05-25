# syntax=docker/dockerfile:1.7
# ----------------------------------------------------------------------------
# Titanic Classifier API — container image
#
# Build (from project root, after running scripts/export_production_model.py):
#     docker build -t titanic-api:latest .
#
# Run:
#     docker run --rm -p 8000:8000 titanic-api:latest
#
# The image bakes in the @production model exported to ./models/exported/,
# so the container has zero runtime dependency on a live MLflow store.
# ----------------------------------------------------------------------------

# Base: a slim Debian-based Python image. "slim" = no docs/man pages/build
# tooling. Smaller image, faster pulls. We don't need a full distro.
FROM python:3.13-slim-bookworm

# Pull the uv binary from Astral's official image. COPY --from=<image> is
# a multi-stage trick: we don't inherit their whole OS, we just grab the
# uv executable. Tiny, fast, no extra layers.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Make Python output unbuffered so `docker logs` shows prints immediately
# instead of waiting for the stdout buffer to flush.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Copy dependency manifests FIRST and install deps in their own layer. This
# is the Docker layer-cache trick: as long as pyproject.toml and uv.lock
# haven't changed, rebuilding the image skips the slow `uv sync` step and
# reuses the cached dependency layer. Source edits become near-instant.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the source. Changes here invalidate only this layer onward.
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY models/exported/ ./models/exported/

# Install the project itself now that the source is present.
RUN uv sync --frozen --no-dev

# Tell the FastAPI app where the baked-in model lives. The presence of this
# env var is what flips main.py from registry-mode to local-path mode.
ENV MODEL_PATH=/app/models/exported

# Document which port the app listens on. EXPOSE is metadata only — it
# doesn't actually publish the port; `docker run -p` does that.
EXPOSE 8000

# Use the exec form (JSON array) so SIGTERM reaches uvicorn directly when
# the container stops, instead of going to a wrapping /bin/sh process.
# --host 0.0.0.0 binds to all interfaces — required for the port to be
# reachable from outside the container.
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
