FROM python:3.13-slim-bookworm

LABEL qfbench2.interface_version="2.0"
LABEL agenthon.track="T2" agenthon.agent="t2-solution"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4

WORKDIR /opt/t2-solution
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 1000 runner
USER runner
WORKDIR /work

