# Development image for agent-traces (Python CLI). Dependencies live in the image (/opt/venv);
# the source is bind-mounted at /workspace and installed in editable mode, so code changes need no rebuild.
FROM python:3.12.16-slim-trixie

ARG UID=1000
ARG GID=1000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN groupadd --gid "$GID" dev \
 && useradd --uid "$UID" --gid "$GID" --create-home --shell /bin/bash dev \
 && python -m venv /opt/venv

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --editable ".[dev]"
# the real source arrives through the bind mount; leave the mount point empty
RUN find /workspace -mindepth 1 -delete

USER dev
CMD ["sleep", "infinity"]
