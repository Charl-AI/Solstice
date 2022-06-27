FROM mcr.microsoft.com/vscode/devcontainers/python:3.9-bullseye

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and add to path (this is equivalent to activating it)
ENV VIRTUAL_ENV=/opt/venv
RUN python3.9 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install poetry in the virtual environment (we install deps with PostCreateCommand)
RUN pip3 --disable-pip-version-check --no-cache-dir install poetry \
    && chmod -R 777 /opt/venv
