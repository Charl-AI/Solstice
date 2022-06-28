ARG PYTHON_VERSION=3.9
FROM mcr.microsoft.com/vscode/devcontainers/python:${PYTHON_VERSION}-bullseye

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# need to repeat after FROM statement
ARG PYTHON_VERSION

# Create virtual environment and add to path (this is equivalent to activating it)
ENV VIRTUAL_ENV=/opt/venv
RUN python${PYTHON_VERSION} -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install poetry in the virtual environment (we install deps with PostCreateCommand)
RUN pip3 --disable-pip-version-check --no-cache-dir install poetry \
    && chmod -R 777 /opt/venv
