FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv

# Install Python 3.10 and tools
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m venv $VENV_PATH && \
    $VENV_PATH/bin/pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Set path to use venv by default
ENV PATH="$VENV_PATH/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy training code
COPY ./src /app

# Install Python dependencies inside the venv
RUN pip install -r requirements.txt

# Default command
CMD ["python", "train.py"]
