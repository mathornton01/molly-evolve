FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch + core deps
RUN pip install --no-cache-dir \
    torch>=2.0 \
    transformers>=4.20 \
    datasets \
    peft \
    bitsandbytes \
    scipy \
    numpy \
    accelerate \
    sentencepiece \
    protobuf

# Install molly-evolution
WORKDIR /app
COPY . /app/
RUN pip install --no-cache-dir -e .

# Default: show help
ENTRYPOINT ["python", "-m", "molly_evolution.cli"]
CMD ["--help"]
