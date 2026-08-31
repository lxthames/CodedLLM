FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        gdb \
        ninja-build \
        clang-format \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*
