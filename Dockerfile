FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list

# Basic dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    unzip \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libjsoncpp-dev

# Install nlohmann_json 3.11.3 from source code.
# RUN git clone https://gitee.com/mirrors/json.git --branch v3.11.3 --single-branch --depth 1 && \
RUN git clone https://github.com/nlohmann/json.git --branch v3.11.3 --single-branch --depth 1 && \
    mkdir -p json/build && \
    cd json/build && \
    cmake .. -DJSON_BuildTests=OFF && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf json

# Install spdlog 1.14.0 from source code.
RUN git clone https://github.com/gabime/spdlog.git --branch v1.14.0 --single-branch --depth 1 && \
    mkdir -p spdlog/build && \
    cd spdlog/build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf spdlog

# Install Eigen3 3.3.7 from source code.
RUN git clone https://gitlab.com/libeigen/eigen.git --branch 3.3.7 --single-branch --depth 1 && \
    mkdir -p eigen/build && \
    cd eigen/build && \
    cmake .. && \
    make install && \
    cd ../.. && \
    rm -rf eigen

# Install Sophus 1.22.10 from source code.
RUN git clone https://github.com/strasdat/Sophus.git --branch 1.22.10 --single-branch --depth 1 && \
    mkdir -p Sophus/build && \
    cd Sophus/build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf Sophus

# Install Ceres Solver 2.2.0 from source code.
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 2.2.0 --single-branch --depth 1 && \
    mkdir -p ceres-solver/build && \
    cd ceres-solver/build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf ceres-solver

RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
