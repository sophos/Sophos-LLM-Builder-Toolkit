################################################################################
# BASE IMAGE
# Choose latest torch version
# Get cuda version from table - https://github.com/pytorch/pytorch/blob/main/RELEASE.md
# Get cuda base image from - https://hub.docker.com/r/nvidia/cuda/tags
# Check if installed compilers work with corresponding nccl version here - https://gist.github.com/ax3l/9489132
################################################################################

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG PYTORCH=2.2.0
# Not using latest version until the following issue is resolved
# https://github.com/microsoft/DeepSpeed/issues/5422
ARG DEEPSPEED=0.14.4
ARG TRANSFORMERS=4.42.4
ARG TRL=0.9.6
ARG ACCELERATE=0.32.1
ARG PEFT=0.11.1
ARG CUDA=cu121
ARG CUDNN_VERSION=8.9.2.26

ENV TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV _GLIBCXX_USE_CXX11_ABI=0

RUN apt -y update

RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc ninja-build \
        libcudnn8=$CUDNN_VERSION-1+cuda12.1 \
        libcudnn8-dev=$CUDNN_VERSION-1+cuda12.1 \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-dev libaio-dev libhwloc-dev libtool && \
        apt clean

ENV PYTHON_VERSION=3
RUN apt-get install -y python3 python3-dev && \
        rm -f /usr/bin/python && \
        ln -s /usr/bin/python3 /usr/bin/python && \
        curl -O https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py && \
        pip install --upgrade pip && \
        python -V && pip -V
RUN pip install --no-cache-dir pyyaml
RUN pip install --no-cache-dir ipython

RUN add-apt-repository ppa:git-core/ppa -y && \
        apt-get update && \
        apt-get install -y git && \
        git --version

RUN pwd

################################################################################
# Add Elastic Fabric Adaptor (EFA) for SageMaker
# This reduces the communication bottleneck between nodes
# See the following for more information about EFA
# https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-efa.html
# https://dataintegration.info/training-large-language-models-on-amazon-sagemaker-best-practices
# https://github.com/aws/deep-learning-containers/blob/master/pytorch/training/docker/2.0/py3/cu121/Dockerfile.gpu

# See the following recommended library versions for SageMaker p4 and p5 instances:
# https://github.com/aws/aws-ofi-nccl/blob/master/doc/efa-env-var.md
# cuda>=12.0
# nccl>=2.18.0 (recommend at least 2.18.5) - Keep what's in the default CUDA umage
# aws-ofi-nccl>=1.7.2 (recommend at least 1.7.3)
# efa-installer>=1.29.0 (to avoid nccl>=2.19.0 raising libfabric errors)
################################################################################

ENV OPEN_MPI_PATH=/opt/amazon/openmpi/
ENV EFA_VERSION=1.31.0
ENV BRANCH_OFI=1.9.1-aws
ENV CUDA_HOME=/usr/local/cuda
ENV NCCL_VERSION=2.21.5
ENV GDRCOPY_VERSION=2.4.1

################################################################################
# EFA and MPI SETUP
################################################################################

RUN mkdir /tmp/efa && \
    cd /tmp/efa && \
    curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod -g && \
    rm -rf /tmp/efa && \
    rm -rf /tmp/aws-efa-installer-${EFA_VERSION}.tar.gz && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="$OPEN_MPI_PATH/bin:$PATH"
ENV LD_LIBRARY_PATH="$OPEN_MPI_PATH/lib/:$LD_LIBRARY_PATH"

WORKDIR /root

# Configure Open MPI and configure NCCL parameters
RUN mv $OPEN_MPI_PATH/bin/mpirun $OPEN_MPI_PATH/bin/mpirun.real && \
    echo '#!/bin/bash' > $OPEN_MPI_PATH/bin/mpirun && \
    echo "${OPEN_MPI_PATH}/bin/mpirun.real --allow-run-as-root \"\$@\"" >> $OPEN_MPI_PATH/bin/mpirun && \
    chmod a+x $OPEN_MPI_PATH/bin/mpirun && \
    echo "hwloc_base_binding_policy = none" >> $OPEN_MPI_PATH/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> $OPEN_MPI_PATH/etc/openmpi-mca-params.conf && \
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf  && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# Install OpenSSH for MPI to communicate between containers, allow OpenSSH to talk to containers without asking for confirmation
RUN apt-get update && \
    apt-get install -y  --allow-downgrades --allow-change-held-packages --no-install-recommends && \
    apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Configure OpenSSH so that nodes can communicate with each other
RUN mkdir -p /var/run/sshd && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
    mkdir -p /root/.ssh/ && \
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    printf "Host *\n StrictHostKeyChecking no\n" >> /root/.ssh/config

RUN mkdir -p /etc/pki/tls/certs && cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

# Removing the cache as it is needed for security verification
RUN rm -rf /root/.cache | true

WORKDIR /workspace

################################################################################
# OFI, NCCL-TEST SETUP
################################################################################

RUN apt-get update && apt-get install -y autoconf

RUN cd /tmp \
 && git clone https://github.com/NVIDIA/gdrcopy.git -b v${GDRCOPY_VERSION} \
 && cd gdrcopy \
 && sed -ie '12s@$@ -L $(CUDA)/lib/stubs@' tests/Makefile \
 && CUDA=${CUDA_HOME} make install \
 && rm -rf /tmp/gdrcopy

RUN cd /tmp \
  && git clone https://github.com/NVIDIA/nccl.git -b v${NCCL_VERSION}-1 \
  && cd nccl \
  && make -j64 src.build BUILDDIR=/usr/local \
  && rm -rf /tmp/nccl
# preload to system nccl for PyTorch to use instead of its statically linked NCCL
ENV LD_PRELOAD="/usr/local/lib/libnccl.so"

RUN cd /tmp && \
    git clone https://github.com/aws/aws-ofi-nccl.git -b v${BRANCH_OFI} && \
    cd aws-ofi-nccl && \
    ./autogen.sh && \
    ./configure --with-libfabric=/opt/amazon/efa \
       --with-mpi=/opt/amazon/openmpi \
       --with-cuda=${CUDA_HOME} \
       --with-nccl=/usr/local --prefix=/usr/local  && \
    make && make install && \
    rm -rf /tmp/aws-ofi-nccl
  
RUN cd /tmp && \
    git clone https://github.com/NVIDIA/nccl-tests && \
    cd nccl-tests && \
    make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=${CUDA_HOME} NCCL_HOME=/usr/local && \
    rm -rf /tmp/nccl-tests

################################################################################
# Install training and inference libraries in the following order:
# torch
# transformers
# accelerate
# peft
# trl
# apex 
# deepspeed 
################################################################################

ARG REF=v$TRANSFORMERS
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN python3 -m pip uninstall -y torch torchvision torchaudio

# If not the newest version of PyTorch then,
# install numpy<2.0: https://github.com/intel/intel-xpu-backend-for-triton/issues/1386
RUN python3 -m pip install --no-cache-dir numpy==1.26.4

RUN git clone https://github.com/pytorch/pytorch.git -b v$PYTORCH && \
    cd pytorch && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    NCCL_INCLUDE_DIR="/usr/local/include/" NCCL_LIB_DIR="/usr/local/lib/" USE_SYSTEM_NCCL=1 python3 setup.py develop && \
    rm -rf pytorch

# RUN python3 -m pip install --no-cache-dir -U torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir ./transformers[deepspeed-testing]

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@v$ACCELERATE#egg=accelerate

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft@v$PEFT#egg=peft

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/trl@v$TRL#egg=trl

RUN python3 -m pip uninstall -y transformer-engine

RUN python3 -m pip uninstall -y torch-tensorrt

RUN python3 -m pip install --no-cache-dir ninja

RUN python3 -m pip uninstall -y apex
RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && python3 -m pip install --no-cache-dir packaging && python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
RUN rm -rf ./apex

RUN git clone https://github.com/Dao-AILab/flash-attention
RUN cd flash-attention && python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./
RUN rm -rf ./flash-attention

RUN python3 -m pip uninstall -y deepspeed
# RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 python3 -m pip install deepspeed==$DEEPSPEED --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1
RUN python3 -m pip install deepspeed==$DEEPSPEED --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1

RUN cd transformers && python3 setup.py develop && rm -rf ./transformers

RUN python3 -c "from deepspeed.launcher.runner import main"

################################################################################
# SageMaker Specific Libraries and Function Definitions
# Specific to SageMaker script mode as seen here:
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/custom-training-containers/script-mode-container-2/docker/Dockerfile
################################################################################

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN python3 -m pip install --no-cache-dir -U \
    smclarify \
    "sagemaker>=2,<3" \
    "sagemaker-experiments<1" \
    sagemaker-pytorch-training \
    sagemaker-training \
    smdebug