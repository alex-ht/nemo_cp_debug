# CUDA 12.3
FROM nvcr.io/nvidia/pytorch:24.02-py3

### config tags
ARG APEX_TAG=master
#ARG TE_TAG=alex-ht-patch-1
ARG MLM_TAG=core_r0.5.0
ARG NEMO_TAG=r1.23.0
ARG PYTRITON_VERSION=0.4.1
ARG PROTOBUF_VERSION=4.24.4
ARG ALIGNER_COMMIT=main

# if you get errors building TE or Apex, decrease this to 4
ARG MAX_JOBS=8

# needed in case git complains that it can't detect a valid email, this email is fake but works
RUN git config --global user.email "worker@nvidia.com"

WORKDIR /opt

# install latest apex
RUN pip uninstall -y apex && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    if [ ! -z $APEX_TAG ]; then \
        git fetch origin $APEX_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./

# place any util pkgs here
RUN pip install --upgrade-strategy only-if-needed nvidia-pytriton==$PYTRITON_VERSION
RUN pip install -U --no-deps protobuf==$PROTOBUF_VERSION
RUN pip install --upgrade-strategy only-if-needed jsonlines

# NeMo
RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    git pull && \
    if [ ! -z $NEMO_TAG ]; then \
        git fetch origin $NEMO_TAG && \
        git checkout FETCH_HEAD; \
    fi && \
    pip uninstall -y nemo_toolkit sacrebleu && \
    git cherry-pick --no-commit -X theirs \
        9940ec60058f644662809a6787ba1b7c464567ad && \
    rm -rf .git && pip install -e ".[nlp]" && \
    cd nemo/collections/nlp/data/language_modeling/megatron && make

# install xformers
ADD xformers /opt/xformers
RUN pip uninstall -y xformers && \
    cd /opt/xformers && \
    TORCH_CUDA_ARCH_LIST="7.0" pip install -v -U .

# install TransformerEngine
ADD TransformerEngine /opt/TransformerEngine
RUN pip uninstall -y transformer-engine && \
    cd TransformerEngine && \
    git submodule init && git submodule update && \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .
    
# MLM
ADD Megatron-LM /opt/Megatron-LM
RUN pip uninstall -y megatron-core && \
    cd Megatron-LM && \
    pip install -e .

# NeMo Aligner
RUN git clone https://github.com/NVIDIA/NeMo-Aligner.git && \
    cd NeMo-Aligner && \
    git pull && \
    if [ ! -z $ALIGNER_COMMIT ]; then \
        git fetch origin $ALIGNER_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install --no-deps -e .

RUN pip install sentencepiece accelerate tiktoken
WORKDIR /workspace
