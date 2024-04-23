#!/bin/bash
#SBATCH --job-name=alex-nemo-sft-cp     ## job name
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=8
#SBATCH --output=log/%x-%j.out        ## output log file name
#SBATCH --account=****
#SBATCH --partition=****
#SBATCH --overcommit

module purge
module load singularity
export TMPDIR=$(mktemp -p /home/u9824269/LLM/llama3/train -d)
trap 'rm -rf -- "$TMPDIR"' EXIT
export NVTE_MASKED_SOFTMAX_FUSION=1
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

export MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0

GPFS=/home/u9824269/LLM/nemo/NeMo-Aligner
export PYTHONPATH="${GPFS}:${PYTHONPATH}"

RESULTS_DIR="$(pwd)/result_dir"

OUTFILE="${RESULTS_DIR}/sft-%j_%t.out"
ERRFILE="${RESULTS_DIR}/sft-%j_%t.err"
mkdir -p ${RESULTS_DIR}

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    -o $OUTFILE \
    -e $ERRFILE \
    "

SINGULARITY_RUN="singularity run \
    --nv \
    --writable-tmpfs \
    --bind /work/u4005115/models \
    --bind $(pwd) \
    /home/u9824269/LLM/image/nemo-aligner.sif \
    "

TRAIN_DATA_PATH="/home/u9824269/LLM/nemo/databricks-dolly-15k-output.jsonl"
VALID_DATA_PATH="/home/u9824269/LLM/nemo/databricks-dolly-15k-output.jsonl"

PRETRAINED_ACTOR_NEMO_FILE="/home/u9824269/LLM/llama3/train/llama3-70b-16-mixed.nemo"

PROJECT="llama3-cp-debug" # if you want to use wandb

cd ${GPFS}
CMD="python -u examples/nlp/gpt/train_gpt_sft.py \
   trainer.precision=16-mixed \
   trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
   trainer.devices=-1 \
   trainer.sft.limit_val_batches=1 \
   trainer.sft.val_check_interval=12 \
   trainer.sft.save_interval=12 \
   model.megatron_amp_O2=True \
   model.restore_from_path=${PRETRAINED_ACTOR_NEMO_FILE} \
   model.optim.lr=5e-6 \
   model.optim.name=distributed_fused_adam \
   model.answer_only_loss=True \
   model.data.num_workers=0 \
   model.data.train_ds.micro_batch_size=1 \
   model.data.train_ds.global_batch_size=8 \
   model.data.train_ds.file_path=${TRAIN_DATA_PATH} \
   model.data.train_ds.max_seq_length=8192 \
   model.data.train_ds.add_eos=True \
   model.data.train_ds.add_bos=True \
   model.data.validation_ds.max_seq_length=8192 \
   model.data.validation_ds.micro_batch_size=1 \
   model.data.validation_ds.global_batch_size=128 \
   model.data.validation_ds.file_path=${VALID_DATA_PATH} \
   model.data.validation_ds.add_bos=True \
   model.data.validation_ds.add_eos=True \
   model.tensor_model_parallel_size=8 \
   model.pipeline_model_parallel_size=4 \
   model.sequence_parallel=True \
   model.activations_checkpoint_granularity=selective \
   model.activations_checkpoint_method=uniform \
   +model.context_parallel_size=2 \
   exp_manager.create_wandb_logger=True \
   exp_manager.explicit_log_dir=${RESULTS_DIR} \
   exp_manager.wandb_logger_kwargs.project=${PROJECT} \
   exp_manager.wandb_logger_kwargs.name=dolly_sft_run_tp8_cp \
   exp_manager.resume_if_exists=True \
   exp_manager.resume_ignore_no_checkpoint=True \
   exp_manager.create_checkpoint_callback=True \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.checkpoint_callback_params.monitor=validation_loss \
   "

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=$(pwd)/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

clear; srun $SRUN_ARGS --jobid $SLURM_JOBID $SINGULARITY_RUN $CMD