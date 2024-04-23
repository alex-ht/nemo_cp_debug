#!/bin/bash
#SBATCH --job-name=alex-ckpt           ## job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --output=log/%x-%j.out        ## output log file name
#SBATCH --account=****
#SBATCH --partition=****       


module purge
module load singularity

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

SINGULARITY_RUN="singularity run \
    --nv \
    --no-mount home \
    --writable-tmpfs \
    --bind /work/u4005115/models \
    --bind $(pwd) \
    /home/u9824269/LLM/image/nemo-aligner.sif \
    "

CMD="python -u convert_llama_hf_to_nemo.py \
    --input_name_or_path /work/u4005115/models/llama/Meta-Llama-3-70B \
    --output_path $(pwd)/llama3-70b-16-mixed.nemo \
    --hparams_file conf/megatron_llama_config.yaml \
    --precision 16-mixed \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOBID $SINGULARITY_RUN $CMD

echo "END TIME: $(date)"