# nemo_cp_debug
**scripts**
```sh
sbatch convert_llama_ckpt.sh
sbatch sft_training.sh
sbatch sft_training_cp.sh
```

**docker image**
```sh
docker pull alexht/nemo-aligner:debug
```
base image: `nvcr.io/nvidia/pytorch:24.02-py3`

- only support SM_70 (V100)

**WANDB**

[https://wandb.ai/alex-ht/llama3-cp-debug](https://wandb.ai/alex-ht/llama3-cp-debug)