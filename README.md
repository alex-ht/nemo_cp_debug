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
- only support SM_70 (V100)

**WANDB**

[https://wandb.ai/alex-ht/llama3-cp-debug](https://wandb.ai/alex-ht/llama3-cp-debug)