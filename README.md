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

**other things**
- `train_grad_norm=nan`
```
Training steps:   0%|          | 0/1876 [01:54<?, ?it/s, train_grad_norm=nan, train_lr=0, train_loss=1.57, train_consumed_samples=8, train_step_time=114, train_epoch=1]
Training steps:   0%|          | 1/1876 [01:54<59:31:20, 114.28s/it, train_grad_norm=nan, train_lr=0, train_loss=1.57, train_consumed_samples=8, train_step_time=114, train_epoch=1]
Training steps:   0%|          | 1/1876 [02:34<59:31:20, 114.28s/it, train_grad_norm=nan, train_lr=5e-7, train_loss=1.44, train_consumed_samples=16, train_step_time=39.8, train_epoch=1]
Training steps:   0%|          | 2/1876 [02:34<36:54:14, 70.89s/it, train_grad_norm=nan, train_lr=5e-7, train_loss=1.44, train_consumed_samples=16, train_step_time=39.8, train_epoch=1] 
Training steps:   0%|          | 2/1876 [03:14<36:54:14, 70.89s/it, train_grad_norm=nan, train_lr=1e-6, train_loss=1.3, train_consumed_samples=24, train_step_time=39.6, train_epoch=1] 
```
