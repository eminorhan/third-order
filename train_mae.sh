#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:05:00
#SBATCH --job-name=train_mae
#SBATCH --output=train_mae_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1
	
srun python -u /scratch/eo41/third-order/train_mae.py \
	--model 'mae_vit_base_patch14_3Dattn' \
	--resume "" \
	--batch_size_per_gpu 512 \
	--num_workers 8 \
	--lr 0.0003 \
	--min_lr 0.0003 \
	--weight_decay 0.0 \
	--mask_ratio 0.8 \
	--input_size 224 \
	--output_dir "/scratch/eo41/third-order/test" \
	--data_path "/scratch/work/public/imagenet/train" \
	--save_prefix "test_vitb14_3dattn"

echo "Done"