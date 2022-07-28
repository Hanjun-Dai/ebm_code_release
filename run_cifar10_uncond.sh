#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3

mpiexec -n 4 python train.py \
	--exp=cifar10_uncond \
	--dataset=cifar10 \
	--num_steps=60 \
	--batch_size=128 \
	--step_lr=10.0 \
	--proj_norm=0.01 \
	--zero_kl \
	--replay_batch \
	--large_model

