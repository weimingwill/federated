#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)/projects/federated
data_dir=/mnt/lustre/$(whoami)/projects/easyfl/easyfl/datasets/data

export PYTHONPATH=$PYTHONPATH:${root_dir}
export PATH="/mnt/lustre/share/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-10.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

srun -u --partition=innova --job-name=tff-cifar8 \
    -n8 --gres=gpu:8 --ntasks-per-node=8 -w SG-IDC2-10-51-3-50 \
    python -u ${root_dir}/tensorflow_federated/python/examples/simple_fedavg/fedavg_main.py \
        --data_dir ${data_dir}/cifar10/cifar10_iid_10_10_1_0.5_0/ \
        --dataset cifar10 --total_rounds 20 --train_clients_per_round 10 --client_epochs_per_round 10 \
        --batch_size 64 --test_batch_size 64 --client_learning_rate 0.01 --test_in_server --gpu 1 | tee log/tff-cifar10-clean-${now}.log &
