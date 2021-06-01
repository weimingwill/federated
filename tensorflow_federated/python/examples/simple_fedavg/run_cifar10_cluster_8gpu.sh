#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)/projects/federated
data_dir=/mnt/lustre/$(whoami)/projects/easyfl/easyfl/datasets/data

export PYTHONPATH=$PYTHONPATH:${root_dir}

srun -u --partition=innova --job-name=femnist \
    -n8 --gres=gpu:8 --ntasks-per-node=8 \
    python -u ${root_dir}/tensorflow_federated/python/examples/simple_fedavg/fedavg_main.py \
        --data_dir ${data_dir}/cifar10/cifar10_iid_10_10_1_0.5_0/ \
        --dataset cifar10 --total_rounds 100 --train_clients_per_round 10 --client_epochs_per_round 10 \
        --batch_size 64 --test_batch_size 64 --client_learning_rate 0.01 --test_in_server | tee log/tff-cifar10-${now}.log &
