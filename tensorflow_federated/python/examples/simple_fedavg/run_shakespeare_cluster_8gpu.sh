#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)/projects/federated
data_dir=/mnt/lustre/$(whoami)/projects/easyfl/easyfl/datasets/data

export PYTHONPATH=$PYTHONPATH:${root_dir}

srun -u --partition=innova --job-name=shakespeare8 \
    -n8 --gres=gpu:8 --ntasks-per-node=8 \
    python -u ${root_dir}/tensorflow_federated/python/examples/simple_fedavg/fedavg_main.py \
        --data_dir ${data_dir}/shakespeare/shakespeare_iid_10_10_1_0.2_0.2_sample_0.9/ \
        --dataset shakespeare --total_rounds 100 --train_clients_per_round 10 --client_epochs_per_round 10 \
        --batch_size 64 --test_batch_size 64 --client_learning_rate 0.8 --test_all | tee log/tff-shakespeare-${now}.log &
