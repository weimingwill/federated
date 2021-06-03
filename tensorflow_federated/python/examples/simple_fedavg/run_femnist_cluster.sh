#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)/projects/federated
data_dir=/mnt/lustre/$(whoami)/projects/easyfl/easyfl/datasets/data

export PYTHONPATH=$PYTHONPATH:${root_dir}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="/mnt/lustre/share/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-10.1/lib64:$LD_LIBRARY_PATH"

srun -u --partition=innova --job-name=femnist \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python ${root_dir}/tensorflow_federated/python/examples/simple_fedavg/fedavg_main.py \
        --data_dir ${data_dir}/femnist/femnist_iid_10_10_1_0.05_0.05_sample_0.9/ --dataset femnist \
        --total_rounds 150 --train_clients_per_round 10 --client_epochs_per_round 10 \
        --batch_size 64 --test_batch_size 64 --client_learning_rate 0.01 --test_all 2>&1 | tee log/tff-femnist-${now}.log &

