#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)/projects/federated
data_dir=/mnt/lustre/$(whoami)/projects/easyfl/easyfl/datasets/data

export PYTHONPATH=$PYTHONPATH:${root_dir}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="/mnt/lustre/share/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-10.1/lib64:$LD_LIBRARY_PATH"

srun -u --partition=innova --job-name=shakespeare \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python -u ${root_dir}/tensorflow_federated/python/examples/simple_fedavg/fedavg_main.py \
        --data_dir ${data_dir}/shakespeare/shakespeare_iid_100_10_1_0.05_0.1_sample_0.9/ \
        --dataset shakespeare --total_rounds 100 --train_clients_per_round 10 --client_epochs_per_round 10 \
        --batch_size 64 --test_batch_size 64 --client_learning_rate 0.8 --test_all | tee log/tff-shakespeare-${now}.log &
