#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

data_dir=~/personal-projects/easyfl/easyfl/datasets/data
root_dir=~/personal-projects/federated/

export PYTHONPATH=$PYTHONPATH:${root_dir}

python -u ${root_dir}/tensorflow_federated/python/examples/simple_fedavg/fedavg_main.py \
    --data_dir ${data_dir}/shakespeare/shakespeare_iid_100_10_1_0.05_0.1_sample_0.9/ \
    --dataset shakespeare --total_rounds 100 --train_clients_per_round 10 --client_epochs_per_round 10 \
    --batch_size 64 --test_batch_size 64 --client_learning_rate 0.8 --test_all --gpu 0 | tee log/tff-shakespeare-cpu-ntu-${now}.log &