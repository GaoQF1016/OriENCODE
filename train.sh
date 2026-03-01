#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate oriencoder_env

cd Downloads_original/ 

python3 -m decode.neuralfitter.train.live_engine -i cuda:0 -p notebook_unfocused.yaml

echo "Start training"
