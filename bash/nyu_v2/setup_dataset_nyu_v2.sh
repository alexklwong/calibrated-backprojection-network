#!/bin/bash

mkdir -p 'data'
mkdir -p 'data/nyu_v2'

cd 'data'

# Download NYUv2 using gdown
if [ $# -eq 0 ]; then
    gdown https://drive.google.com/uc?id=1E5NgaEE8zEr4OizVcxc3nfQGfjSatUjX
fi

unzip -o 'nyu_v2.zip' -d 'data/nyu_v2/'
mv 'nyu_v2.zip' 'data/nyu_v2/'

cd ..
python setup/setup_dataset_nyu_v2.py
