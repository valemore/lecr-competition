#!/bin/sh
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
fi
conda install mamba -y -n base -c conda-forge
mamba create -y -n dl -c rapidsai -c conda-forge -c nvidia  -c pytorch cuml=22.12 python=3.9 cudatoolkit=11.5 pytorch torchvision torchaudio pytorch-cuda=11.7 pandas scikit-learn pip
/opt/conda/envs/dl/bin/pip install neptune-client==0.16.17 transformers==4.26.0 lightning==1.9.3
mamba init bash
