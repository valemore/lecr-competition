#!/bin/sh
cd ~/kolibri
apt install unzip
pip install kaggle
mkdir -p data
cd data
kaggle competitions download -c learning-equality-curriculum-recommendations && unzip learning-equality-curriculum-recommendations.zip && rm learning-equality-curriculum-recommendations.zip

echo "Host github.com
  AddKeysToAgent yes
  IdentityFile ~/.ssh/vastai
  IdentitiesOnly yes
" > ~/.ssh/config
echo 'eval "$(ssh-agent -s)"' >> ~/.bashrc
