#!/bin/sh
SSH_PROFILE=$1
ssh $SSH_PROFILE 'mkdir -p .kaggle'
scp ~/.kaggle/kaggle.json $SSH_PROFILE:~/.kaggle/
ssh $SSH_PROFILE 'mkdir kolibri'
rsync --exclude-from exclude_file.txt -r ~/v/kolibri/kolibri-code/ $SSH_PROFILE:~/kolibri/kolibri-code

scp ~/.ssh/vastai $SSH_PROFILE:~/.ssh/
scp ~/.ssh/vastai.pub $SSH_PROFILE:~/.ssh/

ssh $SSH_PROFILE 'cd ~/kolibri/kolibri-code && bash setup_env.sh'
ssh $SSH_PROFILE 'cd ~/kolibri/kolibri-code && vastai_cmds.sh'
