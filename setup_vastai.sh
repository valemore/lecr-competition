#!/bin/sh
SSH_PROFILE=$1
ssh $SSH_PROFILE 'mkdir -p .kaggle'
scp ~/.kaggle/kaggle.json $SSH_PROFILE:~/.kaggle/
ssh $SSH_PROFILE 'mkdir kolibri'
rsync --exclude-from exclude_file.txt -r ~/v/kolibri/kolibri-code/ $SSH_PROFILE:~/kolibri/kolibri-code

scp ~/.ssh/vastai $SSH_PROFILE:~/.ssh/
scp ~/.ssh/vastai.pub $SSH_PROFILE:~/.ssh/

ssh vastai1 'tmux new -d -s sesh'
ssh vastai1 'tmux send-keys -t sesh.0 "cd ~/kolibri/kolibri-code && bash setup_env.sh" ENTER'
ssh vastai1 'tmux send-keys -t sesh.0 "cd ~/kolibri/kolibri-code && bash vastai_cmds.sh" ENTER'
ssh vastai1 'tmux kill-session -t sesh'
