#!/bin/sh
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit
fi
SSH_PROFILE=$1
BASE_DIR='/root/kolibri'
CODE_DIR="$BASE_DIR/kolibri-code"

ssh $SSH_PROFILE 'mkdir -p .kaggle'
scp ~/.kaggle/kaggle.json $SSH_PROFILE:~/.kaggle/
ssh $SSH_PROFILE "mkdir -p $BASE_DIR"
rsync --exclude-from exclude_file.txt -r . $SSH_PROFILE:$CODE_DIR

scp ~/.ssh/vastai $SSH_PROFILE:~/.ssh/
scp ~/.ssh/vastai.pub $SSH_PROFILE:~/.ssh/

ssh $SSH_PROFILE 'tmux new -d -s sesh'
ssh $SSH_PROFILE 'tmux send-keys -t sesh.0 '"'cd $CODE_DIR && bash setup_env_lit.sh'" ENTER''
ssh $SSH_PROFILE 'tmux send-keys -t sesh.0 '"'cd $CODE_DIR && bash vastai_cmds.sh'" ENTER''
#ssh $SSH_PROFILE 'tmux send-keys -t sesh.0 '"'cd $CODE_DIR && source install_gcloud.sh'" ENTER''
ssh -t $SSH_PROFILE 'tmux a -t sesh.0'
# ssh $SSH_PROFILE 'tmux kill-session -t sesh'
