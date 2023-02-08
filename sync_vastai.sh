#!/bin/sh
SSH_PROFILE=$1
rsync --exclude-from exclude_file.txt -r ~/v/kolibri/kolibri-code/ $SSH_PROFILE:~/kolibri/kolibri-code
