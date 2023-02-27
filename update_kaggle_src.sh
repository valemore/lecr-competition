#!/bin/sh
GITHASH=`git rev-parse --short HEAD`
rsync --exclude-from exclude_file.txt -r --delete . ../kaggle/code/src
touch ../kaggle/code/src/dummy
kaggle datasets version --dir-mode "zip" -p  ../kaggle/code -m $GITHASH
