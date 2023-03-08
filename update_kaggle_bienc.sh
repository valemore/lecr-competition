#!/bin/sh
FNAME=$1
cp -r ../out/$FNAME/* ../kaggle/model/
touch ../kaggle/model/dummy
kaggle datasets version --dir-mode "zip" -p  ../kaggle/model -m "Update biencoder $FNAME"
