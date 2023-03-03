#!/bin/sh
FNAME=$1
cp -r ../cout/$FNAME/* ../kaggle/cross/
touch ../kaggle/cross/dummy
kaggle datasets version --dir-mode "zip" -p  ../kaggle/cross -m "Update crossencoder $FNAME"
