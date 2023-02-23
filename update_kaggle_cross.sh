#!/bin/sh
FNAME=$1
cp -r ../out/$FNAME/* ../kaggle/cross/
kaggle datasets version --dir-mode "zip" -p  ../kaggle/cross -m "Update crossencoder $FNAME"
