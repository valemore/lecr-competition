#!/bin/sh
FNAME=$1
cp -r ../out/$FNAME/* ../kaggle/model/
kaggle datasets version --dir-mode "zip" -p  ../kaggle/cross -m "Update crossencoder $FNAME"
