#!/bin/sh
FNAME=$1
cp ../out/FNAME ../kaggle/model/biencoder.pt
kaggle datasets version --dir-mode "zip" -p  ../kaggle/model -m "Update biencoder $FNAME"
