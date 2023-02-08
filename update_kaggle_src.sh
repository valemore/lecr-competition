#!/bin/sh
rsync --exclude-from exclude_file.txt -r ~/v/kolibri/kolibri-code ~/v/kaggle/kolibri/kolibri-code
touch ~/v/kaggle/kolibri/kolibri-code/dummy
kaggle datasets version --dir-mode "zip" -p  ~/v/kaggle/kolibri-code -m "Update kolibri-code"
