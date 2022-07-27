#!/bin/bash

# This script synchorizes the content of the public repo with the private one.
# Remark: we can NOT share the private one due to possibility of commits containing participants information.


PARENT_DIR="../ML-handedness"

FOLDERS="mlhand notebooks resources scripts statistics"

for FOLDER in $FOLDERS
do
    echo "Cleaning and Copying $FOLDER"
    [ -e $FOLDER ] && rm -r $FOLDER 
    cp -r $PARENT_DIR/$FOLDER .
done
