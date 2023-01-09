#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux operating system
        echo "Linux operating system detected"
        echo "Downloading the datasets..."
        wget -O datasets.zip https://berkeley.box.com/shared/static/fbnni1syztr9zqlt6j1ewyy6vq7bg2w2.zip
        unzip datasets.zip
        rm datasets.zip

elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        echo "Mac OSX operating system detected"
        echo "Downloading the datasets..."
        curl -L https://berkeley.box.com/shared/static/fbnni1syztr9zqlt6j1ewyy6vq7bg2w2 --output datasets.zip
        unzip datasets.zip
        rm datasets.zip
        rm -rf __MACOSX/
else        
        echo "Detected operating system unsupported"
fi