#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux operating system
        echo "Linux operating system detected"
        echo "Downloading the datasets..."
        wget -O datasets.zip https://berkeley.box.com/shared/static/3byiy1599f6kqu5usfijoizp3antpr2x.zip
        unzip datasets.zip
        rm datasets.zip

elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        echo "Mac OSX operating system detected"
        echo "Downloading the datasets..."
        curl -L https://berkeley.box.com/shared/static/3byiy1599f6kqu5usfijoizp3antpr2x --output datasets.zip
        unzip datasets.zip
        rm datasets.zip
        rm -rf __MACOSX/
else        
        echo "Detected operating system unsupported"
fi