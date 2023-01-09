#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux operating system
        echo "Linux operating system detected"
        echo "Downloading the models..."
        wget -O models.zip https://berkeley.box.com/shared/static/rtta5uuufaz5173n4tm33vuth8p6a9yc.zip
        unzip models.zip
        rm models.zip

elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        echo "Mac OSX operating system detected"
        echo "Downloading the models..."
        curl -LA https://berkeley.box.com/shared/static/rtta5uuufaz5173n4tm33vuth8p6a9yc --output models.zip 
        unzip models.zip
        rm models.zip
        rm -rf __MACOSX/
else        
        echo "Detected operating system unsupported"
fi

