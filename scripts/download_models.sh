#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux operating system
        echo "Linux operating system detected"
        echo "Downloading the models..."
        wget -O models.zip https://www.dropbox.com/s/dl/n41vqk95q25d4j3/models.zip
        unzip models.zip
        rm models.zip
        rm -rf __MACOSX/

elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        echo "Mac OSX operating system detected"
        echo "Downloading the models..."
        curl -L https://www.dropbox.com/s/dl/n41vqk95q25d4j3/models.zip --output models.zip 
        unzip models.zip
        rm models.zip
        rm -rf __MACOSX/
else        
        echo "Detected operating system unsupported"
        echo "Visit: https://www.dropbox.com/s/n41vqk95q25d4j3/models.zip?dl=0"
        echo "to download the models."
fi

