#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux operating system
        echo "Linux operating system detected"
        echo "Downloading the models..."
        wget -O models.zip https://berkeley.box.com/shared/static/iao62834avik8y8dw4dth684bhyomesz.zip
        unzip models.zip
        rm models.zip
        rm -rf __MACOSX/

elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        echo "Mac OSX operating system detected"
        echo "Downloading the models..."
        curl -L https://berkeley.box.com/shared/static/iao62834avik8y8dw4dth684bhyomesz --output models.zip 
        unzip models.zip
        rm models.zip
        rm -rf __MACOSX/
else        
        echo "Detected operating system unsupported"
fi

