#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux operating system
        echo "Linux operating system detected"
        echo "Downloading the datasets..."
        wget -O datasets.zip https://www.dropbox.com/s/dl/5datbm5w2cgaqva/datasets.zip
        unzip datasets.zip
        rm datasets.zip
        rm -rf __MACOSX/

elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        echo "Mac OSX operating system detected"
        echo "Downloading the datasets..."
        curl -L https://www.dropbox.com/s/dl/5datbm5w2cgaqva/datasets.zip --output datasets.zip
        unzip datasets.zip
        rm datasets.zip
        rm -rf __MACOSX/
else        
        echo "Detected operating system unsupported"
        echo "Visit: https://www.dropbox.com/s/5datbm5w2cgaqva/datasets.zip?dl=0"
        echo "to download the the datasets."
fi

mkdir generateTrainingSet/input/
cp -R datasets/experimental/train/fullsize/* generateTrainingSet/input/.