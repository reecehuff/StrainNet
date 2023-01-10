<!-- omit in toc -->
# StrainNet
StrainNet is a deep learning based method for predicting strain from images

![Teaser image](figures/StrainNet.gif)

### [Project Page](https://strainnet.net) | Paper | bioRxiv | SB3C Abstract | [Citation](#citation)

**Note:** This paper has been submitted to a journal and the link will be added upon publication.
<br/>

<!-- omit in toc -->
# Table of Contents 
- [Getting Started](#getting-started)
  - [Set-up](#set-up)
  - [Downloading pre-trained models and data](#downloading-pre-trained-models-and-data)
  - [Demo: Applying StrainNet to a Synthetic Test Case](#demo-applying-strainnet-to-a-synthetic-test-case)
- [Generating a training set](#generating-a-training-set)
- [Training StrainNet](#training-strainnet)
  - [Arguments](#arguments)
  - [Resuming training on pre-trained models](#resuming-training-on-pre-trained-models)
  - [Training all models](#training-all-models)
  - [Viewing the progress of your training with Tensorboard](#viewing-the-progress-of-your-training-with-tensorboard)
    - [Viewing the Tensorboard Webpage](#viewing-the-tensorboard-webpage)
    - [Viewing the Tensorboard File in VSCode](#viewing-the-tensorboard-file-in-vscode)
- [Evaluating the performance of StrainNet](#evaluating-the-performance-of-strainnet)
  - [Arguments](#arguments-1)
  - [Evaluating StrainNet on the synthetic test cases](#evaluating-strainnet-on-the-synthetic-test-cases)
- [Applying StrainNet to experimental images](#applying-strainnet-to-experimental-images)
  - [Arguments](#arguments-2)
  - [Applying StrainNet to human flexor tendons *in vivo*](#applying-strainnet-to-human-flexor-tendons-in-vivo)
- [Citation](#citation)
    - [TBD](#tbd)
- [LICENSE](#license)

# Getting Started

## Set-up

Begin by cloning this repository:

```
git clone https://github.com/reecehuff/StrainNet.git
cd StrainNet
```

Next, install the necessary Python packages with [Anaconda](https://www.anaconda.com/products/distribution).
```
conda create -n StrainNet python=3.9
conda activate StrainNet
pip install -r requirements.txt
```

Finally, make sure that Python path is correctly set. The commmand

```
which python
```
should display the path to the Anaconda's environment Python path, *e.g.*, `/opt/anaconda3/envs/StrainNet/bin/python`

## Downloading pre-trained models and data

To download the data and pretrained models for this project, you can use the [`download.sh`](scripts/download.sh) script. This script will download the data and models from a remote server and save them to your local machine.

**Warning: The data is approximately 10 GB in size and may take some time to download.**

To download the data and models, run the following command:

```
. scripts/download.sh
```

This will download the data and models and save them to the current working directory. The data and models may be stored in subdirectories within the current working directory.

## Demo: Applying StrainNet to a Synthetic Test Case

To see a demo of StrainNet in action, you can apply the model to a synthetic test case. The synthetic test case is a simulated image with known strains that can be used to test the accuracy of the model.

To apply StrainNet to the synthetic test case, use the following command:

```
. scripts/demo.sh
```

You should now see a `results` folder with some plots of the performance on a synthetic test case where the largest strain is $4\%$ (see the `04DEF` in `StrainNet/datasets/SyntheticTestCases/04DEF`). 

# Generating a training set

<!-- omit in toc -->
### For a full tutorial, see [generateTrainingSet/README.md](generateTrainingSet/README.md).

<br/>

# Training StrainNet 

After generating a training, StrainNet can be trained. To train StrainNet, you will need to run the `train.py` script. This script can be invoked from the command line, and there are several optional arguments that you can use to customize the training process.

Here is an example command for training StrainNet with the default settings:

```
python train.py
```

You can also adjust the training settings by specifying command-line arguments. For example, to change the optimizer and learning rate, you can use the following command:

```
python train.py --optimizer SGD --lr 0.01
```

## Arguments
Below is a list of some of the available command-line arguments that you can use to customize the training process:

| Argument      | Default     | Description                                                                                             |
|---------------|-------------|---------------------------------------------------------------------------------------------------------|
| `--optimizer` | `Adam`       | The optimizer to use for training.                          |
| `--lr` | `0.001`  | The learning rate to use for the optimizer.                                                             |
| `--batch_size`    | `8`     | The batch size to use for training.                                                                     |
| `--epochs`    | `100`      | The number of epochs to train for.                                                                      |                         
| `--train_all`    | `False`      | Whether to train all of the models.  

For a complete list of available command-line arguments and their descriptions, you can use the `--help` flag:

```
python train.py --help
```

Or examine the [`core/arguments.py`](core/arguments.py) Python script. 

## Resuming training on pre-trained models
Pre-trained Models
You can also use pre-trained models for StrainNet by specifying the `--resume` flag and the path to the pre-trained model. For example:

```
python train.py --resume "path/to/pretrained.pt"
```

## Training all models

By default, `train.py` will only train one of the four models needed for StrainNet. To train all the models needed for StrainNet, you can use the `train.sh` script. This script will invoke the necessary training scripts and pass the appropriate arguments to them.

To run the `train.sh` script, simply execute the following command from the terminal:

```
. scripts/train.sh
```

## Viewing the progress of your training with Tensorboard

By default, running `train.py` will write an `events.out` file to visualize the progress of training StrainNet with Tensorboard. After running `train.py`, locate the `events.out` in the newly-created `runs` folder. 

### Viewing the Tensorboard Webpage
To view the Tensorboard webpage, you will need to start a Tensorboard server. You can do this by running the following command in the terminal:

```
tensorboard --logdir="path/to/events.out"
```

Replace `"path/to/events.out"` with the actual path to your events.out file. This will start a Tensorboard server and print a message with a URL that you can use to access the Tensorboard webpage.

To view the Tensorboard webpage, open a web browser and navigate to the URL printed by the Tensorboard server. This will open the Tensorboard webpage, which allows you to view various training metrics and graphs.

### Viewing the Tensorboard File in [VSCode](https://code.visualstudio.com/Download)

To view the Tensorboard `events.out` file in Visual Studio Code, you may use the Tensorboard command. 

1. Open the command palette (View &rarr; Command Palette... or  Cmd + Shift + P on macOS) 
2. Type "Python: Launch Tensorboard" in the command palette and press Enter. 
3. Select the `runs` folder to view `events.out` files. 

# Evaluating the performance of StrainNet

After training the model, you can evaluate its performance on a test dataset to see how well it generalizes to unseen data. To evaluate the model, you will need to have a test dataset in a format that the model can process.

To evaluate the model, you can use the `eval.py` script. This script loads the trained model and the test dataset, and runs the model on the test data to compute evaluation metrics such as accuracy and precision.

To run the `eval.py` script, use the following command:

```
python eval.py --model_dir "path/to/trained/models" --val_data_dir "path/to/validation/data"
```
Replace `val_data_dir` with the actual path to the trained models, and `"path/to/validation/data"` with the actual path to the validation data.

## Arguments

You can see a list of all the available arguments for the `eval.py` script by using the `--help` flag:

```
python eval.py --help
```
Or examine the [`core/arguments.py`](core/arguments.py) Python script. 

## Evaluating StrainNet on the synthetic test cases

To apply the pretrained models to the synthetic test cases, you can use the `eval.sh` script. This script will invoke the necessary evaluation scripts and pass the appropriate arguments to them.

To run the `eval.sh` script, simply execute the following command from the terminal:

```
. scripts/eval.sh
```

# Applying StrainNet to experimental images

To apply StrainNet to experimental images that do not have known strains, you can use the `apply2experimental.py` script. This script loads the trained StrainNet model and processes the experimental images to predict the strains.

To apply StrainNet to experimental images, use the following command:

```
python apply2experimental.py --model path/to/trained/model --experimental-data path/to/experimental/data
```

Replace `path/to/trained/model` with the actual path to the trained StrainNet model, and `path/to/experimental/data` with the actual path to the experimental data.

The `apply2experimental.py` script will output the predicted strains to the terminal and/or save them to a file, depending on the specified command line arguments.

## Arguments

You can see a list of all the available arguments for the `apply2experimental.py` script by using the `--help` flag:

```
python apply2experimental.py --help
```

**Note:** The `apply2experimental.py` script requires the experimental images to be in a specific format that is compatible with StrainNet. 

## Applying StrainNet to human flexor tendons *in vivo*

To apply the pretrained models real experimental data, you can use the `flexor_tendon.sh` script. This script will invoke the necessary scripts and pass the appropriate arguments to them.

To run the `flexor_tendon.sh` script, simply execute the following command from the terminal:

```
. scripts/flexor_tendon.sh
```

# Citation

### TBD


<!-- omit in toc -->
<!-- ### APA:

Smith, J., & Johnson, J. (2021). The Model: A Novel Approach to Machine Learning. Journal of Machine Learning, 2(1), 1-10. -->

<!-- omit in toc -->
<!-- ### BiBTeX:
```
@article{huff2023StrainNet,
  title={The Model: A Novel Approach to Machine Learning},
  author={Smith, John and Johnson, Jane},
  journal={Journal of Machine Learning},
  volume={2},
  number={1},
  pages={1-10},
  year={2021}
}
``` -->

# LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.