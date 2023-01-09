<!-- omit in toc -->
# StrainNet
StrainNet is a deep learning based method for predicting strain from images

![Teaser image](figures/StrainNet.gif)

### [Project Page](https://strainnet.net) | Paper | bioRxiv | SB3C Abstract | Citation

**Note:** This paper has been submitted to a journal and the link will be added upon publication.
<br/>

<!-- omit in toc -->
# Table of Contents 
- [Getting Started](#getting-started)
  - [Set-up](#set-up)
  - [Downloading pre-trained models and data](#downloading-pre-trained-models-and-data)
  - [Demo](#demo)
- [Generating a training set](#generating-a-training-set)
- [Training StrainNet](#training-strainnet)
  - [Arguments](#arguments)
  - [Resuming training on pre-trained models](#resuming-training-on-pre-trained-models)
  - [Training all models](#training-all-models)
  - [Viewing the progress of your training with Tensorboard](#viewing-the-progress-of-your-training-with-tensorboard)
    - [Viewing the Tensorboard Webpage](#viewing-the-tensorboard-webpage)
    - [Viewing the Tensorboard File in VSCode](#viewing-the-tensorboard-file-in-vscode)
- [Evaluating the performance of StrainNet](#evaluating-the-performance-of-strainnet)
- [Testing StrainNet on experimental images](#testing-strainnet-on-experimental-images)
- [Citation](#citation)
- [License](#license)

# Getting Started

## Set-up

Begin by cloning this repository:

```
git clone https://github.com/reecehuff/StrainNet.git
cd StrainNet
```

Next, install the necessary Python packages with Anaconda.
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

## Demo



# Generating a training set

<!-- omit in toc -->
### For a full tutorial, see [generateTrainingSet/README.md](generateTrainingSet/README.md).

<br/>

# Training StrainNet 

To train StrainNet, you will need to run the train.py script. This script can be invoked from the command line, and there are several optional arguments that you can use to customize the training process.

Here is an example command for training StrainNet with the default settings:

python train.py
You can also adjust the training settings by specifying command-line arguments. For example, to change the optimizer and learning rate, you can use the following command:

```
python train.py --optimizer Adam --learning-rate 0.001
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

To train all the models in this project, you can use the `train.sh` script. This script will invoke the necessary training scripts and pass the appropriate arguments to them.

To run the `train.sh` script, simply execute the following command from the terminal:

```
bash scripts/train.sh
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

The `eval.py` script will output the evaluation metrics. You can also pass the --save-results flag to save the evaluation results to a file:

```
python eval.py --model_dir "path/to/trained/models" --val_data_dir "path/to/validation/data"
```
You can see a list of all the available arguments for the `eval.py` script by using the `--help` flag:

```
python eval.py --help
```
Or examine the [`core/arguments.py`](core/arguments.py) Python script. 

# Testing StrainNet on experimental images

**Note:** If you have not already, please download the pre-trained models. 

# Citation

<!-- omit in toc -->
### APA:

Smith, J., & Johnson, J. (2021). The Model: A Novel Approach to Machine Learning. Journal of Machine Learning, 2(1), 1-10.

<!-- omit in toc -->
### BiBTeX:
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
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.