#%% Imports
import argparse
import datetime
import pandas as pd
import os 
import torch 
import numpy as np

#-- Scripts
from core import utils
from core.gpu import gpu_status

def train_args():

    # Gather the arguments from the command line
    parser = argparse.ArgumentParser(description='Train the models for StrainNet.')

    # Define the random seed
    parser.add_argument('--seed', type=int, default=0, help='The random seed to use.')

    # Define the device to use
    parser.add_argument('--device', type=str, default='cuda', help='The device to use.')

    # Define the gpu(s) to use
    parser.add_argument('--gpu', nargs='+', default=[0], help='The gpu(s) to use.')

    # Define the model type
    parser.add_argument('--model_type', type=str, default='DeformationClassifier', help='The model type to train.')

    # Define the model types 
    model_types = ['DeformationClassifier', 'TensionNet', 'CompressionNet', 'RigidNet']
    parser.add_argument('--model_types', nargs='+', default=model_types, help='The model types to train.')

    # If you wish to train all of the models
    parser.add_argument('--train_all', action='store_true', help='Whether to train all of the models.')

    # Define the model name of the DeformationClassifier
    parser.add_argument('--DeformationClassifier_name', type=str, default='DeformationClassifier', help='The model name of the DeformationClassifier.')

    # Define the model name of the TensionNet
    parser.add_argument('--TensionNet_name', type=str, default='TensionNet', help='The model name of the TensionNet.')

    # Define the model name of the CompressionNet
    parser.add_argument('--CompressionNet_name', type=str, default='CompressionNet', help='The model name of the CompressionNet.')

    # Define the model name of the RigidNet
    parser.add_argument('--RigidNet_name', type=str, default='RigidNet', help='The model name of the RigidNet.')

    # Define the number of classes 
    parser.add_argument('--num_classes', type=int, default=3, help='The number of classes (number of different deformations to predict) to use.')

    # Define the number of input channels
    parser.add_argument('--input_channels', type=int, default=2, help='The number of input channels to use.')

    # Define the height of the input image
    parser.add_argument('--input_height', type=int, default=384, help='The height of the input image to use.')

    # Define the width of the input image
    parser.add_argument('--input_width', type=int, default=192, help='The width of the input image to use.')

    # Define the number of epochs
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train for.')

    # Define the batch size
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size to use.')

    # Define the learning rate
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate to use.')

    # Define the optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSProp'],  help='The optimizer to use.')

    # Define the name of your experiment
    experiment_name = 'training'
    parser.add_argument('--experiment_name', type=str, default=experiment_name, help='The name of your experiment.')

    # Define the log directory
    log_dir = 'runs/' # Note will tack on the experiment name after the arguments are parsed
    parser.add_argument('--log_dir', type=str, default=log_dir, help='The log directory to use.')
    
    # Define the model directory
    model_dir = 'models/' # Note will tack on the experiment name after the arguments are parsed
    parser.add_argument('--model_dir', type=str, default=model_dir, help='The model directory to use.')

    # Define the name of the dataset
    dataset_name = 'train_set_N_tension_5_N_compression_5_N_rigid_5'
    parser.add_argument('--dataset_name', type=str, default=dataset_name, help='The name of the dataset.')

    # Define the location of the training data
    train_data_dir = 'datasets/' + dataset_name + '/training'
    parser.add_argument('--train_data_dir', type=str, default=train_data_dir, help='The directory of the training data.')

    # Define the location of the validation data
    val_data_dir = 'datasets/' + dataset_name + '/validation'
    parser.add_argument('--val_data_dir', type=str, default=val_data_dir, help='The directory of the validation data.')

    # Define whether to visualize the data
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the data.')

    # Add an argument for resuming training that is a path to the direction where the model (the .pt file) is located
    parser.add_argument('--resume', type=str, default=None, help='The path to the model to resume training from.')

    # Parse the arguments
    args = parser.parse_args()

    # Add the experiment name to the model directory
    args.model_dir = os.path.join(args.model_dir, args.experiment_name)

    # Tack on a timestamp to the experiment name
    args.experiment_name = args.experiment_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Add the experiment name to the log directory
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)

    # Create a directory for logging the results
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Create a directory for saving the models
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Assert the training data directory exists
    assert os.path.exists(args.train_data_dir), 'The training data directory does not exist.'

    # Assert the validation data directory exists
    assert os.path.exists(args.val_data_dir), 'The validation data directory does not exist.'

    # Robustly handle the gpu argument
    args.device = gpu_status(args.device, args.gpu)

    # Add the training transform
    args.train_transform = utils.get_transform(args, 'train')

    # Add the validation transform
    args.valid_transform = utils.get_transform(args, 'valid')

    # Save the args as a xlsx file
    args_df = pd.DataFrame.from_dict(vars(args), orient='index')
    args_df.to_excel(args.log_dir + '/args.xlsx')
    
    # Loop through the arguments and print them
    print('================================================================')
    print('======================== Arguments: ============================')
    print('================================================================')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('================================================================')
    print('================================================================')
    print('================================================================')

    return args

# Define a function for evaluating the model arguments
def eval_args():
    
    # Gather the arguments from the command line
    parser = argparse.ArgumentParser(description='Evaluate the models for StrainNet.')

    # Define the random seed
    parser.add_argument('--seed', type=int, default=0, help='The random seed to use.')

    # Define the device to use
    parser.add_argument('--device', type=str, default='cuda', help='The device to use.')

    # Define the gpu(s) to use
    parser.add_argument('--gpu', nargs='+', default=[0], help='The gpu(s) to use.')

    # Define the model type
    parser.add_argument('--model_type', type=str, default='StrainNet', help='The model type to evaluate.')

    # Define the model name of the DeformationClassifier
    parser.add_argument('--DeformationClassifier_name', type=str, default='DeformationClassifier', help='The model name of the DeformationClassifier.')

    # Define the model name of the TensionNet
    parser.add_argument('--TensionNet_name', type=str, default='TensionNet', help='The model name of the TensionNet.')

    # Define the model name of the CompressionNet
    parser.add_argument('--CompressionNet_name', type=str, default='CompressionNet', help='The model name of the CompressionNet.')

    # Define the model name of the RigidNet
    parser.add_argument('--RigidNet_name', type=str, default='RigidNet', help='The model name of the RigidNet.')

    # Define the number of classes 
    parser.add_argument('--num_classes', type=int, default=3, help='The number of classes (number of different deformations to predict) to use.')

    # Define the number of input channels
    parser.add_argument('--input_channels', type=int, default=2, help='The number of input channels to use.')

    # Define the height of the input image
    parser.add_argument('--input_height', type=int, default=384, help='The height of the input image to use.')

    # Define the width of the input image
    parser.add_argument('--input_width', type=int, default=192, help='The width of the input image to use.')

    # Define the batch size
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use.')

    # Define the model directory
    model_dir = 'models/' + 'pretrained/synthetic/' 
    parser.add_argument('--model_dir', type=str, default=model_dir, help='The model directory to use.')

    # Define the log directory
    log_dir = 'results/' + 'pretrained.synthetic/' 
    parser.add_argument('--log_dir', type=str, default=log_dir, help='The log directory to use.')

    # Define the name of the dataset
    dataset_name = 'train_set_N_tension_5_N_compression_5_N_rigid_5'
    parser.add_argument('--dataset_name', type=str, default=dataset_name, help='The name of the dataset.')

    # Define the location of the validation data
    val_data_dir = 'datasets/' + dataset_name + '/validation'
    parser.add_argument('--val_data_dir', type=str, default=val_data_dir, help='The directory of the validation data.')

    # Define whether to visualize the data
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the data.')

    # Define whether to save the strains
    parser.add_argument('--save_strains', action='store_true', help='Whether to save the strains.')

    # Add an argument for dealing with sequential data such as the synthetic test cases
    parser.add_argument('--sequential', action='store_true', help='Whether the data is sequential.')

    # Add an argument the sampling rate of the sequential data
    parser.add_argument('--sampling_rate', type=int, default=1, help='The sampling rate of the sequential data.')

    # Add an argument for specifying a custom sampling for the sequential data
    parser.add_argument('--custom_sampling', action='store_true', help='Whether to use a custom sampling for the sequential data.')
    
    # Just need this argument to keep the same interface as the evaluation script
    parser.add_argument('--resume', type=str, default=None, help='Keep this argument even though it serves no purpose for evaluation.')

    # Parse the arguments
    args = parser.parse_args()

    # Create a directory for logging the results
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Assert the validation data directory exists
    assert os.path.exists(args.val_data_dir), 'The validation data directory does not exist.'

    # Robustly handle the gpu argument
    args.device = gpu_status(args.device, args.gpu)

    # Add the validation transform
    args.valid_transform = utils.get_transform(args, 'valid')

    # Add custom sampling if specified
    # For the synthetic test cases, we use a custom sampling
    # We sample at a rate of 30 for the first 300 frames, then 50 for next 500 frames, and then 30 for the last 300 frames
    # e.g., [0, 30, 60, ..., 300, 350, 400, ..., 800, 830, 860, ..., 1100]
    if args.custom_sampling:
        args.custom_sampling = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 830, 860, 890, 920, 950, 980, 1010, 1040, 1070, 1100]

    # Save the args as a xlsx file
    args_df = pd.DataFrame.from_dict(vars(args), orient='index')
    args_df.to_excel(args.log_dir + '/args.xlsx')
    
    # Loop through the arguments and print them
    print('================================================================')
    print('======================== Arguments: ============================')
    print('================================================================')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('================================================================')
    print('================================================================')
    print('================================================================')

    return args

# Define a function for applying the model to experimental data
def experimental_args():
    
    # Gather the arguments from the command line
    parser = argparse.ArgumentParser(description='Apply StrainNet to experimental data.')

    # Define the random seed
    parser.add_argument('--seed', type=int, default=0, help='The random seed to use.')

    # Define the device to use
    parser.add_argument('--device', type=str, default='cuda', help='The device to use.')

    # Define the gpu(s) to use
    parser.add_argument('--gpu', nargs='+', default=[0], help='The gpu(s) to use.')

    # Define the model type
    parser.add_argument('--model_type', type=str, default='StrainNet', help='The model type to evaluate.')

    # Define the model name of the DeformationClassifier
    parser.add_argument('--DeformationClassifier_name', type=str, default='DeformationClassifier', help='The model name of the DeformationClassifier.')

    # Define the model name of the TensionNet
    parser.add_argument('--TensionNet_name', type=str, default='TensionNet', help='The model name of the TensionNet.')

    # Define the model name of the CompressionNet
    parser.add_argument('--CompressionNet_name', type=str, default='CompressionNet', help='The model name of the CompressionNet.')

    # Define the model name of the RigidNet
    parser.add_argument('--RigidNet_name', type=str, default='RigidNet', help='The model name of the RigidNet.')

    # Define the number of classes 
    parser.add_argument('--num_classes', type=int, default=3, help='The number of classes (number of different deformations to predict) to use.')

    # Define the number of input channels
    parser.add_argument('--input_channels', type=int, default=2, help='The number of input channels to use.')

    # Define the height of the input image
    parser.add_argument('--input_height', type=int, default=384, help='The height of the input image to use.')

    # Define the width of the input image
    parser.add_argument('--input_width', type=int, default=192, help='The width of the input image to use.')

    # Define the batch size
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use.')

    # Define the model directory
    model_dir = 'models/' + 'pretrained/experimental/' 
    parser.add_argument('--model_dir', type=str, default=model_dir, help='The model directory to use.')

    # Define the log directory
    log_dir = 'results/' + 'Subject1.10mvc.trial1/'
    parser.add_argument('--log_dir', type=str, default=log_dir, help='The log directory to use.')

    # Define the location of the experimental data
    exp_data_dir = 'datasets/experimental/test/10mvc/trial1/'
    parser.add_argument('--exp_data_dir', type=str, default=exp_data_dir, help='The directory of the validation data.')

    # Define the crop box for getting the bulk strains from the experimental data
    lower_left_x = 25
    lower_left_y = 125
    upper_right_x = 175
    upper_right_y = 225
    parser.add_argument('--crop_box', nargs='+', default=[lower_left_x, lower_left_y, upper_right_x, upper_right_y], help='The crop box for getting the bulk strains from the experimental data.')

    # Define whether to visualize the data
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the data.')

    # Define whether to save the strains
    parser.add_argument('--save_strains', action='store_true', help='Whether to save the strains.')

    # Add an argument the sampling rate of the sequential data
    parser.add_argument('--sampling_rate', type=int, default=1, help='The sampling rate of the sequential data.')

    # Parse the arguments
    args = parser.parse_args()

    # Create a directory for logging the results
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Assert the validation data directory exists
    print('The experimental data directory is: ' + args.exp_data_dir)
    assert os.path.exists(args.exp_data_dir), 'The experimental data directory does not exist.'

    # Robustly handle the gpu argument
    args.device = gpu_status(args.device, args.gpu)

    # Add the validation transform
    args.test_transform = utils.get_transform(args, 'test')

    # Save the args as a xlsx file
    args_df = pd.DataFrame.from_dict(vars(args), orient='index')
    args_df.to_excel(args.log_dir + '/args.xlsx')
    
    # Loop through the arguments and print them
    print('================================================================')
    print('======================== Arguments: ============================')
    print('================================================================')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('================================================================')
    print('================================================================')
    print('================================================================')

    return args