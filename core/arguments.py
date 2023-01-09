#%% Imports
import argparse
import datetime
import pandas as pd
import os 
import torch 

#-- Scripts
from core import utils

def train_args():

    # Gather the arguments from the command line
    parser = argparse.ArgumentParser(description='Train the models for StrainNet.')

    # Define the random seed
    parser.add_argument('--seed', type=int, default=0, help='The random seed to use.')

    # Define the device to use
    parser.add_argument('--device', type=str, default='cuda', help='The device to use.')

    # Define the model types 
    model_types = ['DeformationClassifier', 'TensionNet', 'CompressionNet', 'RigidNet']
    parser.add_argument('--model_types', type=str, default=model_types, help='The model types to train.')

    # If you wish to train all of the models
    parser.add_argument('--train_all', action='store_true', help='Whether to train all of the models.')

    # Define the model type
    parser.add_argument('--model_type', type=str, default='DeformationClassifier', help='The model type to train.')

    # Define the model name of the DeformationClassifier
    parser.add_argument('--DeformationClassifier_name', type=str, default='DeformationClassifier', help='The model name of the DeformationClassifier.')

    # Define the model name of the TensionNet
    parser.add_argument('--TensionNet_name', type=str, default='TensionNet', help='The model name of the TensionNet.')

    # Define the model name of the CompressionNet
    parser.add_argument('--CompressionNet_name', type=str, default='CompressionNet', help='The model name of the CompressionNet.')

    # Define the model name of the RigidNet
    parser.add_argument('--RigidNet_name', type=str, default='RigidNet', help='The model name of the RigidNet.')

    # Define the number of classes 
    parser.add_argument('--num_classes', type=int, default=3, help='The number of classes to use.')

    # Define the number of input channels
    parser.add_argument('--input_channels', type=int, default=2, help='The number of input channels to use.')

    # Define the number of epochs
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train for.')

    # Define the batch size
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size to use.')

    # Define the learning rate
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate to use.')

    # Define the optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='The optimizer to use.')

    # Define the log directory
    log_dir = 'runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument('--log_dir', type=str, default=log_dir, help='The log directory to use.')
    
    # Define the model directory
    model_dir = 'models/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument('--model_dir', type=str, default=model_dir, help='The model directory to use.')

    # Define the name of the dataset
    dataset_name = 'train_set_N_tension_10_N_compression_10_N_rigid_10'
    parser.add_argument('--dataset_name', type=str, default=dataset_name, help='The name of the dataset.')

    # Define the location of the training data
    train_data_dir = 'datasets/' + dataset_name + '/training'
    parser.add_argument('--train_data_dir', type=str, default=train_data_dir, help='The directory of the training data.')

    # Define the location of the validation data
    val_data_dir = 'datasets/' + dataset_name + '/validation'
    parser.add_argument('--val_data_dir', type=str, default=val_data_dir, help='The directory of the validation data.')

    #%% Define whether to visualize the data
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the data.')

    # Parse the arguments
    args = parser.parse_args()

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

    # Make sure that cuda is available if you wish to use it
    if args.device == 'cuda' and not torch.cuda.is_available():
        Warning('Cuda is not available. Using cpu instead.')
        args.device = 'cpu'

    # Add the training transform
    args.train_transform = utils.get_transform('train')

    # Add the validation transform
    args.valid_transform = utils.get_transform('valid')

    # Add the test transform
    args.test_transform = utils.get_transform('test')

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