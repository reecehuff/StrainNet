#%% Imports
import argparse
import datetime
import pandas as pd
import os 

def generate_args():

    # Gather the arguments from the command line
    parser = argparse.ArgumentParser(description='Generate a training set for StrainNet.')

    # Define the random seed
    parser.add_argument('--seed', type=int, default=0, help='The random seed to use.')

    # Image and mask paths
    parser.add_argument('--image_path', type=str, default='generateTrainingSet/input/images', help='The path to the directory that contains the images.')
    parser.add_argument('--mask_path', type=str, default='generateTrainingSet/input/masks', help='The path to the directory that contains the masks.')

    # Define the output path
    parser.add_argument('--output_path', type=str, default='generateTrainingSet/output', help='The path to the directory that will contain the training set.')

    # Define the number of examples to generate for each deformation type
    N_tension = 1250
    N_compression = 1250
    N_rigid = 1250
    parser.add_argument('--N_tension', type=int, default=N_tension, help='The number of examples to generate for images in tension.')
    parser.add_argument('--N_compression', type=int, default=N_compression, help='The number of examples to generate for images in compression.')
    parser.add_argument('--N_rigid', type=int, default=N_rigid, help='The number of examples to generate for images in rigid motion.')

    # Define the noise level
    parser.add_argument('--noise', type=float, default=10.0, help='The noise level to add to the second image in the training set.')

    #%% Define the parameters for the tension and comopression deformation
    # Define the minimum and maximum values of the longitudinal strain
    min_epsilon_xx = 2.0  / 100.0 # 2.0 % strain
    max_epsilon_xx = 20.0 / 100.0 # 20.0 % strain
    parser.add_argument('--min_epsilon_xx', type=float, default=min_epsilon_xx, help='The minimum value of epsilon_xx.')
    parser.add_argument('--max_epsilon_xx', type=float, default=max_epsilon_xx, help='The maximum value of epsilon_xx.')
    # The minimum and maximum values of longitudinal strain will be reached in how many frames
    min_num_frames = 5
    max_num_frames = 15
    parser.add_argument('--min_num_frames', type=int, default=min_num_frames, help='The minimum number of frames to reach the maximum value of epsilon_xx.')
    parser.add_argument('--max_num_frames', type=int, default=max_num_frames, help='The maximum number of frames to reach the maximum value of epsilon_xx.')
    # Define the minimum and maximum values of the Poisson's ratio
    min_nu = 0.5
    max_nu = 0.5
    parser.add_argument('--min_nu', type=float, default=min_nu, help='The minimum value of nu.')
    parser.add_argument('--max_nu', type=float, default=max_nu, help='The maximum value of nu.')

    #%% Define the parameters for the tension and comopression deformation 
    # Define the minimum and maximum values for the rotation angle
    min_rotation_angle = 0.0
    max_rotation_angle = 2.0
    parser.add_argument('--min_rotation_angle', type=float, default=min_rotation_angle, help='The minimum value of the rotation angle.')
    parser.add_argument('--max_rotation_angle', type=float, default=max_rotation_angle, help='The maximum value of the rotation angle.')
    # Define the minimum and maximum values for the displacement
    min_displacement = 0.0
    max_displacement = 2.0
    parser.add_argument('--min_displacement', type=float, default=min_displacement, help='The minimum value of the displacement.')
    parser.add_argument('--max_displacement', type=float, default=max_displacement, help='The maximum value of the displacement.')

    #%% Define whether to visualize the data
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the data.')

    #%% Define the parameters for finalized training set 
    # Define the name of the training set
    # The training set name will include the number of examples for each deformation type
    train_set_name = 'train_set_N_tension_' + str(N_tension) + '_N_compression_' + str(N_compression) + '_N_rigid_' + str(N_rigid)
    # Tack on a timestamp
    # train_set_name = train_set_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument('--training_set_name', type=str, default=train_set_name, help='The name of the training set.')
    # Define the path to the directory that contains the finalized training set
    parser.add_argument('--finalized_training_set_dir', type=str, default='datasets', help='The path to the directory to save the finalized training set.')
    # Define the percentage of the training set to use for training
    parser.add_argument('--training_percentage', type=float, default=0.8, help='The percentage of the training set to use for training.')

    # Parse the arguments
    args = parser.parse_args()

    # Loop through the arguments and print them
    print('================================================================')
    print('======================== Arguments: ============================')
    print('================================================================')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('================================================================')
    print('================================================================')
    print('================================================================')

    # Create the output directory for the deformation type
    print(args.output_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Save the args as a xlsx file
    args_df = pd.DataFrame.from_dict(vars(args), orient='index')
    args_df.to_excel(args.output_path + '/args.xlsx')
    
    return args

