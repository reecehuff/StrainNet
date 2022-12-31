#%% Imports
import argparse

def generate_args():

    # Gather the arguments from the command line
    parser = argparse.ArgumentParser(description='Generate a training set for StrainNet.')

    # Image and mask paths
    parser.add_argument('--image_path', type=str, default='generateTrainingSet/input/images', help='The path to the directory that contains the images.')
    parser.add_argument('--mask_path', type=str, default='generateTrainingSet/input/masks', help='The path to the directory that contains the masks.')

    # Define the output path
    parser.add_argument('--output_path', type=str, default='generateTrainingSet/output', help='The path to the directory that will contain the training set.')

    # Define the number of examples to generate for each deformation type
    parser.add_argument('--N_tension', type=int, default=10, help='The number of examples to generate for images in tension.')
    parser.add_argument('--N_compression', type=int, default=10, help='The number of examples to generate for images in compression.')
    parser.add_argument('--N_rigid', type=int, default=10, help='The number of examples to generate for images in rigid motion.')

    # Define the noise level
    parser.add_argument('--noise', type=float, default=10.0, help='The noise level to add to the second image in the training set.')

    #%% Define the parameters for the tension deformation
    parser.add_argument('--min_epsilon_xx', type=float, default=0.05, help='The minimum value of epsilon_xx.')
    parser.add_argument('--max_epsilon_xx', type=float, default=0.1, help='The maximum value of epsilon_xx.')
    parser.add_argument('--min_nu', type=float, default=0.5, help='The minimum value of nu.')
    parser.add_argument('--max_nu', type=float, default=0.5, help='The maximum value of nu.')

    # Define whether to visualize the data
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the data.')

    #%% Define the parameters for finalized training set 
    # Make sure the name of the training set is required
    parser.add_argument('--training_set_name', type=str, required=True, help='The name of the training set.')
    # Define the path to the directory that contains the finalized training set
    parser.add_argument('--finalized_training_set_dir', type=str, default='datasets/', help='The path to the directory to save the finalized training set.')
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

    return args

