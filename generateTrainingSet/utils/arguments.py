#%% Imports
import argparse

def gatherArguments():

    # Gather the arguments from the command line
    parser = argparse.ArgumentParser(description='Generate a training set for StrainNet.')

    parser.add_argument('--input', type=str, default='input', help='The input directory.')
    parser.add_argument('--output', type=str, default='output', help='The output directory.')
    parser.add_argument('--num', type=int, default=100, help='The number of images to generate.')
    parser.add_argument('--min_epsilon', type=float, default=0.1, help='The minimum epsilon value.')

    # Parse the arguments
    args = parser.parse_args()

    return args

