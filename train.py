#%% Imports
import numpy as np
import torch 

#-- Scripts 
from core.arguments import train_args
from core.trainer import train_model
from core import utils

#%% Define a function that will create DataSet from image and strain paths 

def main(args):
    
    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Train each of the models 
    if args.train_all:
        for model_type in args.model_types:
            args.model_type = model_type
            args = utils.get_data_dirs(args)
            train_model(args, args.model_type)
    # Or train a single model
    else:
        args = utils.get_data_dirs(args)
        train_model(args, args.model_type)

# Run the main function
if __name__ == "__main__":

    # Arguments for training
    args = train_args()

    # Run the main function
    main(args)


