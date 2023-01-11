#%% Imports
import copy

#-- Scripts 
from core.arguments import train_args
from core.trainer import train_model
from core import utils

#%% Define the main function for training
def main(args):
    
    # Set the random seed
    utils.set_random_seeds(args.seed)

    # Isolate the model types from the arguments
    model_types = args.model_types

    # Save the initial arguments
    init_args = copy.deepcopy(args)

    # Train each of the models 
    if args.train_all:
        for model_type in model_types:
            # Print a message to the user
            print("Training " + model_type + "...")
            # The args change throughout training so we need to reset them
            args = copy.deepcopy(init_args)
            # Set the model type
            args.model_type = model_type
            # Get the data directories
            args = utils.get_data_dirs(args)
            # Train the model
            train_model(args, args.model_type)

    # Or train a single model
    else:
        # Print a message to the user
        print("Training " + args.model_type + "...")
        # Get the data directories
        args = utils.get_data_dirs(args)
        # Train the model
        train_model(args, args.model_type)
    
    # Print a message to the user
    print("Training complete!")

# Run the main function
if __name__ == "__main__":

    # Arguments for training
    args = train_args()

    # Run the main function
    main(args)


