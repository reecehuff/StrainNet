#%% Imports

#-- Scripts 
from core.arguments import experimental_args
from core.apply import apply_StrainNet
from core import utils

#%% Define the main function for evaluation
def main(args):
    
    # Set random seeds
    utils.set_random_seeds(args.seed)

    # Apply StrainNet to experimental data
    apply_StrainNet(args)

    # Print a message to the user
    print("Done!")

# Run the main function
if __name__ == "__main__":

    # Arguments for evaluation
    args = experimental_args()

    # Run the main function
    main(args)