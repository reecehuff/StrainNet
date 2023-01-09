#%% Imports

#-- Scripts 
from core.arguments import eval_args
from core.evaluater import eval_model
from core import utils

#%% Define the main function for evaluation
def main(args):
    
    # Set the random seed
    utils.set_random_seeds(args.seed)
    
    # There are two options for evaluation:
    # 1. Evaluate on the validation set (default)
    # 2. Evaluate on sequential set of test images, such as the synthetic test cases
    # If you wish to apply StrainNet to experimentally collected images (where the strain is unknown)
    # then you can see apply2experimental.py
    args = utils.get_eval_data_dirs(args)
    eval_model(args, args.sequential)

# Run the main function
if __name__ == "__main__":

    # Arguments for evaluation
    args = eval_args()

    # Run the main function
    main(args)