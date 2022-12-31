#%% Imports
import numpy as np
import os 
from tqdm import tqdm

#-- Scripts 
from utils.arguments import generate_args
import utils.utils as utils
from deformations.tension import applyTension as tension
# from deformations.compression import applyCompression as compression
# from deformations.rigid import applyRigid as rigid
from utils.processTrainingSet import * 

#%% Main function
def main(args):

    # Define the COUNT variable
    # The COUNT variable is used to keep track of the number of examples that have been generated
    COUNT = 1

    # Gather the paths of the images and the masks that will be used to generate the training set
    path2images = utils.gatherPaths(args.image_path)
    path2masks  = utils.gatherPaths(args.mask_path)

    # Check that the names of the images and masks are the same
    utils.checkImageMaskFilesNames(path2images, path2masks)    

    #%% Generate the training set for images in tension 

    # Define indices to select the images and masks uniformly to use to generate the training set
    # If N is greater than the number of images, then the images will be repeated
    indices = np.random.choice(len(path2images), args.N_tension)

    # Use tqdm to display a progress bar of the images that are being processed
    for i in tqdm(range(args.N_tension), desc='Generating training set for images in tension'):

        # Load the image and mask
        img, mask = utils.loadImageAndMask(path2images[indices[i]], path2masks[indices[i]])

        # Define a random tension field to apply to the image
        displacement, strain = tension(img, mask, args)

        # Warp the image using the displacement field
        img1, img2 = utils.imwarp(img, displacement)

        # Add noise to the second images (if desired)
        if args.noise > 0.0:
            img2 = utils.addNoise(img2, args.noise)

        # Save the images, displacement field, and strain fields
        tension_path = os.path.join(args.output_path, 'tension')
        COUNT = utils.saveData(img1, img2, displacement, strain, tension_path, COUNT)

        # If desired, visualize the image, mask, displacement field, and strain fields
        if args.visualize:
            utils.visualizeData(img1, img2, displacement, strain, tension_path, COUNT-1)

# Run the main function
if __name__ == '__main__':
    # Generate the arguments
    args = generate_args()

    # Run the main function and generate the training set
    main(args)

    # The training set has been generated
    print('The training set has been generated.')

    # The final step is process the training set by saving the images and strains 
    # to a final directory that will be used to train the neural network
    # At this point, the set is split into two directories: training and validation
    path2TrainingSet = os.path.join(args.output_path, 'tension')
    path2FinalTrainingSet = os.path.join(args.finalized_training_set_dir, args.training_set_name)

    copyDataAndSplitIntoTrainingAndValidation(path2TrainingSet, path2FinalTrainingSet, args.training_percentage)
