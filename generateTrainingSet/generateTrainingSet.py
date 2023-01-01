#%% Imports
import numpy as np
import os 
from tqdm import tqdm

#-- Scripts 
from core.arguments import generate_args
import core.utils as utils
from deformations.tension import applyTension as tension
from deformations.compression import applyCompression as compression
from deformations.rigid import applyRigid as rigid
from core.processTrainingSet import * 

# Define a function that will create a training set for a given deformation type
def createTrainingSet(deformation_type, path2images, path2masks, args, COUNT):
    
    #%% Generate the training set for images for a given deformation type 

    # Define indices to select the images and masks uniformly to use to generate the training set
    # If N is greater than the number of images, then the images will be repeated
    if deformation_type == 'tension':
        num_deformations = args.N_tension
    elif deformation_type == 'compression':
        num_deformations = args.N_compression
    elif deformation_type == 'rigid':
        num_deformations = args.N_rigid
    else:
        raise ValueError('The deformation type must be either "tension", "compression", or "rigid"')

    indices = sorted(np.random.choice(len(path2images), num_deformations))

    print(' ' * 5)
    # Use tqdm to display a progress bar of the images that are being processed
    for i in tqdm(range(num_deformations), desc='Generating training set for images in ' + deformation_type):

        # Load the image and mask
        img, mask = utils.loadImageAndMask(path2images[indices[i]], path2masks[indices[i]])

        # Define a random field to apply to the image
        if deformation_type == 'tension':
            displacement, strain = tension(img, mask, args)
        elif deformation_type == 'compression':
            displacement, strain = compression(img, mask, args)
        elif deformation_type == 'rigid':
            displacement, strain = rigid(img, mask, args)
        else:
            raise ValueError('The deformation type must be either "tension", "compression", or "rigid"')

        # Warp the image using the displacement field
        img1, img2 = utils.imwarp(img, displacement)

        # Add noise to the second images (if desired)
        if args.noise > 0.0:
            img2 = utils.addNoise(img2, args.noise)

        # Save the images, displacement field, and strain fields
        output_path = os.path.join(args.output_path, deformation_type)
        COUNT = utils.saveData(img1, img2, displacement, strain, output_path, COUNT)

        # If desired, visualize the image, mask, displacement field, and strain fields
        if args.visualize:
            utils.visualizeData(img1, img2, displacement, strain, output_path, COUNT-1)

    return COUNT

#%% Main function
def main(args):

    # Set the random seed
    np.random.seed(args.seed)

    # Define the COUNT variable
    # The COUNT variable is used to keep track of the number of examples that have been generated
    COUNT = 1

    # Gather the paths of the images and the masks that will be used to generate the training set
    path2images = utils.gatherPaths(args.image_path)
    path2masks  = utils.gatherPaths(args.mask_path)

    # Check that the names of the images and masks are the same
    utils.checkImageMaskFilesNames(path2images, path2masks)    

    # Generate the training set for images in compression
    COUNT = createTrainingSet('compression', path2images, path2masks, args, COUNT)
   
    # Generate the training set for images in rigid
    COUNT = createTrainingSet('rigid', path2images, path2masks, args, COUNT)

    # Generate the training set for images in tension 
    COUNT = createTrainingSet('tension', path2images, path2masks, args, COUNT)

    # The training set has been generated
    print(' ' * 5)
    print('The training set has been generated.')
    print(' ' * 5)

    # The final step is process the training set by saving the images and strains 
    # to a final directory that will be used to train the neural network
    # At this point, the set is split into two directories: training and validation
    print('The training set is being processed.')
    print(' ' * 5)
    path2TrainingSet = os.path.join(args.output_path)
    path2FinalTrainingSet = os.path.join(args.finalized_training_set_dir, args.training_set_name)

    copyDataAndSplitIntoTrainingAndValidation(path2TrainingSet, path2FinalTrainingSet, args.training_percentage)

    print('The training set has been processed.')
    print(' ' * 5)

# Run the main function
if __name__ == '__main__':

    # Generate the arguments
    args = generate_args()

    # Run the main function and generate the training set
    main(args)