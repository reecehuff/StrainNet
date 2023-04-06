#%% Imports
import numpy as np
import os 
from tqdm import tqdm

#-- Scripts 
from core import utils
from core.arguments import generate_args
from core.processTrainingSet import copyDataAndSplitIntoTrainingAndValidation, augment_training_set
import deformations

# Define a function that will create a training set for a given deformation type
def createTrainingSet(deformation_type, path2images, path2masks, args, COUNT):

    #%% Begin by initializing the deformation maker
    # Load a random image and mask
    print(path2images[0])
    img, mask = utils.load.image_and_mask(path2images[0], path2masks[0])
    deformation_maker = deformations.deformation_maker(img, mask, deformation_type, COUNT, args)

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

    # We divide the number of images by 2 if the double_warp flag is set to True
    # This is because we will be generating two examples for each image we read in the for loop
    if args.double_warp:
        num_deformations = np.round(num_deformations // 2).astype(int)

    # Select the indices of the images and masks to use to generate the training set at random
    indices = sorted(np.random.choice(len(path2images), num_deformations))

    print(' ' * 5)
    # Use tqdm to display a progress bar of the images that are being processed
    for i in tqdm(range(num_deformations), desc='Generating training set for images in ' + deformation_type):

        # Load the image and mask
        img, mask = utils.load.image_and_mask(path2images[indices[i]], path2masks[indices[i]])

        # Update the deformation maker with the new image and mask
        # as well as the deformation type, the count, and the arguments
        deformation_maker.update(img, mask, deformation_type, COUNT, args)

        # Get the displacement field and strain field
        displacement, strain = deformation_maker.deformation

        # Get the images
        img1, img2 = deformation_maker.images

        # Make sure that the images, displacement field, and strain field are the correct size
        assert img1.shape == img2.shape == displacement.shape[:2] == strain.shape[:2] == (args.output_height, args.output_width)

        # Save the images, displacement field, and strain fields
        output_path = os.path.join(args.output_path, deformation_type)
        COUNT = utils.saveData(img1, img2, displacement, strain, output_path, COUNT)

        # If desired, visualize the image, mask, displacement field, and strain fields
        if args.visualize:
            utils.visualizeData(img1, img2, displacement, strain, output_path, COUNT-1)

        # Increment the count
        deformation_maker.COUNT = COUNT

        # If desired, you may also include a double warped image in the training set for robustness on synthetic data
        if args.double_warp:

            # The input image will now be a warped image
            img = deformation_maker.im2.copy()

            # Update the deformation maker with the new image and mask
            # as well as the deformation type, the count, and the arguments
            deformation_maker.update(img, mask, deformation_type, COUNT, args)

            # Get the displacement field and strain field
            displacement, strain = deformation_maker.deformation

            # Get the images
            img1, img2 = deformation_maker.images

            # Make sure that the images, displacement field, and strain field are the correct size
            assert img1.shape == img2.shape == displacement.shape[:2] == strain.shape[:2] == (args.output_height, args.output_width)

            # Save the images, displacement field, and strain fields
            output_path = os.path.join(args.output_path, deformation_type)
            COUNT = utils.saveData(img1, img2, displacement, strain, output_path, COUNT)

            # If desired, visualize the image, mask, displacement field, and strain fields
            if args.visualize:
                utils.visualizeData(img1, img2, displacement, strain, output_path, COUNT-1)

            # Increment the count
            deformation_maker.COUNT = COUNT

    return COUNT

#%% Main function
def main(args):

    # Set random seeds
    utils.set_random_seeds(args.seed)

    # Define the COUNT variable
    # The COUNT variable is used to keep track of the number of examples that have been generated
    COUNT = 1

    # Gather the paths of the images and the masks that will be used to generate the training set
    path2images = utils.get.paths(args.image_path)
    path2masks  = utils.get.paths(args.mask_path)

    # If you only wish to use a subset of the images and masks
    if args.image_mask_subset == "on":
        path2images = ['generateTrainingSet/input/images/10mvc_trial1_0001.png']
        path2masks  = ['generateTrainingSet/input/masks/10mvc_trial1_0001.png']
        
    # Check that the names of the images and masks are the same
    utils.check.image_and_mask_file_names(path2images, path2masks)    

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

    # Augment the training set for the DeformationClassifier if you wish
    if args.augment:
        augment_training_set(path2FinalTrainingSet,args)

    print('The training set has been processed.')
    print(' ' * 5)

# Run the main function
if __name__ == '__main__':

    # Generate the arguments
    args = generate_args()

    # Run the main function and generate the training set
    main(args)