#%% Imports
import numpy as np

#-- Scripts 
from utils.arguments import gatherArguments
from deformations.tension import applyTension as tension
from deformations.compression import applyCompression as compression
from deformations.rigid import applyRigid as rigid

#%% Main function
def main(args):
    
    # Print the arguments
    print(args)

    # Gather the paths of the images (and the masks) that will be used to generate the training set
    path2images = gatherPaths(args.image_path)
    path2masks  = gatherPaths(args.mask_path)

    # Check that the names of the images and masks are the same
    checkImageMaskNames(path2images, path2masks)    

    #%% Generate the training set for images in tension 

    # Define indices to select the images and masks uniformly to use to generate the training set
    indices = np.random.choice(len(path2images), args.N, replace=False)

    for i in range(args.N_tension):

        # Load the image and mask


        # Define a random tension field to apply to the image
        displacement_field, strain_xx, strain_yy, strain_xy = tension(img, mask)

        # Warp the image using the displacement field
        img1, img2 = imwarp(img, displacement_field)

        # Add noise to the second images (if desired)
        if args.noise:
            img2 = addNoise(img2)

        # Save the image, mask, displacement field, and strain fields
        saveData(img1, img2, displacement_field, strain_xx, strain_yy, strain_xy, args.output_path)

        # If desired, visualize the image, mask, displacement field, and strain fields
        if args.visualize:
            visualizeData(img1, img2, displacement_field, strain_xx, strain_yy, strain_xy)

def __main__():
    args = gatherArguments()
    main(args)