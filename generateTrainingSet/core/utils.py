##%% Imports
import glob
import cv2
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#%% Functions

# Define a class for getting things
class get(object):

    @staticmethod
    def paths(path):
        """
        getPaths gets the paths to the images or masks.
        """

        # Gather the paths to the images or masks
        output = glob.glob(path + '/*')

        # Sort the paths
        output.sort()

        return output        

# Define a class for checking things
class check(object):
    
    @staticmethod
    def image_and_mask_file_names(path2images, path2masks):
        """
        checkImageAndMaskFileNames checks that the names of the images and masks are the same.
        """

        # Check that the names of the images and masks are the same
        for i in range(len(path2images)):
            assert os.path.basename(path2images[i]).split('.')[0] == os.path.basename(path2masks[i]).split('.')[0], 'The names of the images and masks must be the same.'

class load(object):
    
    @staticmethod
    def image_and_mask(path2image, path2mask):
        """
        loadImageAndMask loads the image and mask.
        
        The inputs are:
            path2image - The path to the image (string).
            path2mask - The path to the mask (string).
    
        The outputs are:
            img - The image (numpy array).
            mask - The mask (numpy array).
    
        """
        # Load the image and mask
        img = cv2.imread(path2image, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(path2mask, cv2.IMREAD_GRAYSCALE)
    
        # Convert the image and mask to float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
    
        # Convert the mask to a binary mask
        mask[mask > 0] = 1
    
        return img, mask

# Define a function that saves the images, displacement field, and strain fields
def saveData(img1, img2, displacement_field, strain, path2save, COUNT):
    """
    saveData saves the images, displacement field, and strain fields.
    
    The inputs are:
        img1 - The first image (numpy array).
        img2 - The second image (numpy array).
        displacement_field - The displacement field (numpy array).
        strain - The strain field (numpy array).
        path2save - The path to save the images, displacement field, and strain fields (string).
        COUNT - The number of the image pair (integer).

    The outputs are:
        COUNT - The number of the image pair (integer) (incremented by 1).

    """

    # Define the COUNT variable as a string with 4 digits
    COUNT_str = str(COUNT).zfill(4)

    # Create the directory to save the images, displacement field, and strain fields
    if not os.path.exists(path2save):
        os.makedirs(path2save)

    # Create the directory to save the images
    imagePath = os.path.join(path2save, 'images')
    if not os.path.exists(imagePath):
        os.makedirs(imagePath)
    
    # Create the directory to save the displacement field
    dispPath = os.path.join(path2save, 'displacements')
    if not os.path.exists(dispPath):
        os.makedirs(dispPath)

    # Create the directory to save the strain fields
    strainPath = os.path.join(path2save, 'strains')
    if not os.path.exists(strainPath):
        os.makedirs(strainPath)

    # Save the images
    im1_path = os.path.join(imagePath, COUNT_str + '_im1.png')
    im2_path = os.path.join(imagePath, COUNT_str + '_im2.png')
    cv2.imwrite(im1_path, img1)
    cv2.imwrite(im2_path, img2)

    # Save the displacement field as a mat file
    path2displacements_X = os.path.join(dispPath, COUNT_str + '_displacement_X.npy')
    path2displacements_Y = os.path.join(dispPath, COUNT_str + '_displacement_Y.npy')
    # Save each component of the displacement field separately as a numpy array
    np.save(path2displacements_X, displacement_field[:, :, 0])
    np.save(path2displacements_Y, displacement_field[:, :, 1])

    # Save the strain fields as a mat file
    path2strains_XX = os.path.join(strainPath, COUNT_str + '_strain_xx.npy')
    path2strains_YY = os.path.join(strainPath, COUNT_str + '_strain_yy.npy')
    path2strains_XY = os.path.join(strainPath, COUNT_str + '_strain_xy.npy')
    # Save each component of the strain field separately as a numpy array
    np.save(path2strains_XX, strain[:, :, 0])
    np.save(path2strains_YY, strain[:, :, 1])
    np.save(path2strains_XY, strain[:, :, 2])

    return COUNT + 1

# Define a function that visualizes the images, displacement field, and strain fields
def visualizeData(img1, img2, displacement_field, strain, path2save, COUNT):
    """
    visualizeData visualizes the images, displacement field, and strain fields.
    
    The inputs are:
        img1 - The first image (numpy array).
        img2 - The second image (numpy array).
        displacement_field - The displacement field (numpy array).
        strain - The strain field (numpy array).
        path2save - The path to save the images, displacement field, and strain fields visualization (string).
        COUNT - The count of the images, displacement field, and strain fields (integer).
    """

    # Define the COUNT variable as a string with 4 digits
    COUNT_str = str(COUNT).zfill(4)

    # Create the directory to save the images, displacement field, and strain fields visualization
    visualizePath = os.path.join(path2save, 'visualize')
    if not os.path.exists(visualizePath):
        os.makedirs(visualizePath)

    # Create a massive matplotlib figure
    plt.figure(figsize=(20, 10))

    # Add the first image 
    plt.subplot(2, 4, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('First image, I$_1$', fontsize=20)
    plt.axis('off')
    plt.colorbar().set_label(label='Intensity', size=15)

    # Add the second image
    plt.subplot(2, 4, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Second image, I$_2$', fontsize=20)
    plt.axis('off')
    plt.colorbar().set_label(label='Intensity', size=15)

    # Add the displacement field
    plt.subplot(2, 4, 5)
    plt.imshow(displacement_field[:, :, 0], cmap='viridis')
    plt.title('Displacement field ($u_x$)', fontsize=20)
    plt.axis('off')
    plt.colorbar().set_label(label='Displacement [pixels]', size=15)

    plt.subplot(2, 4, 6)
    plt.imshow(displacement_field[:, :, 1], cmap='viridis')
    plt.title('Displacement field ($u_y$)', fontsize=20)
    plt.axis('off')
    plt.colorbar().set_label(label='Displacement [pixels]', size=15)

    # Add the strain fields
    plt.subplot(2, 4, 3)
    plt.imshow(strain[:, :, 0], cmap='viridis')
    plt.title('Strain field ($\epsilon_{xx}$)', fontsize=20)
    plt.axis('off')
    plt.colorbar().set_label(label='$\epsilon_{xx}$ [%]', size=15)
    
    plt.subplot(2, 4, 4)
    plt.imshow(strain[:, :, 1], cmap='viridis')
    plt.title('Strain field ($\epsilon_{yy}$)', fontsize=20)
    plt.axis('off')
    plt.colorbar().set_label(label='$\epsilon_{yy}$ [%]', size=15)

    plt.subplot(2, 4, 7)
    plt.imshow(strain[:, :, 2], cmap='viridis')
    plt.title('Strain field ($\epsilon_{xy}$)', fontsize=20)
    plt.axis('off')
    plt.colorbar().set_label(label='$\epsilon_{xy}$ [%]', size=15)

    plt.subplot(2, 4, 8)
    # Add a little image of our coordinate system
    # x-axis is red, y-axis is green, and the origin is blue
    H, W = img1.shape
    dumby_image = np.ones((H, W))
    dumby_image[0, 0] = 0
    plt.imshow(dumby_image, cmap='gray')
    plt.arrow(0, 0, W, 0, head_width=10, head_length=10, fc='r', ec='r', linewidth=3)
    plt.arrow(0, 0, 0, H, head_width=10, head_length=10, fc='g', ec='g', linewidth=3)
    # Add the x and y labels
    plt.text(W, H/10, 'x', color='r', fontsize=20)
    plt.text(H/10, H, 'y', color='g', fontsize=20)

    # Add a circle to represent the origin
    circle = plt.Circle((0, 0), 5, color='b')
    plt.gca().add_patch(circle)

    plt.title('Coordinate system', fontsize=20)
    plt.axis('off')

    # Save the figure
    path2figure = os.path.join(visualizePath, COUNT_str + '_visualization.png')
    plt.savefig(path2figure, bbox_inches='tight')

    # Close the figure
    plt.close()

# Define a function setting the random seeds
def set_random_seeds(seed):
    # Set the random seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)