#%% Imports
import numpy as np
from scipy import ndimage

# Define a function that applies a random tension to an image about the mask (if the mask is not None)
def applyTension(img, mask=None):
    """
    randomTension applies a random tension to an image about the mask.
    
    The inputs are:
        img - The image to apply the tension to.
        mask - The mask to apply the tension about.

    The outputs are:
        displacement_field - The displacement field that was applied to the image (numpy array of the same size as the image x 2)
        strain_xx - The xx strain (numpy array of the same size as the image).
        strain_yy - The yy strain (numpy array of the same size as the image).
        strain_xy - The xy strain (numpy array of the same size as the image).

    """
    # Assert the image is the same size as the mask
    if mask is not None:
        assert img.shape == mask.shape, 'The image and mask must be the same size.'

    
    # Define a displacement field that is applied to the image
    

    # Use the equation for the displacement field to calculate the equation for the strain field


    return displacement_field, strain_xx, strain_yy, strain_xy

