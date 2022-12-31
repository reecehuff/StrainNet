#%% Imports
import numpy as np
from scipy import ndimage

# Define a function that applies a random tension to an image about the mask (if the mask is not None)
def applyTension(img, mask, args):
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
    assert img.shape == mask.shape, 'The image and mask must be the same size.'

    # Determine the height and width of the image
    H, W = img.shape

    # Initialize the displacement field and strain fields
    displacement_field = np.zeros((H, W, 2))
    strain = np.zeros((H, W, 3))

    # Determine the centroid of the mask
    mask_centroid = ndimage.measurements.center_of_mass(mask)
    x_c = mask_centroid[1]
    y_c = mask_centroid[0]

    # Determine the indices of the mask
    mask_indices = np.where(mask == 1)

    # Calculate the height and width of the mask
    mask_height = np.max(mask_indices[0]) - np.min(mask_indices[0])
    mask_width = np.max(mask_indices[1]) - np.min(mask_indices[1])

    # Gather random tension parameters and save them in a dictionary
    params = randomTensionParameters(x_c, y_c, mask_height, mask_width, args)

    # Loop through the indices of the mask
    for i in range(len(mask_indices[0])):

        # Determine the x and y indices of the mask
        x = mask_indices[1][i]
        y = mask_indices[0][i]

        # Calculate the displacement in the x and y directions
        displacement_x, displacement_y = calculateDisplacement(x, y, params)

        # Calculate the strain in the xx, yy, and xy directions
        strain_xx, strain_yy, strain_xy = calculateStrain(x, y, params)

        # Save the displacement in the displacement field and the strain in the strain field

        displacement_field[y, x, 0] = displacement_x
        displacement_field[y, x, 1] = displacement_y
        strain[y, x, 0] = strain_xx
        strain[y, x, 1] = strain_yy
        strain[y, x, 2] = strain_xy

    return displacement_field, strain

#%% Randomly selected parameters
def randomTensionParameters(x_c, y_c, mask_height, mask_width, args):
    """
    randomTensionParameters returns a dictionary of random tension parameters.

    The inputs are:
        x_c - The x coordinate of the centroid of the mask.
        y_c - The y coordinate of the centroid of the mask.
        mask_height - The height of the mask.
        mask_width - The width of the mask.

    The outputs are:
        params - A dictionary of random tension parameters.

    """
    # Define the random parameters
    epsilon_xx_center = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)
    epsilon_xx_edge = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)
    nu = np.random.uniform(args.min_nu, args.max_nu)

    # Calculate a, which is based on the height of the mask and the random parameters
    a = ( 4*(epsilon_xx_edge - epsilon_xx_center) ) / (mask_height**2)

    # Save the parameters in a dictionary
    params = {'epsilon_xx_center': epsilon_xx_center,
              'epsilon_xx_edge': epsilon_xx_edge,
              'nu': nu, 
              'x_c': x_c,
              'y_c': y_c,
              'mask_height': mask_height,
              'mask_width': mask_width, 
              'a': a
              }

    return params

# Define a function that calculates the displacement in the x and y directions given the x and y indices of the mask and the centroid of the mask
def calculateDisplacement(x,y,params):

    # Unpack the parameters
    epsilon_xx_center = params['epsilon_xx_center']
    epsilon_xx_edge = params['epsilon_xx_edge']
    nu = params['nu']
    x_c = params['x_c']
    y_c = params['y_c']
    mask_height = params['mask_height']
    a = params['a']

    # Calculate the displacement in the x and y directions
    displacement_x = a*(x-x_c)*(y-y_c)**2 + epsilon_xx_center*(x-x_c)
    displacement_y = -nu * ( (a/3)*((y-y_c)**3) + epsilon_xx_center*(y-y_c) )

    return displacement_x, displacement_y

# Define a function that calculates the strain in the xx, yy and xy directions given the x and y indices of the mask and the centroid of the mask
def calculateStrain(x,y,params):

    # Unpack the parameters
    epsilon_xx_center = params['epsilon_xx_center']
    epsilon_xx_edge = params['epsilon_xx_edge']
    nu = params['nu']
    x_c = params['x_c']
    y_c = params['y_c']
    a = params['a']

    # Calculate the strain in the xx, yy and xy directions
    strain_xx = a*(y-y_c)**2 + epsilon_xx_center
    strain_yy = -nu*(a*(y - y_c)**2 + epsilon_xx_center)
    strain_xy = a*(x-x_c)*(y - y_c)

    return strain_xx, strain_yy, strain_xy