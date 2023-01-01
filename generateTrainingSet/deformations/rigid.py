#%% Imports
import numpy as np
from scipy import ndimage

# Define a function that applies a random rigid to an image about the mask (if the mask is not None)
def applyRigid(img, mask, args):
    """
    randomRigid applies a random rigid to an image about the mask.
    
    The inputs are:
        img - The image to apply the rigid to.
        mask - The mask to apply the rigid about.

    The outputs are:
        displacement_field - The displacement field that was applied to the image (numpy array of the same size as the image x 2)
        strain - The strain (numpy array of the same size as the image x 3).

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

    # Gather random rigid parameters and save them in a dictionary
    params = randomRigidParameters(x_c, y_c, args)

    # Determine the x and y indices of the mask
    x = mask_indices[1]
    y = mask_indices[0]

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
def randomRigidParameters(x_c, y_c, args):
    """
    randomRigidParameters returns a dictionary of random rigid parameters.

    The inputs are:
        args - The arguments from the command line.

    The outputs are:
        params - A dictionary of random rigid parameters.

    """
    # Save the parameters in a dictionary
    params = { 'u_x': np.random.uniform(args.min_displacement, args.max_displacement),
               'u_y': np.random.uniform(args.min_displacement, args.max_displacement),
               'min_rotation': args.min_rotation_angle,
               'max_rotation': args.max_rotation_angle,
                'x_c': x_c,
                'y_c': y_c,
               }

    return params

# Define a function that calculates the displacement in the x and y directions given the x and y indices of the mask and the centroid of the mask
def calculateDisplacement(x,y,params):

    # Unpack the parameters
    u_x = params['u_x']
    u_y = params['u_y']
    min_rotation = params['min_rotation']
    max_rotation = params['max_rotation']
    x_c = params['x_c']
    y_c = params['y_c']

    # Calculate the displacement in the x and y directions
    displacement_x = u_x 
    displacement_y = u_y

    # Add rotation to the displacement about the centroid of the mask
    displacement_x = displacement_x * np.cos(np.deg2rad(min_rotation)) - displacement_y * np.sin(np.deg2rad(min_rotation))
    displacement_y = displacement_x * np.sin(np.deg2rad(min_rotation)) + displacement_y * np.cos(np.deg2rad(min_rotation))

    return displacement_x, displacement_y

# Define a function that calculates the strain in the xx, yy and xy direction
def calculateStrain(x,y,params):

    # Calculate the strain in the xx, yy and xy directions
    strain_xx = 0
    strain_yy = 0
    strain_xy = 0

    return strain_xx, strain_yy, strain_xy