#%% Imports
import numpy as np
from scipy import ndimage
import cv2

# Define a function that applies a random deformation to an image about the mask (if the mask is not None)
class deformation_maker(object):
    def __init__(self, img=np.ones([10,10]), mask=np.ones([10,10]), deformation_type="tension", COUNT=1, args=None):

        # Save the inputs
        self.img = img
        self.mask = mask
        self.deformation_type = deformation_type
        self.COUNT = COUNT
        self.args = args

        # # Initialize the deformation function
        self.defineDeformationFunction(deformation_type, COUNT)

        # Initialize the image and mask properties
        self.image_and_mask_properties(img, mask)

        # Initialize the displacement field and strain fields
        self.initialize_displacement_field_and_strain()

        # Initialize the random parameters
        self.randomizeParameters()

    def update(self, img, mask, deformation_type, COUNT, args):

        # Update the image, mask, deformation type, and count
        self.img = img
        self.mask = mask
        self.deformation_type = deformation_type
        self.COUNT = COUNT
        self.args = args

        # Update the random parameters
        self.randomizeParameters()

        # Update the deformation function
        self.defineDeformationFunction(deformation_type, COUNT)

        # Update the image and mask properties
        self.image_and_mask_properties(img, mask)

        # Zeros out the displacement field and strain fields
        self.initialize_displacement_field_and_strain()

        # Update the displacement field and strain fields
        self.calculate()
        
        # Update the output images
        self.im1, self.im2 = self.imwarp(self.img, self.displacement_field)

        # Crop the images, displacement field, and strain
        self.im1_cropped, self.im2_cropped = self.crop(self.im1), self.crop(self.im2)
        self.displacement_field_cropped = self.crop(self.displacement_field)
        self.strain_cropped = self.crop(self.strain)

        # Package the final outputs (the images and the deformation fields)
        self.deformation    = [self.displacement_field_cropped, self.strain_cropped]
        self.images         = [self.im1_cropped, self.im2_cropped]

    def defineDeformationFunction(self, deformation_type, COUNT):

        # For tension
        if deformation_type == 'tension':
            # Import the tension functions
            from . import tension
            # Depending on the count and deformation type, define the deformation function
            count_remainder = COUNT % 2 + 1
            # Define the deformation function
            if count_remainder == 1:
                self.deformation_function = tension.one()
            elif count_remainder == 2:
                self.deformation_function = tension.two()
        
        # For compression 
        elif deformation_type == 'compression':
            # Import the compression functions
            # Depending on the count and deformation type, define the deformation function
            count_remainder = COUNT % 2 + 1
            from . import compression
            # Define the deformation function
            if count_remainder == 1:
                self.deformation_function = compression.one()
            elif count_remainder == 2:
                self.deformation_function = compression.two()

        # For rigid
        elif deformation_type == 'rigid':
            # Import the rigid functions
            from . import rigid
            # Depending on the count and deformation type, define the deformation function
            count_remainder = COUNT % 3 + 1
            # Define the deformation function
            if count_remainder == 1:
                self.deformation_function = rigid.one()
            elif count_remainder == 2:
                self.deformation_function = rigid.two()
            elif count_remainder == 3:
                self.deformation_function = rigid.three()

    def image_and_mask_properties(self, img, mask):

        # Assert the image is the same size as the mask
        assert img.shape == mask.shape, 'The image and mask must be the same size.'

        # Determine the height and width of the image
        self.H, self.W = img.shape

        # Determine the centroid of the mask
        self.mask_centroid = ndimage.measurements.center_of_mass(mask)
        self.x_c = self.mask_centroid[1]
        self.y_c = self.mask_centroid[0]

        # Determine the indices of the mask
        self.mask_indices = np.where(mask == 1)

        # Calculate the height and width of the mask
        self.mask_height = np.max(self.mask_indices[0]) - np.min(self.mask_indices[0])
        self.mask_width = np.max(self.mask_indices[1]) - np.min(self.mask_indices[1])

        # Determine the x and y indices of the mask
        self.x = self.mask_indices[1]
        self.y = self.mask_indices[0]

    def initialize_displacement_field_and_strain(self):

        # Initialize the displacement field and strain fields
        self.displacement_field = np.zeros((self.H, self.W, 2))
        self.strain = np.zeros((self.H, self.W, 3))
        self.deformation = [self.displacement_field, self.strain]

        # Initialize the output images
        self.im1 = np.zeros((self.H, self.W))
        self.im2 = np.zeros((self.H, self.W))
        self.images = [self.im1, self.im2]
    
    def randomizeParameters(self):

        if self.args is None:
            return

        # Randomize the parameters

        # Tension random parameters
        if self.deformation_type == 'tension':
            self.randomTensionParameters()

        # Compression random parameters
        elif self.deformation_type == 'compression':
            self.randomCompressionParameters()

        # Rigid random parameters
        elif self.deformation_type == 'rigid':
            self.randomRigidParameters()

        # Raise an error if the deformation type is not recognized
        else:
            raise ValueError('The deformation type must be either "tension", "compression", or "rigid".')

    def randomTensionParameters(self):

        # Unpack the arguments
        args = self.args
        mask_height = self.mask_height

        # Number of frames for tension
        num_frames = np.random.uniform(args.min_num_frames, args.max_num_frames)

        # Define the parameters for the tensile deformation
        epsilon_xx_center = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)
        epsilon_xx_edge   = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)

        # Define the parameters for the linear compression deformation
        epsilon_xx_distal = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)
        epsilon_xx_proximal = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)

        # Normalize the random parameters
        epsilon_xx_center = epsilon_xx_center / num_frames
        epsilon_xx_edge   = epsilon_xx_edge / num_frames
        epsilon_xx_distal = epsilon_xx_distal / num_frames
        epsilon_xx_proximal = epsilon_xx_proximal / num_frames

        # Poisson's ratio 
        nu = np.random.uniform(args.min_nu, args.max_nu)

        # Calculate a, which is based on the height of the mask and the random parameters
        a = ( 4*(epsilon_xx_edge - epsilon_xx_center) ) / (mask_height**2)

        # Save the new parameters to self
        self.epsilon_xx_center = epsilon_xx_center
        self.epsilon_xx_edge = epsilon_xx_edge
        self.epsilon_xx_distal = epsilon_xx_distal
        self.epsilon_xx_proximal = epsilon_xx_proximal
        self.nu = nu
        self.a = a 

    def randomCompressionParameters(self):

        # Unpack the arguments
        args = self.args
        mask_height = self.mask_height

        # Number of frames for the compression
        num_frames = np.random.uniform(args.min_num_frames, args.max_num_frames)

        # Define the parameters for the quadratic compression deformation
        epsilon_xx_center = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)
        epsilon_xx_edge   = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)

        # Define the parameters for the linear compression deformation
        epsilon_xx_distal = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)
        epsilon_xx_proximal = np.random.uniform(args.min_epsilon_xx, args.max_epsilon_xx)

        # Normalize the strain by the number of frames
        epsilon_xx_center = epsilon_xx_center / num_frames
        epsilon_xx_edge   = epsilon_xx_edge / num_frames
        epsilon_xx_distal = epsilon_xx_distal / num_frames
        epsilon_xx_proximal = epsilon_xx_proximal / num_frames

        # Since its in compression, the strain should be negative
        epsilon_xx_center = -epsilon_xx_center
        epsilon_xx_edge   = -epsilon_xx_edge
        epsilon_xx_distal = -epsilon_xx_distal
        epsilon_xx_proximal = -epsilon_xx_proximal

        # Poisson's ratio 
        nu = np.random.uniform(args.min_nu, args.max_nu)

        # Calculate a, which is based on the height of the mask and the random parameters
        a = ( 4*(epsilon_xx_edge - epsilon_xx_center) ) / (mask_height**2)

        # Save the new parameters to self
        self.epsilon_xx_center = epsilon_xx_center
        self.epsilon_xx_edge = epsilon_xx_edge
        self.epsilon_xx_distal = epsilon_xx_distal
        self.epsilon_xx_proximal = epsilon_xx_proximal
        self.nu = nu
        self.a = a 

    def randomRigidParameters(self):

        # Unpack the arguments
        args = self.args

        # Save the random rigid parameters
        self.u_x            = np.random.uniform(args.min_displacement,      args.max_displacement)
        self.u_y            = np.random.uniform(args.min_displacement,      args.max_displacement)
        self.rotation       = np.random.uniform(args.min_rotation_angle,    args.max_rotation_angle)

        # rigid.two and rigid.three require the tension and compression parameters
        if np.random.uniform() > 0.5:
            self.randomTensionParameters()
        else:
            self.randomCompressionParameters()

    def calculate(self):
        # Calculate the displacement
        self.displacement_x, self.displacement_y = self.deformation_function.calculateDisplacement(self)
        # Calculate the strain
        self.strain_xx, self.strain_yy, self.strain_xy = self.deformation_function.calculateStrain(self)
        # Package the displacement and strain 
        self.displacement_field, self.strain = self.processDisplacementAndStrain(self.displacement_field, self.strain)

    def processDisplacementAndStrain(self, displacement_field, strain):

        # Unpack the arguments
        displacement_field = self.displacement_field
        strain = self.strain
        x = self.x
        y = self.y
        displacement_x = self.displacement_x
        displacement_y = self.displacement_y
        strain_xx = self.strain_xx
        strain_yy = self.strain_yy
        strain_xy = self.strain_xy

        # Save the displacement in the displacement field and the strain in the strain field
        displacement_field[y, x, 0] = displacement_x
        displacement_field[y, x, 1] = displacement_y
        strain[y, x, 0] = strain_xx
        strain[y, x, 1] = strain_yy
        strain[y, x, 2] = strain_xy

        return displacement_field, strain

    # Define a function that warps the image using the displacement field
    def imwarp(self, img, displacement_field):
        """
        imwarp warps the image using the displacement field.
        
        The inputs are:
            img - The image (numpy array).
            displacement_field - The displacement field (numpy array).

        The outputs are:
            img1 - The first image (numpy array).
            img2 - The second image (numpy array).

        """

        # The first image is the original image
        img1 = img
        # The second image is the warped image
        h, w = displacement_field.shape[:2]
        displacement_field_4_warping = np.zeros((h, w, 2), dtype=np.float32)
        displacement_field_4_warping[:,:,0] = -displacement_field[:,:,0]
        displacement_field_4_warping[:,:,1] = -displacement_field[:,:,1]
        displacement_field_4_warping[:,:,0] += np.arange(w)
        displacement_field_4_warping[:,:,1] += np.arange(h)[:,np.newaxis]
        img2 = cv2.remap(img, displacement_field_4_warping, None, cv2.INTER_LINEAR)

        # If the deformation function is a rigid.two or rigid.three
        # Import the necessary functions
        from . import rigid
        if isinstance(self.deformation_function, rigid.two) or isinstance(self.deformation_function, rigid.three):
            # Both images are the same
            img1 = img2.copy()

        # Add noise to the imgs if desired
        if self.args.noise > 0.0:
            img1 = self.add_noise(img1)
            img2 = self.add_noise(img2)

        return img1, img2

    def add_noise(self, img):
        """
        addNoise adds noise to the image.
        
        The inputs are:
            img - The image (numpy array).

        The outputs are:
            img - The image with noise (numpy array).

        """
        noise = self.args.noise

        # Add noise to the image
        img = img + np.random.normal(0, noise, img.shape)

        # Clip the image
        img = np.clip(img, 0, 255)

        return img

    def crop(self, array):
        """
        crop crops the array.
        """
        # Unpack the arguments
        args = self.args
        output_height = args.output_height
        output_width = args.output_width
        upper_left_corner_x = args.upper_left_corner_x
        upper_left_corner_y = args.upper_left_corner_y
    
        # Crop the array
        array = array[upper_left_corner_y:upper_left_corner_y+output_height, upper_left_corner_x:upper_left_corner_x+output_width]

        return array
