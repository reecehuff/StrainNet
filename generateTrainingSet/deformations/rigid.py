#%% Imports
import numpy as np

#%% Define a set of classes for the rigid body motions

class rotation: # Rigid 1

    # Define a function that calculates the displacement in the x and y directions
    @staticmethod
    def calculateDisplacement(self):

        # Calculate the displacement in the x and y directions
        displacement_x = self.u_x
        displacement_y = self.u_y

        # Add rotation to the displacement about the centroid of the mask
        displacement_x = displacement_x * np.cos(np.deg2rad(self.rotation)) - displacement_y * np.sin(np.deg2rad(self.rotation))
        displacement_y = displacement_x * np.sin(np.deg2rad(self.rotation)) + displacement_y * np.cos(np.deg2rad(self.rotation))

        return displacement_x, displacement_y

    # Define a function that calculates the strain in the xx, yy and xy direction
    @staticmethod
    def calculateStrain(self):

        # Calculate the strain in the xx, yy and xy directions
        strain_xx = 0
        strain_yy = 0
        strain_xy = 0

        # Convert strain to percent
        strain_xx = strain_xx * 100
        strain_yy = strain_yy * 100
        strain_xy = strain_xy * 100

        return strain_xx, strain_yy, strain_xy

class quadratic: # Rigid 2 (note: img1 and img2 will both be the image warped by this displacement field)

    # Define a function that calculates the displacement in the x and y directions
    @staticmethod
    def calculateDisplacement(self):

        # Unpack the inputs
        epsilon_xx_center = self.epsilon_xx_center
        nu = self.nu
        x_c = self.x_c
        y_c = self.y_c
        a = self.a
        x = self.x
        y = self.y

        # Calculate the displacement in the x and y directions
        displacement_x = a*(x-x_c)*(y-y_c)**2 + epsilon_xx_center*(x-x_c)
        displacement_y = -nu * ( (a/3)*((y-y_c)**3) + epsilon_xx_center*(y-y_c) )

        return displacement_x, displacement_y

    # Define a function that calculates the strain in the xx, yy and xy direction
    @staticmethod
    def calculateStrain(self):

        # Calculate the strain in the xx, yy and xy directions
        strain_xx = 0
        strain_yy = 0
        strain_xy = 0

        # Convert strain to percent
        strain_xx = strain_xx * 100
        strain_yy = strain_yy * 100
        strain_xy = strain_xy * 100

        return strain_xx, strain_yy, strain_xy

class linear: # Rigid 3 (note: img1 and img2 will both be the image warped by this displacement field)

    # Define a function that calculates the displacement in the x and y directions
    @staticmethod
    def calculateDisplacement(self):

        # Unpack the inputs
        x = self.x
        y = self.y
        nu = self.nu
        y_c = self.y_c
        epsilon_xx_proximal = self.epsilon_xx_proximal
        epsilon_xx_distal = self.epsilon_xx_distal
        w = self.W

        # Calculate the displacement in the x and y directions
        displacement_x = epsilon_xx_distal * x + (epsilon_xx_proximal - epsilon_xx_distal) * (x ** 2 / (2 * w))
        displacement_y = - nu * (epsilon_xx_distal * (y - y_c) + (epsilon_xx_proximal - epsilon_xx_distal) * (x * (y - y_c) / w))

        return displacement_x, displacement_y

    # Define a function that calculates the strain in the xx, yy and xy direction
    @staticmethod
    def calculateStrain(self):

        # Calculate the strain in the xx, yy and xy directions
        strain_xx = 0
        strain_yy = 0
        strain_xy = 0

        # Convert strain to percent
        strain_xx = strain_xx * 100
        strain_yy = strain_yy * 100
        strain_xy = strain_xy * 100

        return strain_xx, strain_yy, strain_xy