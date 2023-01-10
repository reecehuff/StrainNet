#%% Imports
import numpy as np

#%% Define a set of classes for the rigid body motions

class one: # Rigid 1

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

class two: # Rigid 2

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