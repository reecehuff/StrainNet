#%% Define a set of classes for the compressive deformations

class one: # Compression 1

    # Define a function that calculates the displacement in the x and y directions
    @staticmethod
    def calculateDisplacement(self):

        # Unpack the inputs
        x = self.x
        y = self.y
        epsilon_xx_center = self.epsilon_xx_center
        nu = self.nu
        a = self.a
        x_c = self.x_c
        y_c = self.y_c

        # Calculate the displacement in the x and y directions
        displacement_x = a*(x-x_c)*(y-y_c)**2 + epsilon_xx_center*(x-x_c)
        displacement_y = -nu * ( (a/3)*((y-y_c)**3) + epsilon_xx_center*(y-y_c) )

        return displacement_x, displacement_y

    # Define a function that calculates the strain in the xx, yy and xy direction
    @staticmethod
    def calculateStrain(self):

        # Unpack the inputs
        x = self.x
        y = self.y
        epsilon_xx_center = self.epsilon_xx_center
        nu = self.nu
        a = self.a
        x_c = self.x_c
        y_c = self.y_c

        # Calculate the strain in the xx, yy and xy directions
        strain_xx = a*(y-y_c)**2 + epsilon_xx_center
        strain_yy = -nu*(a*(y - y_c)**2 + epsilon_xx_center)
        strain_xy = a*(x-x_c)*(y - y_c)

        # Convert strain to percent
        strain_xx = strain_xx * 100
        strain_yy = strain_yy * 100
        strain_xy = strain_xy * 100

        return strain_xx, strain_yy, strain_xy

class two: # Compression 2

    # Define a function that calculates the displacement in the x and y directions
    @staticmethod
    def calculateDisplacement(self):

        # Unpack the inputs
        x = self.x
        y = self.y
        epsilon_xx_center = self.epsilon_xx_center
        nu = self.nu
        a = self.a
        x_c = self.x_c
        y_c = self.y_c

        # Calculate the displacement in the x and y directions
        displacement_x = a*(x-x_c)*(y-y_c)**2 + epsilon_xx_center*(x-x_c)
        displacement_y = -nu * ( (a/3)*((y-y_c)**3) + epsilon_xx_center*(y-y_c) )

        return displacement_x, displacement_y

    # Define a function that calculates the strain in the xx, yy and xy direction
    @staticmethod
    def calculateStrain(self):

        # Unpack the inputs
        x = self.x
        y = self.y
        epsilon_xx_center = self.epsilon_xx_center
        nu = self.nu
        a = self.a
        x_c = self.x_c
        y_c = self.y_c

        # Calculate the strain in the xx, yy and xy directions
        strain_xx = a*(y-y_c)**2 + epsilon_xx_center
        strain_yy = -nu*(a*(y - y_c)**2 + epsilon_xx_center)
        strain_xy = a*(x-x_c)*(y - y_c)

        # Convert strain to percent
        strain_xx = strain_xx * 100
        strain_yy = strain_yy * 100
        strain_xy = strain_xy * 100

        return strain_xx, strain_yy, strain_xy