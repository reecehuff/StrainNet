#%% Define a set of classes for the tensile deformations

class quadratic: # Tension 1 (Quadratic)

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

        # Unpack the inputs
        epsilon_xx_center = self.epsilon_xx_center
        nu = self.nu
        x_c = self.x_c
        y_c = self.y_c
        a = self.a
        x = self.x
        y = self.y

        # Calculate the strain in the xx, yy and xy directions
        strain_xx = a*(y-y_c)**2 + epsilon_xx_center
        strain_yy = -nu*(a*(y - y_c)**2 + epsilon_xx_center)
        strain_xy = a*(x-x_c)*(y - y_c)

        # Convert strain to percent
        strain_xx = strain_xx * 100
        strain_yy = strain_yy * 100
        strain_xy = strain_xy * 100

        return strain_xx, strain_yy, strain_xy

class linear: # Tension 2 (Linear)

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

        # Unpack the inputs
        x = self.x
        y = self.y
        nu = self.nu
        y_c = self.y_c
        epsilon_xx_proximal = self.epsilon_xx_proximal
        epsilon_xx_distal = self.epsilon_xx_distal
        w = self.W

        # Calculate the strain in the xx, yy and xy directions
        strain_xx = epsilon_xx_distal + (epsilon_xx_proximal - epsilon_xx_distal) * (x / w)
        strain_yy = - nu * (epsilon_xx_distal + (epsilon_xx_proximal - epsilon_xx_distal) * (x / w))
        strain_xy = - (nu/2) * ((epsilon_xx_proximal - epsilon_xx_distal) * ((y-y_c) / w))

        # Convert strain to percent
        strain_xx = strain_xx * 100
        strain_yy = strain_yy * 100
        strain_xy = strain_xy * 100

        return strain_xx, strain_yy, strain_xy