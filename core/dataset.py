#%% Imports
import cv2 
import numpy as np 
import torch.utils.data as data

#-- Scripts 
from core import utils

#%% DataSet Classes 

# Class for loading the dataset for the synthetic test cases
class Dataset_4_Regression(data.Dataset):
    def __init__(self, paths, transform=None):
        """
        Args
        ----------
        paths : Dictionary
            Dictionary containing the paths to the image pairs and the strain
        transform : callable, optional
            Optional transform to be applied on a sample.

        Returns
        -------
        Torch dataset with image pairs as input data and a stack of the strain 
        images as the corresponding target.

        """
        self.paths = paths
        self.transform  = transform

    def __len__(self):
        # Assert that the number of image pairs is the same as the number of strain images
        assert len(self.paths['image1']) == len(self.paths['strain_xx']), 'The number of image pairs is not the same as the number of strain images'

        return len(self.paths['image1'])
    
    def __getitem__(self, index):

        # Save the index
        self.index = index

        # Load the image pair
        image1 = cv2.imread(self.paths['image1'][index], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(self.paths['image2'][index], cv2.IMREAD_GRAYSCALE)

        # Normalize images
        image1 = 2*(image1.astype('float32') / 255) - 1.0
        image2 = 2*(image2.astype('float32') / 255) - 1.0

        # Load the strain images
        strain_XX = np.load(self.paths['strain_xx'][index])
        strain_XY = np.load(self.paths['strain_xy'][index])
        strain_YY = np.load(self.paths['strain_yy'][index])

        # Convert strains to float32
        strain_XX = strain_XX.astype('float32')
        strain_XY = strain_XY.astype('float32')
        strain_YY = strain_YY.astype('float32')
        
        # Apply the transform
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            strain_XX = self.transform(strain_XX)
            strain_XY = self.transform(strain_XY)
            strain_YY = self.transform(strain_YY)

        # Stack the images
        images = utils.stack([image1, image2])

        # Stack the strain images
        strains = utils.stack([strain_XX, strain_XY, strain_YY])

        return images, strains

    def get_im_paths(self):
        return self.paths['image1'][self.index], self.paths['image2'][self.index]

#%% A classifier dataset that will be used to train the DeformationClassifier

class Dataset_4_Classification(data.Dataset):
    def __init__(self, paths, transform=None):
        """
        Args
        ----------
        paths : Dictionary
            Dictionary containing the paths to the image pairs and the strain
        transform : callable, optional
            Optional transform to be applied on a sample.

        Returns
        -------
        Torch dataset with image pairs as input data and a stack of the strain 
        images as the corresponding target.

        """
        self.paths = paths
        self.transform  = transform

    def __len__(self):
        # Assert that the number of image pairs is the same as the number of strain images
        assert len(self.paths['image1']) == len(self.paths['strain_xx']), 'The number of image pairs is not the same as the number of strain images'

        return len(self.paths['image1'])

    def __getitem__(self, index):

        # Save the index
        self.index = index
            
        # Load the image pair
        image1 = cv2.imread(self.paths['image1'][index], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(self.paths['image2'][index], cv2.IMREAD_GRAYSCALE)

        # Normalize images
        image1 = 2*(image1.astype('float32') / 255) - 1.0
        image2 = 2*(image2.astype('float32') / 255) - 1.0

        # Load the strain images
        strain_XX = np.load(self.paths['strain_xx'][index])
        strain_XY = np.load(self.paths['strain_xy'][index])
        strain_YY = np.load(self.paths['strain_yy'][index])

        # Convert strains to float32
        strain_XX = strain_XX.astype('float32')
        strain_XY = strain_XY.astype('float32')
        strain_YY = strain_YY.astype('float32')

        # Apply the transform
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            strain_XX = self.transform(strain_XX)
            strain_XY = self.transform(strain_XY)
            strain_YY = self.transform(strain_YY)

        # Stack the images
        images = utils.stack([image1, image2])

        # Stack the strain images
        strains = utils.stack([strain_XX, strain_XY, strain_YY])

        # Get the deformation class
        deformation_class = utils.get_deformation_class(strains)

        return images, deformation_class

    def get_im_paths(self):
        return self.paths['image1'][self.index], self.paths['image2'][self.index]

#%% A class for the experimental dataset
class Dataset_Experimental(data.Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Args
        ----------
        image_paths : list
            List of image paths. 
        transform : callable, optional
            Optional transform to be applied on a sample.

        Returns
        -------
        Torch dataset with image pairs as input data and a stack of the strain 
        images as the corresponding target.

        """
        self.image_paths = image_paths
        self.transform  = transform

    def __len__(self):
        # Note that substract one because the last image is the reference image
        return len(self.image_paths)-1
        
    def __getitem__(self, index):
        # Load the image pair
        image1 = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(self.image_paths[index+1], cv2.IMREAD_GRAYSCALE)

        # Stack the images
        images = utils.stack([image1, image2])

        # Apply the transform
        if self.transform:
            images = self.transform(images)

        return images