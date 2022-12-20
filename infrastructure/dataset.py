import cv2 
from scipy import io
import numpy as np 
import torch.utils.data as data

# Class for loading the dataset for the synthetic test cases

class Dataset_TestCase(data.Dataset):
    def __init__(self, image1_ids, image2_ids, strain_XX_mat_ids, strain_XY_mat_ids, strain_YY_mat_ids, transform=None):
        """
        Args
        ----------
        image1_ids : list
            List of image 1 ids. 
        image2_ids : list
            List of image 2 ids. 
        strain_XX_mat_ids : list
            List of strain_XX mat ids. There should be one for each image pair.
        strain_XY_mat_ids : list
            List of strain_XY mat ids. There should be one for each image pair.
        strain_YY_mat_ids : list
            List of strain_YY mat ids. There should be one for each image pair.

        Returns
        -------
        Torch dataset with image pairs as input data and a stack of the strain 
        images as the corresponding target.

        """
        self.image1_ids = image1_ids 
        self.image2_ids = image2_ids 
        self.strain_XX_mat_ids = strain_XX_mat_ids
        self.strain_XY_mat_ids = strain_XY_mat_ids
        self.strain_YY_mat_ids = strain_YY_mat_ids
        self.transform  = transform
        
    def __len__(self):
        return len(self.strain_XX_mat_ids)
    
    def __getitem__(self, index):
                
        #--------------INPUT IMAGE PAIRS-------------#
        # Get image ids
        img1_id = self.image1_ids[index]
        img2_id = self.image2_ids[index]
        # Read in images
        img1 = cv2.imread(img1_id, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_id, cv2.IMREAD_GRAYSCALE)
        # Normalize images
        img1 = 2*(img1.astype('float32') / 255) - 1.0
        img2 = 2*(img2.astype('float32') / 255) - 1.0
        # Determine h and w in order to make sure the images are divisible by 32
        h,w   = img1.shape
        h_rem = h % 32
        w_rem = w % 32
        img1  = img1[h_rem:,w_rem:]
        img2  = img2[h_rem:,w_rem:]
        # Stack images
        images = np.dstack((img1,img2))
        # Permute channels to N x H x W
        images = np.transpose(images,(2,0,1))
        
        #------------TARGET STRAIN IMAGES------------#
        # Get strain ids
        strain_XX_id = self.strain_XX_mat_ids[index]
        strain_XY_id = self.strain_XY_mat_ids[index]
        strain_YY_id = self.strain_YY_mat_ids[index] 
        # Read in strain images
        strain_XX = io.loadmat(strain_XX_id)
        strain_XY = io.loadmat(strain_XY_id)
        strain_YY = io.loadmat(strain_YY_id)
        # Get keys 
        strain_XX_key = list(strain_XX.keys())[-1]
        strain_XY_key = list(strain_XY.keys())[-1]
        strain_YY_key = list(strain_YY.keys())[-1]
        # Use key to unlock the strain images
        strain_XX = strain_XX[strain_XX_key]
        strain_XY = strain_XY[strain_XY_key]
        strain_YY = strain_YY[strain_YY_key]
        # Normalize strain images
        strain_XX = strain_XX.astype('float32') 
        strain_XY = strain_XY.astype('float32') 
        strain_YY = strain_YY.astype('float32') 
        # Make sure the images are divisible by 32
        strain_XX  = strain_XX[h_rem:,w_rem:]
        strain_XY  = strain_XY[h_rem:,w_rem:]
        strain_YY  = strain_YY[h_rem:,w_rem:]
        # Stack strain images
        target_strain = np.dstack((strain_XX,strain_XY,strain_YY))
        # Permute channels to N x H x W
        target_strain = np.transpose(target_strain,(2,0,1))
        
        return images, target_strain, img2_id

#%% A classifier dataset that will be used to train the DeformationClassifier


class Dataset_Classifier(data.Dataset):
    def __init__(self, image1_ids, image2_ids, strain_XX_mat_ids, transform=None):
        """
        Args
        ----------
        image1_ids : list
            List of image 1 ids. 
        image2_ids : list
            List of image 2 ids. 
        strain_XX_mat_ids : list
            List of strain_XX mat ids. There should be one for each image pair.
        strain_XY_mat_ids : list
            List of strain_XY mat ids. There should be one for each image pair.
        strain_YY_mat_ids : list
            List of strain_YY mat ids. There should be one for each image pair.

        Returns
        -------
        Torch dataset with image pairs as input data and a stack of the strain 
        images as the corresponding target.

        """
        self.image1_ids = image1_ids 
        self.image2_ids = image2_ids 
        self.strain_XX_mat_ids = strain_XX_mat_ids
        self.transform  = transform
        
    def __len__(self):
        return len(self.strain_XX_mat_ids)
    
    def __getitem__(self, index):
                
        #--------------INPUT IMAGE PAIRS-------------#
        # Get image ids
        img1_id = self.image1_ids[index]
        img2_id = self.image2_ids[index]
        # Read in images
        img1 = cv2.imread(img1_id, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_id, cv2.IMREAD_GRAYSCALE)
        # Normalize images
        img1 = 2*(img1.astype('float32') / 255) - 1.0
        img2 = 2*(img2.astype('float32') / 255) - 1.0
        # Determine h and w in order to make sure the images are divisible by 32
        h,w   = img1.shape
        h_rem = h % 32
        w_rem = w % 32
        img1  = img1[h_rem:,w_rem:]
        img2  = img2[h_rem:,w_rem:]
        # Stack images
        images = np.dstack((img1,img2))
        # Permute channels to N x H x W
        images = np.transpose(images,(2,0,1))
        
        #------------TARGET STRAIN IMAGES------------#
        # Get strain ids
        strain_XX_id = self.strain_XX_mat_ids[index]
        # Read in strain images
        strain_XX = io.loadmat(strain_XX_id)
        # Get keys 
        strain_XX_key = list(strain_XX.keys())[-1]
        # Use key to unlock the strain images
        strain_XX = strain_XX[strain_XX_key]
        # Determine class of strain
        target_class = int(np.sign(np.sum(strain_XX)))
        if target_class == -1:
            target_class = 2

        # Class 0: Rigid
        # Class 1: Tension
        # Class 2: Compression
        
        return images, target_class, img2_id

#%% A class for the experimental dataset

class Dataset_Experimental(data.Dataset):
    def __init__(self, image1_ids, image2_ids, transform=None):
        """
        Args
        ----------
        image1_ids : list
            List of image 1 ids. 
        image2_ids : list
            List of image 2 ids. 
        classes : list
            List of classes associated with each image pair. 

        Returns
        -------
        Torch dataset with image pairs as input data and their class (e.g., 
        tension, compression, or rigid) as the corresponding target.

        """
        self.image1_ids = image1_ids 
        self.image2_ids = image2_ids 
        self.transform  = transform
        
    def __len__(self):
        return len(self.image1_ids)
    
    def __getitem__(self, index):
                
        #--------------INPUT IMAGE PAIRS-------------#
        # Get image ids
        img1_id = self.image1_ids[index]
        img2_id = self.image2_ids[index]
        # Read in images
        img1 = cv2.imread(img1_id, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_id, cv2.IMREAD_GRAYSCALE)
        # Normalize images
        img1 = 2*(img1.astype('float32') / 255) - 1.0
        img2 = 2*(img2.astype('float32') / 255) - 1.0
        # Determine h and w in order to make sure the images are divisible by 32
        h,w   = img1.shape
        h_rem = h % 32
        w_rem = w % 32
        img1  = img1[h_rem:,w_rem:]
        img2  = img2[h_rem:,w_rem:]
        # Stack images
        images = np.dstack((img1,img2))
        # Permute channels to N x H x W
        images = np.transpose(images,(2,0,1))
        
        return images, img2_id