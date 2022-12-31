#%% Imports
import shutil 
import os
import glob

# Define a function that copies the images and strains to a new directory
# while splitting the data into training and validation sets
def copyDataAndSplitIntoTrainingAndValidation(path2input, path2output, trainingSetPercentage):
    """
    copyDataAndSplitIntoTrainingAndValidation copies the images and strains to a new directory while splitting the data into training and validation sets.
    
    The inputs are:
        path2input - The path to the directory that contains the images and strains (string).
        path2output - The path to the directory that will contain the images and strains (string).

    """
    # Create the output directory if it does not exist
    if not os.path.exists(path2output):
        os.makedirs(path2output)
    
    # Create the training and validation directories if they do not exist
    if not os.path.exists(path2output + '/training'):
        os.makedirs(path2output + '/training')
    if not os.path.exists(path2output + '/validation'):
        os.makedirs(path2output + '/validation')
    
    # Create the images and strains directories if they do not exist
    if not os.path.exists(path2output + '/training/images'):
        os.makedirs(path2output + '/training/images')
    if not os.path.exists(path2output + '/training/strains'):
        os.makedirs(path2output + '/training/strains')
    if not os.path.exists(path2output + '/validation/images'):
        os.makedirs(path2output + '/validation/images')
    if not os.path.exists(path2output + '/validation/strains'):
        os.makedirs(path2output + '/validation/strains')
    
    # Gather the paths to the images and strains
    paths2image1 = glob.glob(path2input + '/images/*im1.png')
    paths2image2 = glob.glob(path2input + '/images/*im2.png')
    paths2strainxx = glob.glob(path2input + '/strains/*xx*')
    paths2strainxy = glob.glob(path2input + '/strains/*xy*')
    paths2strainyy = glob.glob(path2input + '/strains/*yy*')

    # Sort the paths to the images and strains
    paths2image1.sort()
    paths2image2.sort()
    paths2strainxx.sort()
    paths2strainxy.sort()
    paths2strainyy.sort()

    print(paths2strainxx)
    
    # Copy the images to the training and validation directories
    for i in range(len(paths2image1)):
        if i < trainingSetPercentage*len(paths2image1):
            shutil.copy(paths2image1[i], path2output + '/training/images')
            shutil.copy(paths2image2[i], path2output + '/training/images')
        else:
            shutil.copy(paths2image1[i], path2output + '/validation/images')
            shutil.copy(paths2image2[i], path2output + '/validation/images')

    # Copy the strains to the training and validation directories
    for i in range(len(paths2strainxx)):
        if i < trainingSetPercentage*len(paths2strainxx):
            shutil.copy(paths2strainxx[i], path2output + '/training/strains')
            shutil.copy(paths2strainxy[i], path2output + '/training/strains')
            shutil.copy(paths2strainyy[i], path2output + '/training/strains')
        else:
            shutil.copy(paths2strainxx[i], path2output + '/validation/strains')
            shutil.copy(paths2strainxy[i], path2output + '/validation/strains')
            shutil.copy(paths2strainyy[i], path2output + '/validation/strains')