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
    # Gather the paths to the images and strains
    paths2image1 = sorted(glob.glob(path2input + '/*/images/*im1.png'))
    paths2image2 = sorted(glob.glob(path2input + '/*/images/*im2.png'))
    paths2strainxx = sorted(glob.glob(path2input + '/*/strains/*xx*'))
    paths2strainxy = sorted(glob.glob(path2input + '/*/strains/*xy*'))
    paths2strainyy = sorted(glob.glob(path2input + '/*/strains/*yy*'))

    # Create the images and strains directories if they do not exist
    # Gather all of the deformation types from the image and strain directories
    deformationTypes = []
    for i in range(len(paths2image1)):
        # Isolate the type of deformation from the path
        deformationType = paths2image1[i].split('/')[-3]
        if deformationType not in deformationTypes:
            deformationTypes.append(deformationType)

    # Create the directories
    for deformationType in deformationTypes:
        if not os.path.exists(path2output + '/training/' + deformationType + '/images'):
            os.makedirs(path2output + '/training/' + deformationType + '/images')
            os.makedirs(path2output + '/training/' + deformationType + '/strains')
            os.makedirs(path2output + '/validation/' + deformationType + '/images')
            os.makedirs(path2output + '/validation/' + deformationType + '/strains')

    # Copy the images to the training and validation directories
    for deformationType in deformationTypes:
        # Gather the paths to the images and strains for the current deformation type
        paths2image1 = sorted(glob.glob(path2input + '/' + deformationType + '/images/*im1.png'))
        paths2image2 = sorted(glob.glob(path2input + '/' + deformationType + '/images/*im2.png'))
        paths2strainxx = sorted(glob.glob(path2input + '/' + deformationType + '/strains/*xx*'))
        paths2strainxy = sorted(glob.glob(path2input + '/' + deformationType + '/strains/*xy*'))
        paths2strainyy = sorted(glob.glob(path2input + '/' + deformationType + '/strains/*yy*'))

        # Copy the images to the training and validation directories
        for i in range(len(paths2image1)):
            if i < trainingSetPercentage*len(paths2image1):
                shutil.copy(paths2image1[i], path2output + '/training/' + deformationType + '/images')
                shutil.copy(paths2image2[i], path2output + '/training/' + deformationType + '/images')
            else:
                shutil.copy(paths2image1[i], path2output + '/validation/' + deformationType + '/images')
                shutil.copy(paths2image2[i], path2output + '/validation/' + deformationType + '/images')

        # Copy the strains to the training and validation directories
        for i in range(len(paths2strainxx)):
            if i < trainingSetPercentage*len(paths2strainxx):
                shutil.copy(paths2strainxx[i], path2output + '/training/' + deformationType + '/strains')
                shutil.copy(paths2strainxy[i], path2output + '/training/' + deformationType + '/strains')
                shutil.copy(paths2strainyy[i], path2output + '/training/' + deformationType + '/strains')
            else:
                shutil.copy(paths2strainxx[i], path2output + '/validation/' + deformationType + '/strains')
                shutil.copy(paths2strainxy[i], path2output + '/validation/' + deformationType + '/strains')
                shutil.copy(paths2strainyy[i], path2output + '/validation/' + deformationType + '/strains')

    # Move args.xlsx to the output directory
    shutil.copy(path2input + '/args.xlsx', path2output)