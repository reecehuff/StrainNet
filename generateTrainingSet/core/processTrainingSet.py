#%% Imports
import shutil 
import os
import glob
import cv2
import numpy as np

# Define a function that copies the images and strains to a new directory
# while splitting the data into training and validation sets
def copyDataAndSplitIntoTrainingAndValidation(path2input, path2output, trainingSetPercentage, shuffle=False):
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

        # If shuffle is true, then we need to shuffle the paths in the same way
        if shuffle:
            c = list(zip(paths2image1, paths2image2, paths2strainxx, paths2strainxy, paths2strainyy))
            np.random.shuffle(c)
            paths2image1, paths2image2, paths2strainxx, paths2strainxy, paths2strainyy = zip(*c)

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

def augment_training_set(path2output, args):
    print('The training set is being augmented.')
    print(' ' * 5)
    # Copy the training set for augmentation 
    path2output_augment = path2output + '_augment_DC'
    if os.path.exists(path2output_augment):
        shutil.rmtree(path2output_augment)
    shutil.copytree(path2output, path2output_augment)
    temp_dir = os.path.join(path2output_augment, 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    for deformation in ['rigid', 'compression', 'tension']:
        os.makedirs(os.path.join(temp_dir, deformation, 'images'))
        os.makedirs(os.path.join(temp_dir, deformation, 'strains'))

    # Move args.xlsx to the output directory
    shutil.copy(path2output + '/args.xlsx', temp_dir)
    
    # Need to define these 
    splits = {'10mvc_trial1': [  1,200,800,1100], '10mvc_trial2': [  1,200,800,1100], '10mvc_trial3': [  1,200,800,1100], # '10mvc_trial4': [  1,200,800,1100], '10mvc_trial5': [  1,200,800,1100],
              '30mvc_trial1': [  1,200,800,1100], '30mvc_trial2': [  1,200,800,1100], '30mvc_trial3': [  1,200,800,1100], # '30mvc_trial4': [  1,200,800,1100], '30mvc_trial5': [  1,200,800,1100],
              '50mvc_trial1': [  1,200,800,1100], '50mvc_trial2': [  1,200,800,1100], '50mvc_trial3': [  1,200,800,1100], } # '50mvc_trial4': [  1,200,800,1100], '50mvc_trial5': [  1,200,800,1100]}
    COUNT = 1
    for key in splits.keys():

        paths2images = sorted(glob.glob(args.path2experimentalImages + key + '*.png'))
        indices = (np.arange(len(paths2images)) + 1).astype('int')
        divide_by = 10
        sampling_rate = int(args.aug_sample_rate / divide_by)
        paths2images = paths2images[::divide_by]
        indices = indices[::divide_by]

        indices = indices[:-sampling_rate]
        paths2images_1 = paths2images[:-sampling_rate]
        paths2images_2 = paths2images[sampling_rate:]

        for i, img_path_1, img_path_2 in zip(indices, paths2images_1, paths2images_2):
            img1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
            img1 = crop(img1, args)
            img2 = crop(img2, args)
            strain_xx = np.zeros((img1.shape))
            strain_xy = np.zeros((img1.shape))
            strain_yy = np.zeros((img1.shape))
            # Define 
            COUNT_str = str(COUNT).zfill(3)

            if splits[key][0] <= i and i <= splits[key][1]: # compression 
                imagePath = os.path.join(temp_dir, 'compression', 'images')
                im1_out_path = os.path.join(imagePath, 'a' + COUNT_str + '_im1.png')
                im2_out_path = os.path.join(imagePath, 'a' + COUNT_str + '_im2.png')
                cv2.imwrite(im1_out_path, img1)
                cv2.imwrite(im2_out_path, img2)
                # Save the strain fields as a mat file
                strainPath = os.path.join(temp_dir, 'compression', 'strains')
                path2strains_XX = os.path.join(strainPath, 'a' + COUNT_str + '_strain_xx.npy')
                path2strains_YY = os.path.join(strainPath, 'a' + COUNT_str + '_strain_yy.npy')
                path2strains_XY = os.path.join(strainPath, 'a' + COUNT_str + '_strain_xy.npy')
                # Save each component of the strain field separately as a numpy array
                np.save(path2strains_XX, strain_xx)
                np.save(path2strains_YY, strain_yy)
                np.save(path2strains_XY, strain_xy)

            elif splits[key][2] <= i and i <= splits[key][3]: # tension 
                imagePath = os.path.join(temp_dir, 'tension', 'images')
                im1_out_path = os.path.join(imagePath, 'a' + COUNT_str + '_im1.png')
                im2_out_path = os.path.join(imagePath, 'a' + COUNT_str + '_im2.png')
                cv2.imwrite(im1_out_path, img1)
                cv2.imwrite(im2_out_path, img2)
                # Save the strain fields as a mat file
                strainPath = os.path.join(temp_dir, 'tension', 'strains')
                path2strains_XX = os.path.join(strainPath, 'a' + COUNT_str + '_strain_xx.npy')
                path2strains_YY = os.path.join(strainPath, 'a' + COUNT_str + '_strain_yy.npy')
                path2strains_XY = os.path.join(strainPath, 'a' + COUNT_str + '_strain_xy.npy')
                # Save each component of the strain field separately as a numpy array
                np.save(path2strains_XX, strain_xx)
                np.save(path2strains_YY, strain_yy)
                np.save(path2strains_XY, strain_xy)

            else: # rigid 
                imagePath = os.path.join(temp_dir, 'rigid', 'images')
                im1_out_path = os.path.join(imagePath, 'a' + COUNT_str + '_im1.png')
                im2_out_path = os.path.join(imagePath, 'a' + COUNT_str + '_im2.png')
                cv2.imwrite(im1_out_path, img1)
                cv2.imwrite(im2_out_path, img2)
                # Save the strain fields as a mat file
                strainPath = os.path.join(temp_dir, 'rigid', 'strains')
                path2strains_XX = os.path.join(strainPath, 'a' + COUNT_str + '_strain_xx.npy')
                path2strains_YY = os.path.join(strainPath, 'a' + COUNT_str + '_strain_yy.npy')
                path2strains_XY = os.path.join(strainPath, 'a' + COUNT_str + '_strain_xy.npy')
                # Save each component of the strain field separately as a numpy array
                np.save(path2strains_XX, strain_xx)
                np.save(path2strains_YY, strain_yy)
                np.save(path2strains_XY, strain_xy)

            COUNT += 1

    copyDataAndSplitIntoTrainingAndValidation(temp_dir, path2output_augment, args.training_percentage, shuffle=True)
    shutil.rmtree(temp_dir)

    print('The training set has been augmented.')
    print(' ' * 5)

def crop(array, args):
    output_height = args.output_height
    output_width = args.output_width
    upper_left_corner_x = args.upper_left_corner_x
    upper_left_corner_y = args.upper_left_corner_y

    # Crop the array
    array = array[upper_left_corner_y:upper_left_corner_y+output_height, upper_left_corner_x:upper_left_corner_x+output_width]
    return array