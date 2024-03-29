#%% Imports
import glob
import os
import numpy as np
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
import re 

#-- Scripts 
from core.archs import DeformationClassifier
from core.archs import UNet
from core.dataset import Dataset_4_Regression, Dataset_4_Classification, Dataset_Experimental

# Define a function for initializing the model
def initialize_model(args, model_type, train=False):

    # Initialize the model depending on the model type
    if model_type == 'DeformationClassifier':
        model = DeformationClassifier(args)
        model_name = args.DeformationClassifier_name
    elif model_type == 'TensionNet':
        model = UNet(args)
        model_name = args.TensionNet_name
    elif model_type == 'CompressionNet':
        model = UNet(args)
        model_name = args.CompressionNet_name
    elif model_type == 'RigidNet':
        model = UNet(args)
        model_name = args.RigidNet_name
    
    # Initialize the model
    print('Initializing model in: {}'.format(os.path.join(args.model_dir, model_name + '.pt')))

    # Send the model to the device
    model.to(args.device)

    # Set the model to training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    return model

# Define a function for loading the model
def load_model(args, model_type, train=False, resume=None):

    # Initialize the model depending on the model type
    if model_type == 'DeformationClassifier':
        model = DeformationClassifier(args)
        model_name = args.DeformationClassifier_name
    elif model_type == 'TensionNet':
        model = UNet(args)
        model_name = args.TensionNet_name
    elif model_type == 'CompressionNet':
        model = UNet(args)
        model_name = args.CompressionNet_name
    elif model_type == 'RigidNet':
        model = UNet(args)
        model_name = args.RigidNet_name

    # Load the model
    if resume:
        print('Resuming training on: {}'.format(os.path.join(resume, model_name + '.pt')))
        model.load_state_dict(torch.load(os.path.join(resume, model_name + '.pt'), map_location=torch.device(args.device)))
    else:
        print('Loading model from: {}'.format(os.path.join(args.model_dir, model_name + '.pt')))
        model.load_state_dict(torch.load(os.path.join(args.model_dir, model_name + '.pt'), map_location=torch.device(args.device)))

    # Send the model to the device
    model.to(args.device)

     # Set the model to training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    return model

# Define a function for getting the optimizer
def get_optimizer(args, model):
    # Define the optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    return optimizer

# define a function for getting the loss function
def get_loss_function(args, model_type):
    # Define the loss function
    if model_type == 'DeformationClassifier':
        loss_function = torch.nn.CrossEntropyLoss()
    elif model_type == 'TensionNet':
        loss_function = torch.nn.MSELoss()
    elif model_type == 'CompressionNet':
        loss_function = torch.nn.MSELoss()
    elif model_type == 'RigidNet':
        loss_function = torch.nn.MSELoss()

    return loss_function

# Define a function for getting the paths
def get_paths(path, sampling_rate=1):

    # Get all of the paths 
    # The depth of the paths to search depends on where the data is stored
    # Therefore, we will increase the depth until we find the data
    max_depth = 10
    for i in range(max_depth):
        depth = "/*" * i + "/*.*"
        depth = depth[1:]
        paths = sorted(glob.glob(os.path.join(path, depth), recursive=True))
        if len(paths) != 0:
            break
        if i == max_depth - 1:
            raise Exception('No data is present in the path you specified')


    # Separate the paths for the images and strains
    image1paths = []
    image2paths = []
    strain_xx_paths = []
    strain_yy_paths = []
    strain_xy_paths = []
    for path in paths:
        if 'im1' in path:
            image1paths.append(path)
        elif 'im2' in path:
            image2paths.append(path)
        elif 'xx' in path:
            strain_xx_paths.append(path)
        elif 'yy' in path:
            strain_yy_paths.append(path)
        elif 'xy' in path:
            strain_xy_paths.append(path)

    # Sort the paths
    image1paths = sorted(image1paths)
    image2paths = sorted(image2paths)
    strain_xx_paths = sorted(strain_xx_paths)
    strain_yy_paths = sorted(strain_yy_paths)
    strain_xy_paths = sorted(strain_xy_paths)

    # Sample the paths
    image1paths = image1paths[::sampling_rate]
    image2paths = image2paths[::sampling_rate]
    strain_xx_paths = strain_xx_paths[::sampling_rate]
    strain_yy_paths = strain_yy_paths[::sampling_rate]
    strain_xy_paths = strain_xy_paths[::sampling_rate]

    # Create a dictionary of the paths
    paths = {'image1': image1paths, 
             'image2': image2paths, 
             'strain_xx': strain_xx_paths, 
             'strain_yy': strain_yy_paths, 
             'strain_xy': strain_xy_paths}

    return paths

# Define a function for getting the paths
def get_sequential_paths(path, sampling_rate=1, custom_sampling=False):

    # Get all of the paths 
    # The depth of the paths to search depends on where the data is stored
    # Therefore, we will increase the depth until we find the data
    max_depth = 10
    for i in range(max_depth):
        depth = "/*" * i + "/*.*"
        depth = depth[1:]
        paths = sorted(glob.glob(os.path.join(path, depth), recursive=True))
        if len(paths) != 0:
            break
        if i == max_depth - 1:
            raise Exception('No data is present in the path you specified')

    # Separate the paths for the images and strains
    imagepaths = []
    strain_xx_paths = []
    strain_yy_paths = []
    strain_xy_paths = []
    for path in paths:
        if 'im' in path:
            imagepaths.append(path)
        elif 'xx' in path:
            strain_xx_paths.append(path)
        elif 'yy' in path:
            strain_yy_paths.append(path)
        elif 'xy' in path:
            strain_xy_paths.append(path)

    # Sort the paths
    imagepaths = sorted(imagepaths)
    strain_xx_paths = sorted(strain_xx_paths)
    strain_yy_paths = sorted(strain_yy_paths)
    strain_xy_paths = sorted(strain_xy_paths)

    # Sample the paths
    if custom_sampling:
        imagepaths = [imagepaths[i] for i in custom_sampling]
        strain_xx_paths = [strain_xx_paths[i] for i in custom_sampling]
        strain_yy_paths = [strain_yy_paths[i] for i in custom_sampling]
        strain_xy_paths = [strain_xy_paths[i] for i in custom_sampling]
    else:
        imagepaths = imagepaths[::sampling_rate]
        strain_xx_paths = strain_xx_paths[::sampling_rate]
        strain_yy_paths = strain_yy_paths[::sampling_rate]
        strain_xy_paths = strain_xy_paths[::sampling_rate]

    # Image 1 paths will be all of the image paths except the last one
    image1paths = imagepaths[:-1]
    # Image 2 paths will be all of the image paths except the first one
    image2paths = imagepaths[1:]
    # Strain paths will be all of the strain paths except the first one
    strain_xx_paths = strain_xx_paths[1:]
    strain_yy_paths = strain_yy_paths[1:]
    strain_xy_paths = strain_xy_paths[1:]

    # Print the number of paths
    print('Number of image1 paths: ', len(image1paths))
    print('Number of image2 paths: ', len(image2paths))
    print('Number of strain_xx paths: ', len(strain_xx_paths))
    print('Number of strain_yy paths: ', len(strain_yy_paths))
    print('Number of strain_xy paths: ', len(strain_xy_paths))

    # Create a dictionary of the paths
    paths = {'image1': image1paths, 
             'image2': image2paths, 
             'strain_xx': strain_xx_paths, 
             'strain_yy': strain_yy_paths, 
             'strain_xy': strain_xy_paths}

    return paths

# Define a function for getting the paths for experimental image paths
def get_exp_image_paths(path, sampling_rate=1, custom_sampling=False):

    # Get all of the paths 
    # The depth of the paths to search depends on where the data is stored
    # Therefore, we will increase the depth until we find the data
    max_depth = 10
    for i in range(max_depth):
        depth = "/*" * i + "/*.*"
        depth = depth[1:]
        paths = sorted(glob.glob(os.path.join(path, depth), recursive=True))
        if len(paths) != 0:
            break
        if i == max_depth - 1:
            raise Exception('No data is present in the path you specified')

    # Separate the paths for the images and strains
    imagepaths = []
    for path in paths:
        # if 'im' in path:
        imagepaths.append(path)

    # Sort the paths
    imagepaths = sorted(imagepaths)

    # Sample the paths
    if custom_sampling:
        imagepaths = [imagepaths[i] for i in custom_sampling]
    else:
        imagepaths = imagepaths[::sampling_rate]

    # Image 1 paths will be all of the image paths except the last one
    image1paths = imagepaths[:-1]
    # Image 2 paths will be all of the image paths except the first one
    image2paths = imagepaths[1:]

    # Print the number of paths
    print('Number of image1 paths: ', len(image1paths))
    print('Number of image2 paths: ', len(image2paths))
    num_blanks = len(os.path.basename(image1paths[0]))
    message1 = "path1"; message2 = "path2"
    message1 += " " * (num_blanks - len(message1))
    message2 += " " * (num_blanks - len(message2))
    print("  ", message1, "  ||  ", message2)
    for path1, path2 in zip(image1paths[:5],image2paths[:5]):
        path1 = os.path.basename(path1)
        path2 = os.path.basename(path2)
        print("  ", path1, "  ||  ", path2)

    # Create a dictionary of the paths
    paths = {'image1': image1paths, 
             'image2': image2paths}

    return paths

# Define a function for getting the data loader for training or validation
def get_data_loader(args, model_type, train=True):

    # Gather the paths for the images and strains
    if train:
        paths = get_paths(args.train_data_dir)
    else:
        paths = get_paths(args.val_data_dir)

    # Create DataSet objects
    if model_type == 'DeformationClassifier':
        if train:
            data_set = Dataset_4_Classification(paths, transform=args.train_transform)
        else:
            data_set = Dataset_4_Classification(paths, transform=args.valid_transform)
    else:
        if train:
            data_set = Dataset_4_Regression(paths, transform=args.train_transform)
        else:
            data_set = Dataset_4_Regression(paths, transform=args.valid_transform)

    # Create the data loader
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=train)

    return data_loader

# Define a function for getting the data loader for evaluation
def get_eval_data_loader(args):

    # Gather the paths for the images and strains
    if args.sequential:
        paths = get_sequential_paths(args.val_data_dir, sampling_rate=args.sampling_rate, custom_sampling=args.custom_sampling)
    else:
        paths = get_paths(args.val_data_dir, sampling_rate=args.sampling_rate)

    # Print the first five paths
    for key in paths.keys():
        print(key)
        some_paths = paths[key][:5]
        for path in some_paths:
            print(path)

    # Create DataSet object
    data_set = Dataset_4_Regression(paths, args.valid_transform)

    # Create the data loader
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return data_loader

# Define a function for getting the data loader for testing
def get_experimental_data_loader(args):
    
    # Gather the paths for the images
    paths = get_exp_image_paths(args.exp_data_dir, args.sampling_rate)

    # Create DataSet object
    data_set = Dataset_Experimental(paths, args.test_transform)

    # Create the data loader
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return data_loader

# Define a function for saving the model
def save_model(model, args, model_type):

    # Define the model name
    if model_type == 'DeformationClassifier':
        model_name = args.DeformationClassifier_name
    elif model_type == 'TensionNet':
        model_name = args.TensionNet_name
    elif model_type == 'CompressionNet':
        model_name = args.CompressionNet_name
    elif model_type == 'RigidNet':
        model_name = args.RigidNet_name

    # Save the model both to the save directory and to the logging directory
    torch.save(model.state_dict(), os.path.join(args.model_dir, model_name + '.pt'))
    torch.save(model.state_dict(), os.path.join(args.log_dir, model_name + '.pt'))

# Define a function for obtaining the transforms
def get_transform(args, transform_type):

    # Define the transforms
    if transform_type == 'train':
        transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((args.input_height, args.input_width)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.squeeze(0))
                                        ])
    elif transform_type == 'valid':
        transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((args.input_height, args.input_width)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.squeeze(0))
                                        ])
    elif transform_type == 'test':
        transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((args.input_height, args.input_width)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.squeeze(0))
                                        ])
    else:
        raise ValueError('Invalid transform type')

    return transform

# Define a function for getting the deformation class based sum of the strains in the xx direction
def get_deformation_class(strains):

    # If the input is a tensor, convert it to a numpy array
    if torch.is_tensor(strains):
        strains = strains.numpy()

    # Isolate the strain in the xx direction
    if len(strains.shape) == 3:
        strain_xx = strains[0, :, :]
        sum_strain_xx = np.sum(strain_xx, axis=(0, 1))
    elif len(strains.shape) == 4:
        strain_xx = strains[:, 0, :, :]
        sum_strain_xx = np.sum(strain_xx, axis=(1, 2))

    # Get the deformation type
    if sum_strain_xx < 0:
        deformation_type = 2 # Compression
    elif sum_strain_xx > 0:
        deformation_type = 1 # Tension
    else:
        deformation_type = 0 # Rigid

    return deformation_type

# Define a function for getting the deformation type 
def get_deformation_type(class_pred):

    # Get the deformation type
    if class_pred == 1:
        deformation_type = 'Tension'
    elif class_pred == 2:
        deformation_type = 'Compression'
    else:
        deformation_type = 'Rigid'

    return deformation_type

# Define a function for stacking the images and strains
def stack(list_of_images_or_strains):

    # Stack the images or strains
    stacked = np.stack(list_of_images_or_strains, axis=2)
    
    # Permute the stacked images or strains
    stacked = np.transpose(stacked, (2, 0, 1))

    return stacked

# Define a function for converting the softmax output to a deformation class
def softmax_to_class(softmax_output):

    # If the softmax output is a numpy array
    if type(softmax_output) == np.ndarray:
        # Convert the softmax output to a deformation class
        class_pred = np.argmax(softmax_output, axis=1)
    # If the softmax output is a tensor
    elif type(softmax_output) == torch.Tensor:
        # Convert the softmax output to a deformation class
        class_pred = torch.argmax(softmax_output, dim=1)
    else:
        raise ValueError('Invalid type for softmax output')

    return class_pred

# Define a function for adjusting the data directories depending on the model type
def get_data_dirs(args):

    # Adjust the data directories depending on the model type
    if args.model_type == 'DeformationClassifier':
        pass 
    elif args.model_type == 'TensionNet':
        args.train_data_dir = os.path.join(args.train_data_dir, 'tension')
        args.val_data_dir = os.path.join(args.val_data_dir, 'tension')
    elif args.model_type == 'CompressionNet':
        args.train_data_dir = os.path.join(args.train_data_dir, 'compression')
        args.val_data_dir = os.path.join(args.val_data_dir, 'compression')
    elif args.model_type == 'RigidNet':
        args.train_data_dir = os.path.join(args.train_data_dir, 'rigid')
        args.val_data_dir = os.path.join(args.val_data_dir, 'rigid')
    else:
        raise ValueError('Invalid model type')

    # Assert that the data directories exist
    assert os.path.exists(args.train_data_dir), 'Training data directory does not exist'
    assert os.path.exists(args.val_data_dir), 'Validation data directory does not exist'

    return args

# Define a function for getting the current frame number 
def get_frame_number(data_loader, init=False):
    
    # Get the current frame number
    im1_path, im2_path = data_loader.dataset.get_im_paths()
    # Isolate the file name
    im1_fn, im2_fn = os.path.basename(im1_path), os.path.basename(im2_path)
    # Isolate the frame number from the file name
    # Note the frame could be anywhere in the file name
    im1_frame_num, im2_frame_num = int(re.findall(r'\d+', im1_fn)[0]), int(re.findall(r'\d+', im2_fn)[0])

    if init:
        return im1_frame_num
    else:
        return im2_frame_num

# Define a function setting the random seeds
def set_random_seeds(seed):
    # Set the random seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define a function for cropping the strains
def crop_strains(strains, args):

        # Unpack the cropping coordinates
        lower_left_x, lower_left_y, upper_right_x, upper_right_y = args.crop_box

        # Crop the strains
        if len(strains.shape) == 3:
            strains = strains[:, lower_left_y:upper_right_y, lower_left_x:upper_right_x]
        elif len(strains.shape) == 4:
            strains = strains[:, :, lower_left_y:upper_right_y, lower_left_x:upper_right_x]
    
        return strains