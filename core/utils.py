#%% Imports
import glob
import os
import numpy as np
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms

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

    # Send the model to the device
    model.to(args.device)
    # Set the model to training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    return model

# Define a function for loading the model
def load_model(args, model_type, train=False):

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
    if args.resume:
        print('Loading model from: {}'.format(args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=torch.device(args.device)))
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

# Define a function for getting the paths for the images or strains
def get_paths(path):

    # Get all of the paths 
    paths = sorted(glob.glob(os.path.join(path, '*/*/*'), recursive=True))

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

    # Create a dictionary of the paths
    paths = {'image1': image1paths, 
             'image2': image2paths, 
             'strain_xx': strain_xx_paths, 
             'strain_yy': strain_yy_paths, 
             'strain_xy': strain_xy_paths}

    return paths

# Define a function for getting the paths for the images or strains for evalation
def get_eval_paths(path, sampling_rate=1):

    # Get all of the paths 
    paths = sorted(glob.glob(os.path.join(path, '*/*/*'), recursive=True))

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
            data_set = Dataset_4_Classification(paths)
        else:
            data_set = Dataset_4_Classification(paths)
    else:
        if train:
            data_set = Dataset_4_Regression(paths)
        else:
            data_set = Dataset_4_Regression(paths)

    # Create the data loader
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=train)

    return data_loader

# Define a function for getting the data loader for evaluation
def get_eval_data_loader(args):

    # Gather the paths for the images and strains
    paths = get_eval_paths(args.val_data_dir, sampling_rate=args.sampling_rate)

    # Loop through the paths and print them 
    for key in paths.keys():
        print(key)
        for path in paths[key]:
            print(path)

    
    # Create DataSet object
    data_set = Dataset_4_Regression(paths)

    # Create the data loader
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return data_loader

# Define a function for getting the data loader for testing
def get_experimental_data_loader(args):
    
    # Gather the paths for the images and strains
    paths = get_paths(args.test_data_dir)

    # Create DataSet object
    data_set = Dataset_Experimental(paths, args.valid_transform)

    # Create the data loader
    data_loader = DataLoader(data_set, batch_size=args.exp_batch_size, shuffle=False)

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
                                        transforms.ToTensor()
                                        ])
    elif transform_type == 'valid':
        transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((args.input_height, args.input_width)),
                                        transforms.ToTensor()
                                        ])
    elif transform_type == 'test':
        transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((args.input_height, args.input_width)),
                                        transforms.ToTensor()
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
    
    # Convert the class prediction to the deformation type
    class_pred = softmax_to_class(class_pred)

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

# Define a function for gathering the data directories for evaluation
def get_eval_data_dirs(args):

    # If dealing with sequential data
    if args.sequential:
        
        # Assert that the data directories exist
        assert os.path.exists(args.val_data_dir), 'Validation data directory does not exist'

    # If dealing with a standard validation dataset 
    else:

        # Assert that the data directories exist
        assert os.path.exists(args.val_data_dir), 'Validation data directory does not exist'

    return args


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