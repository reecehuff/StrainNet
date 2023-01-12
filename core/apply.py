#%% Imports
import torch 
from tqdm import tqdm
import os

#-- Scripts 
from core import utils
from core import visualize
from core import logger
from core.StrainNet import StrainNet

# Define a function for evaluating the model
def apply_StrainNet(args):

    # Define the device
    device = torch.device(args.device)

    # Load StrainNet
    model = StrainNet(args)   
    
    # Load the data
    data_loader = utils.get_experimental_data_loader(args)

    # Load the model to the device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Initialize the strain history
    for imgs in data_loader:
        if args.visualize:
            # Initialize the visualization
            visualize.exp_init_visualization(args, imgs, data_loader)
        # Save the predicted strains if desired
        if args.save_strains:
            # Save the predicted strains
            B, _, H, W = imgs.shape
            zeros_strains = torch.zeros((B, 3, H, W))
            logger.save_strains(zeros_strains, 0, args)
        break

    # Initialize the running strains
    B, _, H, W = imgs.shape
    running_strains = torch.zeros((B, 3, H, W))

    # Initialize the indices (i.e., frame numbers)
    indices = [0]

    # Initialize the predictions dictionary
    predictions = logger.init_predictions_dict()

    # Initialize the prediction classes
    pred_strain_classes = ['Rigid']

    # Iterate over the data
    for imgs in tqdm(data_loader, desc="Evaluating " + args.model_type):

        # Get the current frame number
        frame_num = utils.get_frame_number(data_loader)

        # Append the frame number to the indices
        indices.append(frame_num)

        # Load the images and strains to the device
        imgs = imgs.to(device)

        # Forward pass
        with torch.no_grad():
            outputs, pred_strain = model(imgs)

        # Get the predicted strain classes
        pred_strain_classes.append(utils.get_deformation_type(pred_strain))

        # Update the running strains
        running_strains = running_strains + outputs

        if args.visualize:
            # Visualize the images
            visualize.visualize_experiment(args, imgs, running_strains, data_loader)

        # Log the predictions
        predictions = logger.log_predictions(predictions, running_strains, args)

        # Save the predicted strains if desired
        if args.save_strains:
            # Save the predicted strains
            logger.save_strains(running_strains, frame_num, args)

    # Save the indices to the predictions dictionary
    predictions["Frame Number"] = indices

    # Save the accuracy to the predictions dictionary
    predictions["DeformationClassifier Predictions"] = pred_strain_classes

    # Process the predictions dictionary
    predictions = logger.process_predictions(predictions)

    # If visualize is true, then visualize the errors
    if args.visualize:
        # Visualize the errors
        visualize.visualize_strains(predictions, args)

    # Save the predictions dictionary
    logger.save_predictions(predictions, args, rows=predictions["Frame Number"])