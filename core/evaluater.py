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
def eval_model(args, sequential=False):

    # Define the device
    device = torch.device(args.device)

    # Load StrainNet
    model = StrainNet(args)   
    
    # Load the data
    data_loader = utils.get_eval_data_loader(args)

    # If the images are sequential, then use the sequential evaluation function
    if sequential:
        evaluate_sequential(args, model, data_loader, device)
    # Otherwise, use the standard evaluation function
    else:
        evaluate(args, model, data_loader, device)

# Define a function for evaluating the model 
def evaluate(args, model, data_loader, device):
    
    # Load the model to the device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Define a performance dictionary
    performance = {"median_errors": [], "mean_errors": [], "max_errors": [], "min_errors": [], 
                    "75th": [], "25th": [], "95th": [], "5th": []}

    # Define the number of correct predictions
    num_correct = 0

    # Iterate over the data
    for imgs, strains in tqdm(data_loader, desc="Evaluating " + args.model_type):

        # Load the images and strains to the device
        imgs, strains = imgs.to(device), strains.to(device)

        # Forward pass
        with torch.no_grad():
            outputs, pred_strain = model(imgs)

        # Define the true strain class and the predicted strain class
        true_strain = utils.get_deformation_class(strains)
        
        # Calculate the accuracy
        if true_strain == pred_strain:
            num_correct += 1

        # Calculate the loss
        error_image = createStrainErrorImage(strains, outputs)

        # Log the performance
        performance = logger.log_performance(performance, error_image)
    
    # Calculate the accuracy
    accuracy = num_correct / len(data_loader) * 100 

    # Save the accuracy to the performance dictionary
    performance["accuracy"] = accuracy

    # Save the performance dictionary
    logger.save_performance(performance, args)

# Define a function for evaluating the model on sequential images
def evaluate_sequential(args, model, data_loader, device):

    # Load the model to the device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Define a performance dictionary
    performance = {"median_errors": [0], "mean_errors": [0], "max_errors": [0], "min_errors": [0], 
                    "75th": [0], "25th": [0], "95th": [0], "5th": [0]}

    # Define the number of correct predictions
    num_correct = 0
    correct_ones = [True]
    accuracy_so_far = [100]

    # Initialize the strain history
    for imgs, strains in data_loader:
        if args.visualize:
            # Initialize the visualization
            visualize.init_visualization(args, imgs, strains, data_loader)
        # Save the predicted strains if desired
        if args.save_strains:
            # Save the predicted strains
            zeros_strains = torch.zeros_like(strains)
            logger.save_strains(zeros_strains, 0, args)
        break
    running_strains = torch.zeros_like(strains)

    # Define the previous strain (useful for determining the magnitude of the strain)
    prev_strain = torch.zeros_like(strains)

    # Initialize the indices (i.e., frame numbers)
    indices = [0]

    # Iterate over the data
    for imgs, strains in tqdm(data_loader, desc="Evaluating " + args.model_type):

        # Get the current frame number
        frame_num = utils.get_frame_number(data_loader)
        # Append the frame number to the indices
        indices.append(frame_num)

        # Load the images and strains to the device
        imgs, strains = imgs.to(device), strains.to(device)

        # Convert the strains to percentages
        strains = strains * 100

        # Forward pass
        with torch.no_grad():
            outputs, pred_strain = model(imgs)

        # Update the running strains
        running_strains = running_strains + outputs

        # Define the true strain class and the predicted strain class
        true_strain = utils.get_deformation_class(strains - prev_strain)

        # Convert the DeformationClassifier predictions to classes
        pred_strain_type = utils.get_deformation_type(pred_strain)
        true_strain_type = utils.get_deformation_type(true_strain)
        
        # Calculate the accuracy
        if true_strain == pred_strain:
            num_correct += 1
            correct_ones.append(True)
        else:
            correct_ones.append(False)
        accuracy_so_far.append((num_correct+1) / len(correct_ones) * 100)

        # Calculate the loss
        error_image = createStrainErrorImage(strains, running_strains)

        if args.visualize:
            # Visualize the images
            visualize.visualize_and_save_errors(args, imgs, strains, running_strains, error_image, data_loader)

        # Log the performance
        performance = logger.log_performance(performance, error_image)

        # Save the predicted strains if desired
        if args.save_strains:
            # Save the predicted strains
            logger.save_strains(running_strains, frame_num, args)

        # Update the previous strain
        prev_strain = strains
    
    # Calculate the accuracy
    accuracy = num_correct / len(data_loader) * 100 

    # Print the accuracy
    print("Accuracy: ", accuracy)

    # Save the accuracy to the performance dictionary
    performance["DeformationClassifier Correct"] = correct_ones
    performance["Accuracy So Far"] = accuracy_so_far
    performance["Overall Accuracy"] = accuracy

    # Save the indices to the performance dictionary
    performance["Frame Number"] = indices

    # Process the performance dictionary
    performance = logger.process_performance(performance)

    # If visualize is true, then visualize the errors
    if args.visualize:
        # Visualize the errors
        visualize.visualize_errors(performance, args)

    # Save the performance dictionary
    logger.save_performance(performance, args, rows=performance["Frame Number"])

# Define a function for calcuting the error image 
def createStrainErrorImage(true_strain, pred_strain):
    # Make sure that the batch size is 1
    assert true_strain.shape[0] == 1
    assert pred_strain.shape[0] == 1

    # Remove the batch dimension
    true_strain = true_strain.squeeze(0)
    pred_strain = pred_strain.squeeze(0)

    # Duplicate the middle channel of the true strain and pred strain (the shear strain)
    true_strain = torch.cat((true_strain, true_strain[1:2, :, :]), dim=0)
    pred_strain = torch.cat((pred_strain, pred_strain[1:2, :, :]), dim=0)

    # Calculate the error image is the RMSE between the true strain and pred strain along the channel dimension
    error_image = torch.mean((true_strain - pred_strain)**2, dim=0)
    error_image = torch.sqrt(error_image)

    # Return the error image
    return error_image
