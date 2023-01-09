#%% Imports
import torch 
from tqdm import tqdm

#-- Scripts 
from core import utils
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
        performance = evaluate_sequential(args, model, data_loader, device)
    # Otherwise, use the standard evaluation function
    else:
        performance = evaluate(args, model, data_loader, device)

# Define a function for evaluating the model 
def evaluate(args, model, data_loader, device):
    
    # Load the model to the device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Define a performance dictionary
    performance = {"median_errors": [], "mean_errors": [], "max_errors": [], "min_errors": []}

    # Define the number of correct predictions
    num_correct = 0

    # Iterate over the data
    for imgs, strains in tqdm(data_loader, desc="Evaluating " + args.model_type):

        # Load the images and strains to the device
        imgs, strains = imgs.to(device), strains.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(imgs)

        # Define the true strain class and the predicted strain class
        true_strain = utils.get_deformation_class(strains)
        pred_strain = utils.get_deformation_class(outputs)
        
        # Calculate the accuracy
        if true_strain == pred_strain:
            num_correct += 1

        # Calculate the loss
        error_image = createStrainErrorImage(strains, outputs)

        # Append the errors to the performance dictionary
        performance["median_errors"].append(torch.median(error_image))
        performance["mean_errors"].append(torch.mean(error_image))
        performance["max_errors"].append(torch.max(error_image))
        performance["min_errors"].append(torch.min(error_image))
    
    # Calculate the accuracy
    accuracy = num_correct / len(data_loader) * 100 

    # Print the accuracy
    print("Accuracy: ", accuracy)

    # Print the performance
    print("Median Error: ", torch.mean(torch.stack(performance["median_errors"])))
    print("Mean Error: ", torch.mean(torch.stack(performance["mean_errors"])))
    print("Max Error: ", torch.mean(torch.stack(performance["max_errors"])))
    print("Min Error: ", torch.mean(torch.stack(performance["min_errors"])))

    # Save the accuracy to the performance dictionary
    performance["accuracy"] = accuracy

    return performance

# Define a function for evaluating the model on sequential images
def evaluate_sequential(args, model, data_loader, device):
    pass 

# Define a function for calcuting the error image 
def createStrainErrorImage(true_strain, pred_strain):
    # Make sure that the batch size is 1
    assert true_strain.shape[0] == 1
    assert pred_strain.shape[0] == 1

    # Duplicate the last channel of the true strain and pred strain
    true_strain = torch.cat((true_strain, true_strain[:, -1:]), dim=1)
    pred_strain = torch.cat((pred_strain, pred_strain[:, -1:]), dim=1)

    # Calculate the error image is the RMSE between the true strain and pred strain along the channel dimension
    error_image = torch.mean((true_strain - pred_strain)**2, dim=1)
    error_image = torch.sqrt(error_image)

    # Return the error image
    return error_image
