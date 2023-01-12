##%% Imports
import numpy as np
import torch 
import pandas as pd
import os

#-- Scripts
from core import utils

# Define a function for logging the performance from the evaluation
def log_performance(performance, error_image):
    '''
    Function for updating a performance dictiononary with the performance from the evaluation.
    '''

    # Print the median, mean, max and min errors in a visually pleasing way
    # Just three decimal places and include the units and make sure the spacing is correct
    # It should look like a table
    print("Median error: {:.3f} %".format(torch.median(error_image).item()))
    print("Mean error  : {:.3f} %".format(torch.mean(error_image).item()))
    print("Max error   : {:.3f} %".format(torch.max(error_image).item()))
    print("Min error   : {:.3f} %".format(torch.min(error_image).item()))

    # Append the errors to the performance dictionary
    performance["median_errors"].append(torch.median(error_image).item())
    performance["mean_errors"].append(torch.mean(error_image).item())
    performance["max_errors"].append(torch.max(error_image).item())
    performance["min_errors"].append(torch.min(error_image).item())
    performance["75th"].append(torch.quantile(error_image, 0.75).item())
    performance["25th"].append(torch.quantile(error_image, 0.25).item())
    performance["95th"].append(torch.quantile(error_image, 0.95).item())
    performance["5th"].append(torch.quantile(error_image, 0.05).item())

    return performance

# Define a function for logging the predictions
def log_predictions(predictions, strains, args):
    '''
    Function for updating a predictions dictiononary with the predictions from the evaluation.
    '''
    # Crop the strains to get just the bulk strains
    cropped_strains = utils.crop_strains(strains, args)

    # Squeeze off the batch dimension
    strains = strains.squeeze(0)
    cropped_strains = cropped_strains.squeeze(0)

    # Unravel the strain fields to the xx, xy, yy components
    strain_xx = strains[0, :, :]
    strain_xy = strains[1, :, :]
    strain_yy = strains[2, :, :]
    cropped_strain_xx = cropped_strains[0, :, :]
    cropped_strain_xy = cropped_strains[1, :, :]
    cropped_strain_yy = cropped_strains[2, :, :]

    # Append the predictions to the predictions dictionary
    predictions["median_strain_xx"].append(torch.median(strain_xx).item())
    predictions["median_strain_xy"].append(torch.median(strain_xy).item())
    predictions["median_strain_yy"].append(torch.median(strain_yy).item())
    predictions["bulk_median_strain_xx"].append(torch.median(cropped_strain_xx).item())
    predictions["bulk_median_strain_xy"].append(torch.median(cropped_strain_xy).item())
    predictions["bulk_median_strain_yy"].append(torch.median(cropped_strain_yy).item())
    predictions["mean_strain_xx"].append(torch.mean(strain_xx).item())
    predictions["mean_strain_xy"].append(torch.mean(strain_xy).item())
    predictions["mean_strain_yy"].append(torch.mean(strain_yy).item())
    predictions["bulk_mean_strain_xx"].append(torch.mean(cropped_strain_xx).item())
    predictions["bulk_mean_strain_xy"].append(torch.mean(cropped_strain_xy).item())
    predictions["bulk_mean_strain_yy"].append(torch.mean(cropped_strain_yy).item())
    predictions["max_strain_xx"].append(torch.max(strain_xx).item())
    predictions["max_strain_xy"].append(torch.max(strain_xy).item())
    predictions["max_strain_yy"].append(torch.max(strain_yy).item())
    predictions["bulk_max_strain_xx"].append(torch.max(cropped_strain_xx).item())
    predictions["bulk_max_strain_xy"].append(torch.max(cropped_strain_xy).item())
    predictions["bulk_max_strain_yy"].append(torch.max(cropped_strain_yy).item())
    predictions["min_strain_xx"].append(torch.min(strain_xx).item())
    predictions["min_strain_xy"].append(torch.min(strain_xy).item())
    predictions["min_strain_yy"].append(torch.min(strain_yy).item())
    predictions["bulk_min_strain_xx"].append(torch.min(cropped_strain_xx).item())
    predictions["bulk_min_strain_xy"].append(torch.min(cropped_strain_xy).item())
    predictions["bulk_min_strain_yy"].append(torch.min(cropped_strain_yy).item())
    predictions["75th_strain_xx"].append(torch.quantile(strain_xx, 0.75).item())
    predictions["75th_strain_xy"].append(torch.quantile(strain_xy, 0.75).item())
    predictions["75th_strain_yy"].append(torch.quantile(strain_yy, 0.75).item())
    predictions["bulk_75th_strain_xx"].append(torch.quantile(cropped_strain_xx, 0.75).item())
    predictions["bulk_75th_strain_xy"].append(torch.quantile(cropped_strain_xy, 0.75).item())
    predictions["bulk_75th_strain_yy"].append(torch.quantile(cropped_strain_yy, 0.75).item())
    predictions["25th_strain_xx"].append(torch.quantile(strain_xx, 0.25).item())
    predictions["25th_strain_xy"].append(torch.quantile(strain_xy, 0.25).item())
    predictions["25th_strain_yy"].append(torch.quantile(strain_yy, 0.25).item())
    predictions["bulk_25th_strain_xx"].append(torch.quantile(cropped_strain_xx, 0.25).item())
    predictions["bulk_25th_strain_xy"].append(torch.quantile(cropped_strain_xy, 0.25).item())
    predictions["bulk_25th_strain_yy"].append(torch.quantile(cropped_strain_yy, 0.25).item())
    predictions["std_strain_xx"].append(torch.std(strain_xx).item())
    predictions["std_strain_xy"].append(torch.std(strain_xy).item())
    predictions["std_strain_yy"].append(torch.std(strain_yy).item())
    predictions["bulk_std_strain_xx"].append(torch.std(cropped_strain_xx).item())
    predictions["bulk_std_strain_xy"].append(torch.std(cropped_strain_xy).item())
    predictions["bulk_std_strain_yy"].append(torch.std(cropped_strain_yy).item())    
    
    return predictions

# Define a function for initializing the predictions dictionary
def init_predictions_dict():
    '''
    Function for initializing the predictions dictionary.
    '''
    keys = ["median_strain_xx", "median_strain_xy", "median_strain_yy", 
            "bulk_median_strain_xx", "bulk_median_strain_xy", "bulk_median_strain_yy", 
            "mean_strain_xx", "mean_strain_xy", "mean_strain_yy", 
            "bulk_mean_strain_xx", "bulk_mean_strain_xy", "bulk_mean_strain_yy", 
            "max_strain_xx", "max_strain_xy", "max_strain_yy", 
            "bulk_max_strain_xx", "bulk_max_strain_xy", "bulk_max_strain_yy", 
            "min_strain_xx", "min_strain_xy", "min_strain_yy", 
            "bulk_min_strain_xx", "bulk_min_strain_xy", "bulk_min_strain_yy", 
            "75th_strain_xx", "75th_strain_xy", "75th_strain_yy", 
            "bulk_75th_strain_xx", "bulk_75th_strain_xy", "bulk_75th_strain_yy", 
            "25th_strain_xx", "25th_strain_xy", "25th_strain_yy", 
            "bulk_25th_strain_xx", "bulk_25th_strain_xy", "bulk_25th_strain_yy", 
            "std_strain_xx", "std_strain_xy", "std_strain_yy", 
            "bulk_std_strain_xx", "bulk_std_strain_xy", "bulk_std_strain_yy"] 

    predictions = {}

    for key in keys:
        predictions[key] = [0] 
    
    return predictions

def process_performance(performance):
    '''
    Function for processing the performance dictionary.
    It loops through the keeps and converts the lists to numpy arrays.
    '''

    # Loop through the keys in the performance dictionary
    for key in performance.keys():
        # Convert the lists to numpy arrays
        performance[key] = np.array(performance[key])

    return performance

def process_predictions(predictions):
    '''
    Function for processing the predictions dictionary.
    It loops through the keeps and converts the lists to numpy arrays.
    '''

    # Loop through the keys in the predictions dictionary
    for key in predictions.keys():
        # Convert the lists to numpy arrays
        predictions[key] = np.array(predictions[key])

    return predictions

def save_performance(performance, args, rows=None):
    '''
    Function for saving the performance dictionary to a file.
    '''

    print("Saving the performance dictionary to a file...")

    # Save the performance dictionary as an xlsx file
    if rows is None:
        performance_df = pd.DataFrame.from_dict(performance)
    else:
        # Begin by removing the "Frame number" key
        performance.pop("Frame Number")
        performance_df = pd.DataFrame.from_dict(performance)
        performance_df.index = rows
        performance_df.index.name = "Frame Number"
    performance_df.to_excel(args.log_dir + '/performance.xlsx')

def save_predictions(predictions, args, rows=None):
    '''
    Function for saving the predictions dictionary to a file.
    '''

    print("Saving the predictions dictionary to a file...")

    # Save the predictions dictionary as an xlsx file
    if rows is None:
        predictions_df = pd.DataFrame.from_dict(predictions)
    else:
        # Begin by removing the "Frame number" key
        predictions.pop("Frame Number")
        predictions_df = pd.DataFrame.from_dict(predictions)
        predictions_df.index = rows
        predictions_df.index.name = "Frame Number"
    predictions_df.to_excel(args.log_dir + '/predictions.xlsx')

def save_strains(strains, frame_num, args):
    '''
    Function for saving the strains to a file.
    '''

    # Define the save path
    base_dir = os.path.join(args.log_dir, "pred_strains")

    # Create the directory to save strain fields
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Squeeze off the batch dimension
    strains = strains.squeeze(0)

    # Unravel the strain fields to the xx, xy, yy components
    strain_xx = strains[0, :, :]
    strain_xy = strains[1, :, :]
    strain_yy = strains[2, :, :]

    # Define the frame number as a string with 4 digits
    frame_num = str(frame_num).zfill(4)

    # Save the strains as a npy files 
    np.save(os.path.join(base_dir, frame_num + "_strain_xx.npy"), strain_xx.detach().cpu().numpy())
    np.save(os.path.join(base_dir, frame_num + "_strain_xy.npy"), strain_xy.detach().cpu().numpy())
    np.save(os.path.join(base_dir, frame_num + "_strain_yy.npy"), strain_yy.detach().cpu().numpy())

    