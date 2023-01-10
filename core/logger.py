##%% Imports
import numpy as np
import torch 
import pandas as pd
import os

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

    