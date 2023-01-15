##%% Imports
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

#-- Scripts 
from core import utils

#%% Visualization settings
color_bar_range_xx = [0, 10]
color_bar_range_xy = [-5, 5]
color_bar_range_yy = [-5, 0]
color_bar_range_error = [0, 5]

#%% Functions
def visualize_and_save_errors(args, imgs, strains, running_strains, error_image, data_loader):

    # Get the save path for visualizing the images and strains
    save_path = get_visualization_save_path(args, data_loader, init=False)
    # Visualize the images and strains
    visualize_eval(imgs, strains, running_strains, error_image, save_path, args)

    # Get the save path for the error image
    save_path = get_error_image_save_path(args, data_loader, init=False)
    # Save error image
    save_error_image(error_image, save_path, args)

def visualize_experiment(args, imgs, strains, data_loader):

    # Get the save path for visualizing the images and strains
    save_path = get_visualization_save_path(args, data_loader, init=False)
    
    # Visualize the images and strains
    visualize_exp(imgs, strains, save_path, args)

def init_visualization(args, imgs, strains, data_loader):

    # For the very first image, we want to visualize the images and strains
    # The image pair will just be the same image twice
    # The strain will be zero
    # The running strain will be zero
    # The error image will be zero
    imgs[:, 1, :, :] = imgs[:, 0, :, :]
    strains = torch.zeros_like(strains)
    running_strains = torch.zeros_like(strains)
    error_image = torch.zeros_like(strains[:, 0, :, :])

    # Get the save path for visualizing the images and strains
    save_path = get_visualization_save_path(args, data_loader, init=True)

    # Visualize the images and strains
    visualize_eval(imgs, strains, running_strains, error_image, save_path, args)

    # Get the save path for the error image
    save_path = get_error_image_save_path(args, data_loader, init=True)
    # Save error image
    save_error_image(error_image, save_path, args)

def exp_init_visualization(args, imgs, data_loader):

    # For the very first image, we want to visualize the images and strains
    # The image pair will just be the same image twice
    # The strain will be zero
    # The running strain will be zero
    # The error image will be zero
    imgs[:, 1, :, :] = imgs[:, 0, :, :]
    B, C, H, W = imgs.shape
    strains = torch.zeros((B, C+1, H, W))

    # Get the save path for visualizing the images and strains
    save_path = get_visualization_save_path(args, data_loader, init=True)
    
    # Visualize the images and strains
    visualize_exp(imgs, strains, save_path, args)

def get_visualization_save_path(args, data_loader, init=False):

    # Begin by getting the current frame number
    frame_num = utils.get_frame_number(data_loader, init)
    # Define the frame number str (should be 4 digits long)
    frame_num_str = str(frame_num).zfill(4)
    # Create the save path
    fn = frame_num_str + "_visual.png"
    save_path = os.path.join(args.log_dir, 'visualize', fn)

    return save_path

def get_error_image_save_path(args, data_loader, init=False):

    # Begin by getting the current frame number
    frame_num = utils.get_frame_number(data_loader, init) 
    # Define the frame number str (should be 4 digits long)
    frame_num_str = str(frame_num).zfill(4)
    # Create a save path for the error image
    fn = frame_num_str + "_error_image.npy"
    save_path = os.path.join(args.log_dir, 'error_images', fn)

    return save_path 

# Define a function that visualizes the images and strains
def visualize_eval(imgs, strains, running_strains, error_image, savePath, args):
    """
    visualizeData visualizes the images, the predicted strains, and the true strains.
    
    The inputs are:
        imgs - The images (tensor).
        running_strains - The running strains (tensor).
        strains - The true strains (tensor).
        error_image - The error image (tensor).
        args - The arguments (dictionary).
    """

    # Create the directory to save the images and strain fields visualization
    # The visualization path is the save path with the last directory removed
    visualizePath = os.path.dirname(savePath)
    if not os.path.exists(visualizePath):
        os.makedirs(visualizePath)

    # Begin by splitting the images and strain fields
    img1 = imgs[0, 0, :, :].cpu().numpy()
    img2 = imgs[0, 1, :, :].cpu().numpy()
    strain = strains[0, :, :, :].cpu().numpy()
    running_strain = running_strains[0, :, :, :].cpu().numpy()
    true_strain_xx = strain[0, :, :]
    true_strain_xy = strain[1, :, :]
    true_strain_yy = strain[2, :, :]
    pred_strain_xx = running_strain[0, :, :]
    pred_strain_xy = running_strain[1, :, :]
    pred_strain_yy = running_strain[2, :, :]

    # Squeeze all of the tensors
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)
    true_strain_xx = np.squeeze(true_strain_xx)
    true_strain_yy = np.squeeze(true_strain_yy)
    true_strain_xy = np.squeeze(true_strain_xy)
    pred_strain_xx = np.squeeze(pred_strain_xx)
    pred_strain_yy = np.squeeze(pred_strain_yy)
    pred_strain_xy = np.squeeze(pred_strain_xy)
    error_image = np.squeeze(error_image)

    # Create a massive matplotlib figure
    plt.figure(figsize=(20, 20))

    # Add the first image 
    plt.subplot(3, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('First image, I$_1$', fontsize=20)
    plt.axis('off')
    plt.colorbar().set_label(label='Intensity', size=15)

    # Add the second image
    plt.subplot(3, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Second image, I$_2$', fontsize=20)
    plt.axis('off')
    plt.colorbar().set_label(label='Intensity', size=15)

    # Add the error image
    plt.subplot(3, 3, 3)
    title = 'Error image, $\epsilon_{error}$'
    color_bar_label = 'error [%]'
    strainOverImage(img2, error_image, title, color_bar_label, color_bar_range_error)

    # Add the true strain xx
    plt.subplot(3, 3, 4)
    title = 'True strain, $\epsilon_{xx}^{true}$'
    color_bar_label = 'strain [%]'
    strainOverImage(img2, true_strain_xx, title, color_bar_label, color_bar_range_xx)

    # Add the true strain yy
    plt.subplot(3, 3, 5)
    title = 'True strain, $\epsilon_{yy}^{true}$'
    color_bar_label = 'strain [%]'
    strainOverImage(img2, true_strain_yy, title, color_bar_label, color_bar_range_yy)

    # Add the true strain xy
    plt.subplot(3, 3, 6)
    title = 'True strain, $\epsilon_{xy}^{true}$'
    color_bar_label = 'strain [%]'
    strainOverImage(img2, true_strain_xy, title, color_bar_label, color_bar_range_xy)

    # Add the predicted strain xx
    plt.subplot(3, 3, 7)
    title = 'Predicted strain, $\epsilon_{xx}^{pred}$'
    color_bar_label = 'strain [%]'
    strainOverImage(img2, pred_strain_xx, title, color_bar_label, color_bar_range_xx)

    # Add the predicted strain yy
    plt.subplot(3, 3, 8)
    title = 'Predicted strain, $\epsilon_{yy}^{pred}$'
    color_bar_label = 'strain [%]'
    strainOverImage(img2, pred_strain_yy, title, color_bar_label, color_bar_range_yy)

    # Add the predicted strain xy
    plt.subplot(3, 3, 9)
    title = 'Predicted strain, $\epsilon_{xy}^{pred}$'
    color_bar_label = 'strain [%]'
    strainOverImage(img2, pred_strain_xy, title, color_bar_label, color_bar_range_xy)

    # Add a title to the figure
    super_title = 'Strain fields and error image: ' + os.path.basename(savePath)
    plt.suptitle(super_title, fontsize=30)

    # Save the figure
    plt.savefig(savePath, bbox_inches='tight')

    # Close the figure
    plt.close()

# Define a function that visualizes the images, displacement field, and strain fields
def visualize_exp(imgs, strains, savePath, args):
    """
    visualizeData visualizes the images and the predicted strains.
    
    The inputs are:
        imgs - The images (tensor).
        strains - The predicted strains (tensor).
        savePath - The path to save the visualization.
        args - The arguments (dictionary).
    """

    # Create the directory to save the images and strain fields visualization
    # The visualization path is the save path with the last directory removed
    visualizePath = os.path.dirname(savePath)
    if not os.path.exists(visualizePath):
        os.makedirs(visualizePath)

    # Begin by splitting the images and strain fields
    img1 = imgs[0, 0, :, :].cpu().numpy()
    img2 = imgs[0, 1, :, :].cpu().numpy()
    strain = strains[0, :, :, :].cpu().numpy()
    pred_strain_xx = strain[0, :, :]
    pred_strain_yy = strain[1, :, :]
    pred_strain_xy = strain[2, :, :]

    # Squeeze all of the tensors
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)
    pred_strain_xx = np.squeeze(pred_strain_xx)
    pred_strain_yy = np.squeeze(pred_strain_yy)
    pred_strain_xy = np.squeeze(pred_strain_xy)

    # Create a massive matplotlib figure
    plt.figure(figsize=(20, 10))

    # Add the second image with the predicted strain xx overlaid
    plt.subplot(1, 3, 1)
    title = 'Predicted strain ($\epsilon_{xx}^{pred}$)'
    color_bar_label = 'strain [%]'
    strainOverImage(img2, pred_strain_xx, title, color_bar_label, color_bar_range_xx)

    # colorbarplot_xx = plt.imshow(pred_strain_xx,alpha=1, cmap='viridis', vmin=color_bar_range_xx[0], vmax=color_bar_range_xx[1])
    # plt.imshow(img2, alpha=0.7, cmap=plt.cm.gray)
    # plt.title('Predicted strain ($\epsilon_{xx}^{pred}$)', fontsize=20)
    # # Add the colorbar
    # cbar = plt.colorbar(colorbarplot_xx)
    # cbar.set_label(label="strain [%]", size=16)
    # # Adjust the colorbar range
    # cbar.set_ticks(np.ceil(np.linspace(color_bar_range_xx[0], color_bar_range_xx[1], 5)))
    # # Make sure the colorbar ticks do not have decimal points
    # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=16)

    # Add the second image with the predicted strain yy overlaid
    plt.subplot(1, 3, 2)
    title = 'Predicted strain ($\epsilon_{yy}^{pred}$)'
    color_bar_label = 'strain [%]'
    strainOverImage(img2, pred_strain_yy, title, color_bar_label, color_bar_range_yy)
    # colorbarplot_yy = plt.imshow(pred_strain_yy,alpha=1, cmap='viridis', vmin=color_bar_range_yy[0], vmax=color_bar_range_yy[1])
    # plt.imshow(img2, alpha=0.7, cmap=plt.cm.gray)
    # plt.title('Predicted strain ($\epsilon_{yy}^{pred}$)', fontsize=20)
    # # Add the colorbar
    # cbar = plt.colorbar(colorbarplot_yy)
    # cbar.set_label(label="strain [%]", size=16)
    # # Adjust the colorbar range
    # cbar.set_ticks(np.ceil(np.linspace(color_bar_range_yy[0], color_bar_range_yy[1], 5)))
    # # Make sure the colorbar ticks do not have decimal points
    # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=16)

    # Add the second image with the predicted strain xy overlaid
    plt.subplot(1, 3, 3)
    title = 'Predicted strain ($\epsilon_{xy}^{pred}$)'
    color_bar_label = 'strain [%]'
    strainOverImage(img2, pred_strain_xy, title, color_bar_label, color_bar_range_xy)
    # colorbarplot_xy = plt.imshow(pred_strain_xy,alpha=1, cmap='viridis', vmin=color_bar_range_xy[0], vmax=color_bar_range_xy[1])
    # plt.imshow(img2, alpha=0.7, cmap=plt.cm.gray)
    # plt.title('Predicted strain ($\epsilon_{xy}^{pred}$)', fontsize=20)
    # # Add the colorbar
    # cbar = plt.colorbar(colorbarplot_xy)
    # cbar.set_label(label="strain [%]", size=16)
    # # Adjust the colorbar range
    # cbar.set_ticks(np.ceil(np.linspace(color_bar_range_xy[0], color_bar_range_xy[1], 5)))
    # # Make sure the colorbar ticks do not have decimal points
    # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=16)

    # Add a title to the figure
    super_title = 'Strains Predicted by StrainNet: ' + os.path.basename(savePath)
    plt.suptitle(super_title, fontsize=30)

    # Save the figure
    plt.savefig(savePath, bbox_inches='tight')

    # Close the figure
    plt.close()

def strainOverImage(img, strain, title, color_bar_label, color_bar_range):

    colorbarplot = plt.imshow(strain, alpha=1, cmap='viridis', vmin=color_bar_range[0], vmax=color_bar_range[1])
    plt.imshow(img, alpha=0.7, cmap=plt.cm.gray)
    plt.title(title, fontsize=20)
    # Add the colorbar
    cbar = plt.colorbar(colorbarplot)
    cbar.set_label(label=color_bar_label, size=16)
    # Make sure the colorbar ticks do not have decimal points
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=16)
    # Turn off the axis
    plt.axis('off')

# Define a function for creating a Bayesian plot
def create_bayesian_plot(xaxis,line,upper,lower,label,fill_color):
    # Accuracy plots
    plt.plot(xaxis, line, color=fill_color,label=label)
    plt.fill_between(xaxis, lower, upper, alpha=0.2, color=fill_color)
    # plt.errorbar(xaxis, line,  upper - line, color=color1)
    # plt.plot(xaxis, upper,'--', color=color1)
    # plt.plot(xaxis, lower,'--', color=color1)
    pass

# Define a function for visualizing the errors using Bayesian plots
def visualize_errors(performance, args):
    '''
    Visualizes the errors using Bayesian plots
    '''

    # Begin by unraveling the performance dictionary
    median_errors = performance["median_errors"]
    mean_errors = performance["mean_errors"]
    max_errors = performance["max_errors"]
    min_errors = performance["min_errors"]
    percentile_75 = performance["75th"]
    percentile_25 = performance["25th"]
    percentile_95 = performance["95th"]
    percentile_5 = performance["5th"]
    indices = performance["Frame Number"]

    # Create a figure
    plt.figure(figsize=(10, 10))

    # Define the inputs to the Bayesian plot
    xaxis = indices
    line = median_errors
    upper = percentile_75
    lower = percentile_25
    label = "Median error (25th - 75th percentile)"
    fill_color = "blue"
    create_bayesian_plot(xaxis,line,upper,lower,label,fill_color)

    # Make the plot better looking
    plt.xlabel('Frame number', fontsize=20)
    plt.ylabel('Error [%]', fontsize=20)
    plt.title('Error distribution', fontsize=30)
    
    # Increase the font size of the tick labels
    plt.tick_params(axis='both', which='major', labelsize=15)

    # Save the figure
    savePath = os.path.join(args.log_dir, "median_errors.png")
    plt.savefig(savePath, bbox_inches='tight')

# Define a function for visualizing the strains using Bayesian plots
def visualize_strains(predictions, args):
    '''
    Visualizes the strains using Bayesian plots
    '''

    # Begin by unraveling the predictions dictionary
    indices = predictions["Frame Number"]
    bulk_median_strain_xx = predictions["bulk_median_strain_xx"]
    bulk_75th_strain_xx = predictions["bulk_75th_strain_xx"]
    bulk_25th_strain_xx = predictions["bulk_25th_strain_xx"]

    # Create a figure
    plt.figure(figsize=(10, 10))

    # Define the inputs to the Bayesian plot
    xaxis = indices
    line = bulk_median_strain_xx
    upper = bulk_75th_strain_xx
    lower = bulk_25th_strain_xx
    label = "Median strains (25th - 75th percentile)"
    fill_color = "blue"
    create_bayesian_plot(xaxis,line,upper,lower,label,fill_color)

    # Make the plot better looking
    plt.xlabel('Frame number', fontsize=20)
    plt.ylabel('Strain XX [%]', fontsize=20)
    plt.title('Strains over time', fontsize=30)
    
    # Increase the font size of the tick labels
    plt.tick_params(axis='both', which='major', labelsize=15)

    # Save the figure
    savePath = os.path.join(args.log_dir, "median_strains.png")
    plt.savefig(savePath, bbox_inches='tight')

# Define a function that saves error images as npy arrays for later use
def save_error_image(error_image, savePath, args):
    """
    Saves the error images as npy arrays for later use
    """

    # Create the directory to save the images and strain fields visualization
    # The visualization path is the save path with the last directory removed
    visualizePath = os.path.dirname(savePath)
    if not os.path.exists(visualizePath):
        os.makedirs(visualizePath)

    # Save the error image as a numpy array
    np.save(savePath, error_image)
# %%
#