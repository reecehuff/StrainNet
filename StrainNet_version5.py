#%% Imports
import os
import numpy as np
from scipy import io
import glob
from PIL import Image

from dataset import Dataset
import archs

#---ML imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torchvision import transforms

#---Plotting imports 
import matplotlib.pyplot as plt
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2021/bin/universal-darwin'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Make a directory for where we will save our figures
if not os.path.exists('figures'):
    os.makedirs('figures')
    
# Make a directory for where we will save our models
if not os.path.exists('models'):
    os.makedirs('models')
    
#%% Clear and close
os.system("clear")
plt.close("all")

#%% ----------------------- Beginning of the script ----------------------- %%#
###############################################################################
#%% Inputs 
batch_size = 10
learning_rate = 1e-3
wdecay = 0.01
epsilon = 1e-08

input_beta = 0.0001

# Percentage of images used for training 
training_percentage = 0.8333333333333333333
epochs = 100

model_name = "1200_Compression_SR30_Noise.pth"
mainPath   = "/home/rdhuff/Documents/GitHub/DevelopTrainingSet/CROP_TrainingSets/1200_Compression_SR30/"
imagePath  = "dataNoise/"

predictions = "on"

#%% Read in images and strains
def getTrainAndValidDatasets(mainPath, training_percentage, imagePath):
    
    path2tifs       = mainPath + imagePath
    path2strains_XX = mainPath + "Strains_XX/"
    path2strains_XY = mainPath + "Strains_XY/"
    path2strains_YY = mainPath + "Strains_YY/"
    
    image1_file_list       = sorted(glob.glob(path2tifs + "*img1.tif"))
    if imagePath == "dataNoise/":
        image2_file_list       = sorted(glob.glob(path2tifs + "*img2_noise.tif"))
    elif imagePath == "data/":
        image2_file_list       = sorted(glob.glob(path2tifs + "*img2.tif"))
    
    strain_XX_file_list    = sorted(glob.glob(path2strains_XX + "*.mat"))
    strain_XY_file_list    = sorted(glob.glob(path2strains_XY + "*.mat"))
    strain_YY_file_list    = sorted(glob.glob(path2strains_YY + "*.mat"))

    split_number = int( len(image1_file_list) * training_percentage )

    image1_file_list_train      = image1_file_list[:split_number]
    image1_file_list_valid      = image1_file_list[split_number:]
    image2_file_list_train      = image2_file_list[:split_number]
    image2_file_list_valid      = image2_file_list[split_number:]
    strain_XX_file_list_train   = strain_XX_file_list[:split_number]
    strain_XX_file_list_valid   = strain_XX_file_list[split_number:]
    strain_XY_file_list_train   = strain_XY_file_list[:split_number]
    strain_XY_file_list_valid   = strain_XY_file_list[split_number:]
    strain_YY_file_list_train   = strain_YY_file_list[:split_number]
    strain_YY_file_list_valid   = strain_YY_file_list[split_number:]

    train_Dataset = Dataset(image1_file_list_train, image2_file_list_train, strain_XX_file_list_train, strain_XY_file_list_train, strain_YY_file_list_train)
    valid_Dataset = Dataset(image1_file_list_valid, image2_file_list_valid, strain_XX_file_list_valid, strain_XY_file_list_valid, strain_YY_file_list_valid)
    
    return (train_Dataset, valid_Dataset)

train_transform = transforms.Compose([
                                        # transforms.RandomRotate90(),
                                        # transforms.Flip(),
                                        transforms.Normalize(0.5, 0.5),
                                    ])
valid_transform = transforms.Compose([
                                        transforms.Normalize(0.5, 0.5),
                                    ])

#-----------------Read in images with zero strain applied---------------------#  
# mainPath                        = "/home/rdhuff/Documents/GitHub/DevelopTrainingSet/DS_TrainingSets/10000_Zero_Noiseless/"

# mainPath                        = "/home/rdhuff/Documents/GitHub/DevelopTrainingSet/CROP_TrainingSets/1200_ZeroStrain/"
# train_Dataset, valid_Dataset    = getTrainAndValidDatasets(mainPath, training_percentage)
# train_DataLoader    = DataLoader(train_Dataset, batch_size=batch_size) 
# valid_DataLoader    = DataLoader(valid_Dataset, batch_size=batch_size) 

#-------------------Read in images with tension applied-----------------------#  
(train_Dataset, valid_Dataset)  = getTrainAndValidDatasets(mainPath, training_percentage, imagePath)
train_DataLoader    = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True) 
valid_DataLoader    = DataLoader(valid_Dataset, batch_size=batch_size, shuffle=False) 

#-----------------Read in images with compression applied---------------------#  
# mainPath                        = "/home/rdhuff/Documents/GitHub/DevelopTrainingSet/DS_TrainingSets/1200_Compression/"

# mainPath                        = "/home/rdhuff/Documents/GitHub/DevelopTrainingSet/CROP_TrainingSets/1200_Compression_10mean_5SD/"
# train_Dataset, valid_Dataset    = getTrainAndValidDatasets(mainPath, training_percentage)
# train_DataLoader    = DataLoader(train_Dataset, batch_size=batch_size) 
# valid_DataLoader    = DataLoader(valid_Dataset, batch_size=batch_size) 

#%% Verify the shapes of the DataLoader
#----------------------verify the shapes of your data-------------------------#

for X, y, img_ids in train_DataLoader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print(img_ids)
    break

for X, y, img_ids in valid_DataLoader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print(img_ids)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

#%% Define model
model = archs.UNet(3)
model = model.to(device)
print(model)

# Defining loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wdecay, eps=epsilon)

def train(dataloader, model, loss_fn, optimizer):
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for X, y, _ in dataloader:
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if device == "cuda":
            torch.cuda.empty_cache()
    train_loss /= num_batches
    print(f"Training: \n Avg loss: {train_loss:>8f} \n")
    return train_loss
    
def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    for X, y, _ in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        if device == "cuda":
            torch.cuda.empty_cache()
    test_loss /= num_batches
    print(f"Validation: \n Avg loss: {test_loss:>8f} \n")
    return test_loss
    
best_valid_loss = 10000000
train_losses = []
valid_losses = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss_t = train(train_DataLoader, model, loss_fn, optimizer)
    valid_loss_t = test(valid_DataLoader, model, loss_fn)
    if valid_loss_t < best_valid_loss:
        torch.save(model.state_dict(), "models/%s" % model_name)
        best_valid_loss = valid_loss_t
        print("======> New best model is saved <======== \n")
    train_losses.append(train_loss_t)
    valid_losses.append(valid_loss_t)
    
print("Done!")

#%% Plot the losses and save the training 
# Close all figures
plt.close("all")

# Create figure
plt.figure(figsize=(6,6))

# Plot losses vs epochs
plt.plot(np.arange(epochs), np.array(train_losses), label=r"Training loss")
plt.plot(np.arange(epochs), np.array(valid_losses), label=r"Validation loss")

# xlabel and ylabel
plt.xlabel(r'epochs',fontsize=16)
plt.ylabel(r'Loss (mean squared error)',fontsize=16)
plt.title(r"\textbf{Training and validation loss}" , fontsize=20)

# Add legend 
plt.legend(fontsize=16)

# Increase tick font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save plot 
plt.savefig("figures/%s_losses.png" % model_name[:-4], dpi=600)

#%% Evaluate one of the predictions

if predictions == "on":
    for X, y, _ in valid_DataLoader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    model = archs.UNet(3)
    model.load_state_dict(torch.load("models/%s" % model_name))
    model = model.to(device)
    model.eval()
    
    for i in range(batch_size):
        pred_strain = model(X.to(device))[i][0]
        pred_strain = pred_strain.cpu().detach().numpy()
        true_strain = y[i][0].detach().numpy()
        
        # Create figure
        plt.figure(figsize=(14,6))
        
        # Create subplot 2
        plt.subplot(1,2,1)
        f1 = plt.imshow(true_strain,cmap="gnuplot")
        
        # xlabel and ylabel
        plt.xlabel(r'$x_1$',fontsize=16)
        plt.ylabel(r'$x_2$',fontsize=16)
        plt.title(r"True Strain Field" , fontsize=20)
        
        # Create colorbar
        cb = plt.colorbar(f1)
        cb.set_label(label=r"$\epsilon_{xx}$", fontsize=16)
        plt.clim(np.min(true_strain), np.max(true_strain))
        
        # Increase tick font size
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        cb.ax.tick_params(labelsize=12)
        
        # Set aspect ratio
        plt.gca().set_aspect('equal')
        
        # Create subplot 2
        plt.subplot(1,2,2)
        f1 = plt.imshow(pred_strain,cmap="gnuplot")
        
        # xlabel and ylabel
        plt.xlabel(r'$x_1$',fontsize=16)
        plt.ylabel(r'$x_2$',fontsize=16)
        plt.title(r"Predicted Strain Field" , fontsize=20)
        
        # Create colorbar
        cb = plt.colorbar(f1)
        cb.set_label(label=r"$\epsilon_{xx}$", fontsize=16)
        plt.clim(np.min(true_strain), np.max(true_strain))
        
        # Increase tick font size
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        cb.ax.tick_params(labelsize=12)
        
        # Set aspect ratio
        plt.gca().set_aspect('equal')
    
    model.train()