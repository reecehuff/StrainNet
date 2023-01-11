#%% Imports
import os
import torch 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#-- Scripts 
from core import utils

# Define a function for training the model
def train_model(args, model_type):

    # Define the device
    device = torch.device(args.device)
    
    # Initialize the model
    if args.resume:
        model = utils.load_model(args, model_type, train=True)
    else:
        model = utils.initialize_model(args, model_type, train=True)

    # Define the optimizer
    optimizer = utils.get_optimizer(args, model)

    # Define the loss function
    loss_function = utils.get_loss_function(args, model_type)

    # Define the data loaders for training and validation
    train_data_loader = utils.get_data_loader(args, model_type)
    val_data_loader = utils.get_data_loader(args, model_type, train=False)

    # Define a tensorboard writer
    writer = SummaryWriter(os.path.join(args.log_dir, model_type))

    # Train the model
    for epoch in tqdm(range(args.epochs), desc='Training ' + model_type):
        # Train for one epoch
        train_one_epoch(model, train_data_loader, optimizer, loss_function, device, writer, epoch)
        # Evaluate on the validation dataset
        evaluate(model, val_data_loader, loss_function, device, writer, epoch)
        # Save the model
        utils.save_model(model, args, model_type)

# Define a function for training for one epoch
def train_one_epoch(model, data_loader, optimizer, loss_function, device, writer, epoch):

    # Set the model to training mode
    model.train()

    # Iterate over the data
    for imgs, strains in data_loader:
        # Send the data to the device
        imgs = imgs.to(device)
        strains = strains.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(imgs)
        # Calculate the loss
        loss = loss_function(outputs, strains)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()

        # Log the loss
        writer.add_scalar('losses/training', loss.item(), epoch)


# Define a function for evaluating the model during training
def evaluate(model, data_loader, loss_function, device, writer, epoch):
    
    # Set the model to evaluation mode
    model.eval()

    # Initialize the number of correct predictions
    num_correct = 0
    # Iterate over the data
    for imgs, strains in data_loader:
        # Send the data to the device
        imgs = imgs.to(device)
        strains = strains.to(device)

        # Forward pass
        outputs = model(imgs)

        # Calculate the loss
        loss = loss_function(outputs, strains)

        # Log the loss
        writer.add_scalar('losses/validation', loss, epoch)
        
        # If the model is a DeformationClassifier, calculate the accuracy
        if model.model_type == 'DeformationClassifier':
            # Calculate the accuracy
            predictions = utils.softmax_to_class(outputs)
            num_correct += (predictions == strains).sum().item()

    # If the model is a DeformationClassifier, calculate the accuracy
    if model.model_type == 'DeformationClassifier':
        # Calculate the accuracy
        accuracy = num_correct / len(data_loader.dataset)
        # Log the accuracy
        writer.add_scalar('accuracies/validation', accuracy, epoch)