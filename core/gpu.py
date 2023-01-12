#%% Imports
import torch


def gpu_status(device, gpu_ids):
    """
    Check if the GPU is available and if the user wants to use the GPU.
    If the GPU is available and the user wants to use the GPU, then the device is set to the GPU.
    Otherwise, the device is set to the CPU.

    Parameters
    ----------
    device : torch.device
        The device that the model will be trained on.
    gpu_ids : list
        The list of GPU IDs that the user wants to use.

    Returns
    -------
    device : torch.device
        The device that the model will be trained on.
    """

    # Check if the GPU is available
    if torch.cuda.is_available():
        # Set the device to the GPU to the first GPU in the list of GPU IDs
        device = torch.device('cuda:' + str(gpu_ids[0]))
    else:
        # Set the device to the CPU
        device = torch.device('cpu')

    # Let's print out some information about the device being used 
    if torch.cuda.is_available():
        print_status(device)
    else:
        print("\n" * 2)
        print("=========================================")
        print("Device: " + str(device))
        print("=========================================")
        print("\n" * 2)

    return device

def print_status(device):
    """
    Print the memory usage of the GPU.

    Parameters
    ----------
    device : torch.device
        The device that the model will be trained on.
    """

    print("\n" * 2)
    print("================================================")
    print("Device                   : " + str(device))
    print("Number of GPUs           : " + str(torch.cuda.device_count()))
    print("Current GPU              : " + str(torch.cuda.current_device()))
    print("GPU Name                 : " + str(torch.cuda.get_device_name(0)))
    print("GPU Memory Allocated     : " + str(round(torch.cuda.memory_allocated(0)/1024**3,2)) + " GB")
    print("GPU Total Memory         : " + str(round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)) + " GB")
    print("================================================")
    print("\n" * 2)