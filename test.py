import torch


def check_cuda_availability():
    if torch.cuda.is_available() == True:
        return 'Cuda is available on this device'
    else: 
        return 'Cuda is not setup properly'
    
print(check_cuda_availability())