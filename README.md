# 02456 DL SBERT


## Setup
### Activating the environment
Go to your command prompt and navigate into the folder "02456_DL_SBERT" and write the command: venv\Scripts\activate 

### Installing dependecies
When you have activated the environment, use the following command to install all the dependencies: pip install -r requirements.txt

### Testing CUDA is available on your device
From cmd run the following: python tester.py check_cuda_availability

If it says "Cuda is available on this device" you are good to go, otherwise you might have to install the CUDA toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local 

or try https://pytorch.org/get-started/locally/