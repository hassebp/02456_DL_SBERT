import torch
import argparse

def check_cuda_availability():
    if torch.cuda.is_available() == True:
        return print('Cuda is available on this device')
    else: 
        return print('Cuda is not setup properly')

def run_tests():
    parser = argparse.ArgumentParser(description='Run a specific function in the script.')
    
    # Add an argument for the function name
    parser.add_argument('function', choices=['check_cuda_availability'], help='Name of the function to run')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the specified function
    if args.function == 'check_cuda_availability':
        check_cuda_availability()
    else:
        print(f"Error: Unknown function '{args.function}'")

if __name__ == "__main__":
    run_tests()