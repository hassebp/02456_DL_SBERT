import torch
import argparse
from models import SBERT
from data_retrieval import webscraping
def check_cuda_availability():
    if torch.cuda.is_available() == True:
        return print('Cuda is available on this device')
    else: 
        return print('Cuda is not setup properly')

def test_sbert():
    SBERT()

def test_webscraping():
    webscraping()

def run_tests():
    parser = argparse.ArgumentParser(description='Run a specific function in the script.')
    
    # Add functions to be tested in the choices list
    parser.add_argument('function', choices=['check_cuda_availability','test_sbert','test_webscraping'], help='Name of the function to run')
    args = parser.parse_args()

    # Call the specified function
    if args.function == 'check_cuda_availability':
        check_cuda_availability()
    elif args.function == 'test_sbert':
        test_sbert()
    elif args.function == 'test_webscraping':
        webscraping()
    else:
        print(f"Error: Unknown function '{args.function}'")

if __name__ == "__main__":
    run_tests()