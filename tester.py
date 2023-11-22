import torch
import argparse
from scraper import webscraping, generate_urls
from datetime import datetime
from preprocessing import generate_binary_answer_file, split_queries

def check_cuda_availability():
    print('what')
    if torch.cuda.is_available() == True:
        return print('Cuda is available on this device')
    else: 
        return print('Cuda is not setup properly')


def run_tests():
    parser = argparse.ArgumentParser(description='Run a specific function in the script.')
    
    # Add functions to be tested in the choices list
    parser.add_argument('function', choices=['split_queries','generate_bin','generate_urls','generate_triples','test_msmarco','test_training','check_cuda_availability','test_sbert','test_snli',
                                             'test_webscraping', 'test_data_loading', 'test_model', 'train_model'], help='Name of the function to run')
    parser.add_argument("--years", required=False, default=None, nargs='+', type=int, help="Interval of the years wanted for generating urls, example 2018 2022")
    parser.add_argument("--max_pages_pr_year", required=False, default=None, type=int, help="Maximum number of pages to loop through pr year, 1 page is 10 articles")
    parser.add_argument("--max_articles", required=False, default=None, type=int, help="Max number of articles in total")
    parser.add_argument("--filename", default=f'generic_filename_{datetime.now().strftime("%Y%m%dT%H%M%S")}')
    args = parser.parse_args()

    # Call the specified function
    if args.function == 'check_cuda_availability':
        check_cuda_availability()
    elif args.function == 'generate_urls':
        if args.years == None or args.max_pages_pr_year == None or args.max_articles == None:
            raise ValueError("You must specifiy years, max number of pages pr year and total number of articles.\n See tester.py for more help")
        years = args.years
        max_pg_yr = args.max_pages_pr_year
        max_articles = args.max_articles
        filename = args.filename
        generate_urls(years, filename, max_pg_yr, max_articles)
    elif args.function == 'test_webscraping':
        filename = args.filename
        if args.max_articles:
            webscraping(filename, args.max_articles)
        else:
            webscraping(filename)
    elif args.function == 'generate_bin':
        generate_binary_answer_file('C:/Users/hasse/Skrivebord/02456_DL_SBERT/data_articlev2/keywords.csv')
    elif args.function == 'split_queries':
        split_queries('C:/Users/hasse/Skrivebord/02456_DL_SBERT/data_articlev2/queries.csv')
        
    else:
        print(f"Error: Unknown function '{args.function}'")
    

if __name__ == "__main__":
    run_tests()