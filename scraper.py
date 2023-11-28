# scraper.py
import os
import csv
import time
import requests
from bs4 import BeautifulSoup
from preprocessing import preprocess_text
import random
from tqdm import tqdm
import re
import numpy as np


def get_article_links(years, max_pages_per_year, max_articles):
    """
    Retrieve links to articles within a specified range of years.
    """
    start_year, end_year = years
    links = set()
    base_url = "https://findit.dtu.dk/en/catalog?type=article_journal_article"
    for year in range(start_year, end_year + 1):
        for page in range(1, max_pages_per_year + 1):
            url = f"{base_url}&year={year}&page={page}"
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                article_links = [link['href'] for link in soup.find_all('a', {'class': 'result__title'})]
                links.update(article_links)
                if len(links) >= max_articles:
                    return list(links)[:max_articles]
            time.sleep(0.5)  # Delay to prevent IP ban
    return list(links)

def extract_keywords(soup):
    """
    Extract and preprocess keywords from a BeautifulSoup object.
    """
    keywords = []
    for identifier in ['Keywords', 'Other keywords']:
        keyword_tag = soup.find('strong', text=identifier)
        if keyword_tag:
            p_tag = keyword_tag.find_next('p')
            keywords.extend(a_tag.text for a_tag in p_tag.find_all('a'))

    unique_words = set()
    for k in keywords:
        cleaned_keyword = re.sub(r'[^\w\s;]', '', k)
        unique_words.update(word.strip() for word in cleaned_keyword.split(';') if len(word.strip()) > 1)
    return list(unique_words)


def scrape_article(url):
    """
    Scrape information from a single article page.
    """
    response = requests.get(url)
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1', {'itemprop': 'name'}).text.strip() if soup.find('h1', {'itemprop': 'name'}) else None
    abstract = soup.find('div', {'class': 'show__abstract is-long is-initial-letter'}).text.strip() if soup.find('div', {'class': 'show__abstract is-long is-initial-letter'}) else None
    keywords = extract_keywords(soup) if title and abstract else None

    if keywords and len(keywords) >= 5 and len(abstract) >= 500:
        return {'title': title, 'abstract': abstract, 'keywords': keywords}
    return None


def save_to_csv(file_path, data):
    """
    Save a list of data rows to a CSV file.
    """
    with open(file_path, mode='a+', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(data)


def split_data(data, split_ratio, seed=12345):
    """
    Split data into training, validation, and test sets using synchronized indices.
    """
    np.random.seed(seed)  # Ensures reproducibility
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    train_size = int(len(indices) * split_ratio['train'])
    valid_size = int(len(indices) * split_ratio['valid'])

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    train_data = [data[i] for i in train_indices]
    valid_data = [data[i] for i in valid_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, valid_data, test_data

from tqdm.auto import tqdm

def process_articles(urls, folder, target_article_count, save_interval):
    """
    Process each article URL, organize the scraped data, and save it directly until target_article_count are saved.
    """
    
    x = target_article_count
    range_of_numbers = range(1, 1000000)  
    random_numbers = random.sample(range_of_numbers, x)
    
    # Initialize lists to store the scraped data
    corpus_data, queries_data, keywords_data = [], [], []
    saved_count = 0  # Number of saved articles that meet the criteria
    
    # Initialize tqdm progress bar
    pbar = tqdm(total=target_article_count, desc='Processing Articles')

    # Loop through URLs and continue processing articles until target_article_count are saved
    article_urls_iter = iter(urls)
    while saved_count < target_article_count:
        try:
            url = next(article_urls_iter)
            article_data = scrape_article(url)
            if article_data:
                # If the article meets the criteria, process and append to lists
                corpus_data.append([saved_count, preprocess_text(article_data['abstract'])])
                queries_data.append([random_numbers[saved_count], article_data['title']])
                keywords_data.append([saved_count, random_numbers[saved_count], ';'.join([preprocess_text(keyword) for keyword in article_data['keywords']])])

                saved_count += 1  # Increment the count of saved articles
                pbar.update(1)  # Update tqdm progress bar

                # Save data at the save interval or if we've reached the target number of articles
                if saved_count % save_interval == 0 or saved_count == target_article_count:
                    save_to_csv(os.path.join(folder, 'corpus.csv'), corpus_data)
                    save_to_csv(os.path.join(folder, 'queries.csv'), queries_data)
                    save_to_csv(os.path.join(folder, 'keywords.csv'), keywords_data)
                    
                    # Clear the batch lists after saving
                    corpus_data, queries_data, keywords_data = [], [], []

            # Provide a small delay to prevent hammering the server with requests
            if saved_count % 23 == 0:
                time.sleep(1.7)

        except StopIteration:
            # If there are no more URLs to process, break out of the loop
            break

    # Save any remaining data not saved due to interval
    if corpus_data or queries_data or keywords_data:
        save_to_csv(os.path.join(folder, 'corpus.csv'), corpus_data)
        save_to_csv(os.path.join(folder, 'queries.csv'), queries_data)
        save_to_csv(os.path.join(folder, 'keywords.csv'), keywords_data)

    # Close the tqdm progress bar
    pbar.close()

    # Print the total number of saved articles that met the criteria
    print(f"Total articles processed and saved: {saved_count}")





def generate_urls(years, filename, max_pages_pr_year=20, max_articles=1000):
    article_links = get_article_links(years, max_pages_pr_year, max_articles)
    with open(filename + '_links.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for index, url in enumerate(article_links, start=1):
            writer.writerow([index, url])

def webscraping(folder, max_articles=105, save_interval=100, split_ratio={'train': 0.7, 'valid': 0.15, 'test': 0.15}):
    """
    Main webscraping function orchestrating the scraping process.
    """
    
    os.makedirs(folder, exist_ok=True)
    
    # Define train, test, validation folders:
    train_folder = os.path.join(folder, 'train')
    valid_folder = os.path.join(folder, 'valid')
    test_folder = os.path.join(folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    file_path = os.path.join(os.getcwd(), "generic_filename_20231114T202027.csv")
    with open(file_path, 'r', encoding="utf-8") as file:
        data_list = list(csv.reader(file))
    urls = [row[1] for row in data_list][:max_articles]

    # Call process_articles with the folder parameter
    process_articles(urls, folder, max_articles, save_interval)


    # Load and split the saved data
    for data_type in ['corpus', 'queries', 'keywords']:
        with open(os.path.join(folder, f'{data_type}.csv'), 'r', encoding="utf-8") as file:
            data = list(csv.reader(file, delimiter=';'))
            train_data, valid_data, test_data = split_data(data, split_ratio)
            save_to_csv(os.path.join(train_folder, f'train_{data_type}.csv'), train_data)
            save_to_csv(os.path.join(valid_folder, f'valid_{data_type}.csv'), valid_data)
            save_to_csv(os.path.join(test_folder, f'test_{data_type}.csv'), test_data)


"""
# Example usage
if __name__ == "__main__":
    webscraping("article_data", max_articles=105, save_interval=10, split_ratio={'train': 0.7, 'valid': 0.15, 'test': 0.15})
"""
