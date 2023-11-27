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


def split_data(data, split_ratio):
    """
    Split data into training, validation, and test sets.
    """
    random.shuffle(data)
    train_size = int(len(data) * split_ratio['train'])
    valid_size = int(len(data) * split_ratio['valid'])
    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]
    return train_data, valid_data, test_data


def process_articles(urls, max_articles, save_interval):
    """
    Process each article URL and organize the scraped data.
    """
    
    x = max_articles
    range_of_numbers = range(1, 1000000)  
    random_numbers = random.sample(range_of_numbers, x)
    
    corpus_data, queries_data, keywords_data = [], [], []

    for index, url in tqdm(enumerate(urls, start=0), total=max_articles):
        if index < max_articles:
            article_data = scrape_article(url)
            if article_data:
                corpus_data.append([index, preprocess_text(article_data['abstract'])])
                queries_data.append([random_numbers[index], article_data['title']])
                keywords_data.append([index, random_numbers[index], ';'.join([preprocess_text(keyword) for keyword in article_data['keywords']])])

                if index % save_interval == 0 or index == len(urls) - 1:
                    yield corpus_data, queries_data, keywords_data
                    corpus_data, queries_data, keywords_data = [], [], []

    if corpus_data or queries_data or keywords_data:
        yield corpus_data, queries_data, keywords_data


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

    # Save the entire data
    for corpus_batch, queries_batch, keywords_batch in process_articles(urls, max_articles, save_interval):
        print(len(queries_batch))
        save_to_csv(os.path.join(folder, 'corpus.csv'), corpus_batch)
        save_to_csv(os.path.join(folder, 'queries.csv'), queries_batch)
        save_to_csv(os.path.join(folder, 'keywords.csv'), keywords_batch)

    # Load and split the saved data
    for data_type in ['corpus', 'queries', 'keywords']:
        with open(os.path.join(folder, f'{data_type}.csv'), 'r+', encoding="utf-8") as file:
            data = list(csv.reader(file, delimiter=';'))
            train_data, valid_data, test_data = split_data(data, split_ratio)
            save_to_csv(os.path.join(train_folder, f'train_{data_type}.csv'), train_data)
            save_to_csv(os.path.join(valid_folder, f'valid_{data_type}.csv'), valid_data)
            save_to_csv(os.path.join(test_folder, f'test_{data_type}.csv'), test_data)

# Example usage
if __name__ == "__main__":
    webscraping("article_data", max_articles=105, save_interval=10, split_ratio={'train': 0.7, 'valid': 0.15, 'test': 0.15})
