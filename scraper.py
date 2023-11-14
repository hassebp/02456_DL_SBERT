# scraper.py
import os
import csv
import time
import requests
from bs4 import BeautifulSoup
from preprocessing import preprocess_text
from uuid import uuid4

def scrape_article(url):
    """ Scrape information from a single article page. """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1', {'itemprop': 'name'})
    abstract = soup.find('div', {'class': 'show__abstract is-long is-initial-letter'})

    if not title or not abstract:
        return None

    title = title.text.strip()
    abstract = abstract.text.strip()
    keywords = extract_keywords(soup)

    if len(keywords) < 5 or len(abstract) < 500:
        return None

    return {'title': title, 'abstract': abstract, 'keywords': keywords}


def extract_keywords(soup):
    """ Extract keywords from the BeautifulSoup object. """
    keywords = []
    for identifier in ['Keywords', 'Other keywords']:
        keyword_tag = soup.find('strong', text=identifier)
        if keyword_tag:
            p_tag = keyword_tag.find_next('p')
            keywords.extend(a_tag.text for a_tag in p_tag.find_all('a'))
    return list(set(keywords))


def write_to_csv(folder, filename, data_rows, mode='a+'):
    """ Write a list of data rows to a CSV file. """
    with open(os.path.join(folder, filename), mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(data_rows)  # writerows is used to write multiple rows


def generate_urls(years, fileame, max_pages_pr_year=20, max_articles=1000):
    article_links = get_article_links(years, max_pages_pr_year, max_articles)
    with open(filename + '_links.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for index, url in enumerate(article_links, start=1):
            writer.writerow([index, url])


def webscraping(filename, max_articles=105, save_interval=10):
    """ Main webscraping function to orchestrate the scraping process. """
    file_path = os.path.join(os.getcwd(), "generic_filename_20231114T000628.csv")
    folder = 'data_' + filename
    os.makedirs(folder, exist_ok=True)
    
    # Initialize lists to store batches of scraped data
    batch_corpus_data = []
    batch_queries_data = []
    batch_keywords_data = []
    
    with open(file_path, 'r') as file:
        data_list = list(csv.reader(file))

    urls = [row[1] for row in data_list][:max_articles]

    for index, url in enumerate(urls, start=1):
        article_data = scrape_article(url)
        if article_data:
            preprocessed_title = preprocess_text(article_data['title'])
            preprocessed_abstract = preprocess_text(article_data['abstract'])
            preprocessed_keywords = [preprocess_text(keyword) for keyword in article_data['keywords']]

            # Append the processed data to the respective batch lists
            batch_corpus_data.append([index, preprocessed_abstract])
            batch_queries_data.append([str(uuid4()).split('-')[0], preprocessed_title])
            batch_keywords_data.append([index, ';'.join(preprocessed_keywords)])
        
        # Save data at the save interval
        if index % save_interval == 0 or index == len(urls):
            write_to_csv(folder, 'corpus.csv', batch_corpus_data, mode='a+')
            write_to_csv(folder, 'queries.csv', batch_queries_data, mode='a+')
            write_to_csv(folder, 'keywords.csv', batch_keywords_data, mode='a+')

            # Clear the batch lists after saving
            batch_corpus_data = []
            batch_queries_data = []
            batch_keywords_data = []

        if index % 25 == 0:
            time.sleep(1)
    
    # Save any remaining data
    if batch_corpus_data:
        write_to_csv(folder, 'corpus.csv', batch_corpus_data, mode='a+')
    if batch_queries_data:
        write_to_csv(folder, 'queries.csv', batch_queries_data, mode='a+')
    if batch_keywords_data:
        write_to_csv(folder, 'keywords.csv', batch_keywords_data, mode='a+')
