# scraper.py
import os
import csv
import time
import requests
from bs4 import BeautifulSoup
from preprocessing import preprocess_text
import random
from tqdm import tqdm


# Function to get links to all articles within a given range of yearspage={page}&
def get_article_links(years, max_pages_pr_year, max_articles):
    start_year, end_year = years[0], years[1]
    links = []
    for year in range(start_year, end_year + 1):
        # type=article_journal just to only get the journals
        base_url = f"https://findit.dtu.dk/en/catalog?type=article_journal_article"
        url_year = f"{base_url}&year={year}"
        for page in range(1, max_pages_pr_year + 1):
            
            url = f"{url_year}&page={page}"
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                article_links = soup.find_all('a', {'class': 'result__title'})

                # Extract and print the URLs
                for link in article_links:
                    if len(links) >= max_articles:
                        break
                    article_url = link['href']
                    links.append(article_url)
                # Should remove if any duplicates somehow?
                links = list(set(links))  
        # Dunno if you can get banned by IP from DTU Findit if you make too many requests? So I just put in a timer
        time.sleep(0.5)
    return links

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


def generate_urls(years, filename, max_pages_pr_year=20, max_articles=1000):
    article_links = get_article_links(years, max_pages_pr_year, max_articles)
    with open(filename + '_links.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for index, url in enumerate(article_links, start=1):
            writer.writerow([index, url])


def webscraping(filename, max_articles=105, save_interval=10):
    """ Main webscraping function to orchestrate the scraping process. """
    file_path = os.path.join(os.getcwd(), "generic_filename_20231114T202027.csv")
    folder = 'data_' + filename
    os.makedirs(folder, exist_ok=True)
    
    ### Generates a random number for qid
    x = max_articles
    range_of_numbers = range(1, 1000000)  
    random_numbers = random.sample(range_of_numbers, x)
    
    
    # Initialize lists to store batches of scraped data
    batch_corpus_data = []
    batch_queries_data = []
    batch_keywords_data = []
    
    with open(file_path, 'r') as file:
        data_list = list(csv.reader(file))

    urls = [row[1] for row in data_list][:max_articles]

    for index, url in tqdm(enumerate(urls, start=1)):
        article_data = scrape_article(url)
        if article_data:
            preprocessed_title = preprocess_text(article_data['title'])
            preprocessed_abstract = preprocess_text(article_data['abstract'])
            preprocessed_keywords = [preprocess_text(keyword) for keyword in article_data['keywords']]

            # Append the processed data to the respective batch lists
            batch_corpus_data.append([index, preprocessed_abstract])
            batch_queries_data.append([random_numbers[index], preprocessed_title])
            batch_keywords_data.append([index, random_numbers[index], ';'.join(preprocessed_keywords)])
        
        # Save data at the save interval
        if index % save_interval == 0 or index == len(urls):
            write_to_csv(folder, 'corpus.csv', batch_corpus_data, mode='a+')
            write_to_csv(folder, 'queries.csv', batch_queries_data, mode='a+')
            write_to_csv(folder, 'keywords.csv', batch_keywords_data, mode='a+')

            # Clear the batch lists after saving
            batch_corpus_data = []
            batch_queries_data = []
            batch_keywords_data = []

        if index % 2500 == 0:
            time.sleep(1)
    
    # Save any remaining data
    if batch_corpus_data:
        write_to_csv(folder, 'corpus.csv', batch_corpus_data, mode='a+')
    if batch_queries_data:
        write_to_csv(folder, 'queries.csv', batch_queries_data, mode='a+')
    if batch_keywords_data:
        write_to_csv(folder, 'keywords.csv', batch_keywords_data, mode='a+')
