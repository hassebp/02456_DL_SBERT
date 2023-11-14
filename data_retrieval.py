import requests, os, csv, time
from bs4 import BeautifulSoup
from pelutils import TT
from uuid import uuid4
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re




# Define a function to preprocess text data
def preprocess_text(text):
    # Initialize the Porter Stemmer
    ps = PorterStemmer()
    
    # Tokenize the text, remove stop words, and apply stemming
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    stemmed = [ps.stem(word) for word in tokens if word not in ENGLISH_STOP_WORDS]
    
    # Remove non-word characters/numbers and extra whitespace
    cleaned_text = re.sub(r'[^\w\s]', ' ', ' '.join(stemmed))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


# Function to scrape information from a single article page
def scrape_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Attempt to extract title, abstract, and keywords
        title = soup.find('h1', {'itemprop': 'name'})
        abstract = soup.find('div', {'class': 'show__abstract is-long is-initial-letter'})
        keywords = []
        
        # If title or abstract is missing, return None
        if not title or not abstract:
            return None
        
        # Process title and abstract
        title = title.text.strip()
        abstract = abstract.text.strip()
        
        # Extract keywords and additional keywords if present
        for identifier in ['Keywords', 'Other keywords']:
            keyword_tag = soup.find('strong', text=identifier)
            if keyword_tag:
                p_tag = keyword_tag.find_next('p')
                keywords.extend([a_tag.text for a_tag in p_tag.find_all('a')])
        
        # Remove duplicates and ensure a sufficient number of keywords and abstract length
        keywords = list(set(word.lower() for word in keywords))
        if len(keywords) < 5 or len(abstract) < 500:
            return None
        
        return {'title': title, 'abstract': abstract, 'keywords': keywords}
    else:
        print(f"Failed to fetch {url}")
        return None

# Function to get links to all articles within a given range of yearspage={page}&
def get_article_links(years, max_pages_pr_year, max_articles):
    TT.profile(f"Starting webscraping for {max_articles} articles")
    start_year, end_year = years[0], years[1]
    links = []
    for year in range(start_year, end_year + 1):
        TT.profile(f"Time for webscraping for year: {year}")
        # type=article_journal just to only get the journals
        base_url = f"https://findit.dtu.dk/en/catalog?type=article_journal_article"
        url_year = f"{base_url}&year={year}"
        for page in range(1, max_pages_pr_year + 1):
            #TT.profile(f"Webscraping for year: {year}, page: {page}")
            url = f"{url_year}&page={page}"
            response = requests.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                article_links = soup.find_all('a', {'class': 'result__title'})

                # Extract and print the URLs
                for link in article_links:
                    if len(links) >= max_articles:
                        #TT.end_profile()
                        TT.end_profile()
                        TT.end_profile()
                        print(TT)
                        break
                    article_url = link['href']
                    links.append(article_url)
                # Should remove if any duplicates somehow?
                links = list(set(links))  
            #TT.end_profile()
        # Dunno if you can get banned by IP from DTU Findit if you make too many requests? So I just put in a timer
        time.sleep(0.5)
        TT.end_profile()
    TT.end_profile()
    print(TT)
    return links


def generate_urls(years, filename, max_pages_pr_year=20, max_articles=1000):
    article_links = get_article_links(years, max_pages_pr_year, max_articles)
    with open(filename + '_links.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for index, url in enumerate(article_links, start=1):
            writer.writerow([index, url])
       
        

def webscraping(filename, file_path, max_articles=50):
    folder = 'data_' + filename
    os.makedirs(folder, exist_ok=True)


# Main function to initiate the scraping process
def webscraping(filename):
    file_path = os.path.join(os.getcwd(), "generic_filename_20231114T000628.csv")
    data_list = []
    folder = 'data_' + filename
    os.makedirs(folder, exist_ok=True)
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        # Iterate over rows and append them to the list
        for row in csv_reader:
            data_list.append(row)
    
    # Testing on the 50 first
    urls = [row[1] for row in data_list][:50]
 
    for index, url in enumerate(urls, start=1):
        data = scrape_article(url)
        if data is None:
            continue
        
        
        # Apply the preprocessing to title, abstract, and keywords
        preprocessed_title = preprocess_text(data['title'])
        preprocessed_abstract = preprocess_text(data['abstract'])
        preprocessed_keywords = [preprocess_text(keyword) for keyword in data['keywords']]
        
        
        # Just to ensure no DDOS
        if index % 25 == 0:
            time.sleep(1)
        
        # Writing abstracts
        with open(os.path.join(folder, 'corpus_' + filename + '.csv'), mode='a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([index, preprocessed_abstract])
        
        # Writing titles
        with open(os.path.join(folder, 'queries_' + filename + '.csv'), mode='a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(uuid4()).split('-')[0], preprocessed_title])
        
        # Writing keywords
        with open(os.path.join(folder, 'keywords_' + filename + '.csv'), mode='a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            keywords = '; '.join(preprocessed_keywords)  # Join keywords into a single string separated by semicolons
            writer.writerow([index, keywords])
            
        

    


