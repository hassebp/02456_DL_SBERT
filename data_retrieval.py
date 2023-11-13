import requests, os, csv, time
from bs4 import BeautifulSoup
from pelutils import TT
from uuid import uuid4
# Function to get information from a single article page
def scrape_article(url):
    # Get response from link, to see if site is available
    response = requests.get(url)
    if response.status_code == 200:
        # Load in the html code
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title, abstract, and keywords
        title = soup.find('h1', {'itemprop': 'name'})
        
        ### All of this just checks if there is a title, abstract and keywords, else return None
        if title:
            title = title.text.strip()
        else:
            return None
        abstract = soup.find('div', {'class': 'show__abstract is-long is-initial-letter'})
        
        if abstract:
            abstract = abstract.text.strip()
        else:
            return None

        keywords = []
        identifier_keywords = soup.find('strong', text='Keywords')
        if identifier_keywords:
            # Find the <p> tag after the <strong> tag
            p_tag = identifier_keywords.find_next('p')

            # Find all <a> tags within the <p> tag
            all_a_tags = p_tag.find_all('a')

            # Print the text content of each <a> tag
            for a_tag in all_a_tags:
                keywords.append(a_tag.text)
        else:
            return None
                
        identifier_keywords = soup.find('strong', text='Other keywords')
        if identifier_keywords:
            # Find the <p> tag after the <strong> tag
            p_tag = identifier_keywords.find_next('p')

            # Find all <a> tags within the <p> tag
            all_a_tags = p_tag.find_all('a')

            # Print the text content of each <a> tag
            for a_tag in all_a_tags:
                keywords.append(a_tag.text)
        
        # Making sure no duplicates in keywords
        keywords = list(set(word.lower() for word in keywords))
      
        # Thinking we need enough keywords? And a long enough abstract
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


def generate_urls(years, filename, max_pages_pr_year: int = 20, max_articles: int = 10e5):
    article_links = get_article_links(years, max_pages_pr_year, max_articles)
    with open(filename + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for index, url in enumerate(article_links, start=1):
            writer.writerow([index] + [url])
       
        



# Main function to initiate the scraping process
def webscraping(filename):
    
    file_path = 'C:/Users/hasse/Skrivebord/02456_DL_SBERT/generic_filename_20231114T000628.csv'
    data_list = []
    folder = 'data_' + filename
    os.makedirs(folder)
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)

        # Iterate over rows and append them to the list
        for row in csv_reader:
            data_list.append(row)
    # Testing on the 50 first
    urls = [row[1] for row in data_list][:50]
 
    for index, url in enumerate(urls, start=1):
        data = scrape_article(url)
        if data == None:
            continue
        
        # Just to ensure no DDOS
        if index % 25 == 0:
            time.sleep(1)
        
        with open(os.path.join(folder,'corpus_' + filename + '.csv'), mode='a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([index, data['abstract']])
            
        with open(os.path.join(folder,'queries_' + filename + '.csv'), mode='a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([str(uuid4()).split('-')[0], data['title']])
            
        with open(os.path.join(folder,'keywords_' + filename + '.csv'), mode='a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([index, data['keywords']])
    


