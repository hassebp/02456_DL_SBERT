import requests
from bs4 import BeautifulSoup

# Function to get information from a single article page
def scrape_article(url):
    # Get response from link, to see if site is available
    response = requests.get(url)
    if response.status_code == 200:
        # Load in the html code
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title, abstract, and keywords
        title = soup.find('h1', {'itemprop': 'name'}).text.strip()
        abstract = soup.find('div', {'class': 'show__abstract is-long is-initial-letter'}).text.strip()

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
        

        return {'title': title, 'abstract': abstract, 'keywords': keywords}
    else:
        print(f"Failed to fetch {url}")
        return None

# Function to get links to all articles within a given range of years
def get_article_links(start_year, end_year):
    base_url = "https://findit.dtu.dk/en/catalog"
    links = []

    for year in range(start_year, end_year + 1):
        url = f"{base_url}?fq=publishDate: {year}"
        response = requests.get(url)
    
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            print(soup)
            p.p
            articles = soup.find_all('a', {'class': 'title'})
            print(articles[0:10])
            p.p
            links.extend([f"{base_url}/{article['href'].split('/')[-1]}" for article in articles])

    return links

# Main function to initiate the scraping process
def webscraping():
    # Insert url - this is just for testing 
    return print(scrape_article('https://findit.dtu.dk/en/catalog/6230c6250f6bfb853b3f3d46'))
    
    start_year = 2018
    end_year = 2022

    article_links = get_article_links(start_year, end_year)

    for link in article_links:
        article_info = scrape_article(link)
        if article_info:
            print(article_info)
            # You can save the information to a file or database here

