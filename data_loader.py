import json, gzip, numpy, os
from dataclasses import dataclass

# Just to make sure its in the right format
@dataclass
class NewsArticle:
    link: str
    headline: str
    category: str
    short_description: str
    authors: str
    date: str
    

def load_json_data(json_file_path):
    articles = []
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            article_data = json.loads(line)
            article = NewsArticle(**article_data)
            articles.append(article)
    return articles


def test_msmarco():
    file_path = os.path.join(os.getcwd(), 'msmarco-doctrain-top100.gz')

    with gzip.open(file_path, 'rt') as file:
        content = file.read()

    print(numpy.shape(content))


def test_data_loading():
    json_file_path = 'C:/Users/hasse/Skrivebord/02456_DL_SBERT/News_Category_Dataset_v3.json'

    articles = load_json_data(json_file_path)

    print(f'There are {len(articles)} articles available in this dataset')

    try:
        articles = load_json_data(json_file_path)
       
        if articles:
            first_article = articles[0]
            print(f"Link: {first_article.link}")
            print(f"Headline: {first_article.headline}")
            print(f"Category: {first_article.category}")
            print(f"Short Description: {first_article.short_description}")
            print(f"Authors: {first_article.authors}")
            print(f"Date: {first_article.date}")
        else:
            print("No articles found in the JSON file.")
    except ValueError as e:
        print(f"Error: {e}")