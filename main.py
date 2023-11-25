import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import os
import csv
from preprocessing import preprocess_text, embedding_text
from scraper import scrape_article  


def load_model(model_path):
    # Function for loading trained S-Bert model
    
    model = SentenceTransformer(model_path)
    return model

def load_corpus_embeddings(corpus_embeddings_path):
    # Load the precomputed embeddings for the corpus
    with open(corpus_embeddings_path, 'rb') as f:
        corpus_embeddings = pickle.load(f)
    return corpus_embeddings


### csv:
def load_corpus_embeddings_csv(corpus_embeddings_path):
    # Load embeddings from a CSV file
    embeddings = []
    with open(corpus_embeddings_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            embeddings.append([float(x) for x in row])  # Assuming each row is an embedding
    return embeddings

def load_corpus_ids_csv(corpus_ids_path):
    # Load corpus IDs from a CSV file
    ids = []
    with open(corpus_ids_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ids.append(row[0])  # Assuming first column contains the ID
    return ids




def search_articles(query, model, corpus_embeddings, corpus_ids, top_k=5):
    # Encode the query to the same space as the corpus

    # Preprocess query to same format as the trained data
    user_input = preprocess_text(query)
    query_embedding = embedding_text(user_input)

    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    top_results = torch.topk(cos_scores, k=top_k)

    print("\nTop {} most similar articles in the corpus:".format(top_k))
    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus_ids[idx], "(Score: {:.4f})".format(score))




def main():
    model_path = "path/to/your/model"
    corpus_embeddings_path = "path/to/your/corpus_embeddings.csv"
    corpus_data_path = "path/to/your/corpus_data.csv"

    model = load_model(model_path)
    corpus_embeddings = load_corpus_embeddings_csv(corpus_embeddings_path)
    corpus_ids = load_corpus_ids_csv(corpus_data_path)

    while True:
        user_query = input("Enter your research query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        search_articles(user_query, model, corpus_embeddings, corpus_ids)

if __name__ == "__main__":
    main()

