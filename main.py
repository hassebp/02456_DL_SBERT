import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import os
from preprocessing import preprocess_text 
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



def search_articles(query, model, corpus_embeddings, corpus_ids, top_k=5):
    # Encode the query to the same space as the corpus

    # Preprocess query to same format as the trained data
    user_input = preprocess_text(query)
    
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    top_results = torch.topk(cos_scores, k=top_k)

    print("\nTop {} most similar articles in the corpus:".format(top_k))
    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus_ids[idx], "(Score: {:.4f})".format(score))






def main():
    model_path = ""
    
    # skal være pickle filer. vi kan ændre til csv hvis vi går med det
    corpus_embeddings_path = "orpus_embeddings.pkl"
    corpus_ids_path = "ids.pkl"

    # Load model and corpus embeddings
    model = load_model(model_path)
    corpus_embeddings = load_corpus_embeddings(corpus_embeddings_path)
    
    # Load a list of titles / identifiers for your corpus articles to output
    with open(corpus_ids_path, 'rb') as f:
        corpus_ids = pickle.load(f)

    # User query input
    user_query = input("Enter your research query: ")

    # Search articles
    while True:
        user_query = input("Enter your research query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        search_articles(user_query, model, corpus_embeddings, corpus_ids)



if __name__ == "__main__":
    main()
