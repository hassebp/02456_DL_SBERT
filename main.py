import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import os
import csv, numpy
from preprocessing import preprocess_text, embedding_text
from scraper import scrape_article  
from tqdm import tqdm
from postprocessing import embed, get_info
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
    
    # Preprocess query to same format as the trained data
    user_input = preprocess_text(query)
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    query_embedding = query_embedding.to(corpus_embeddings.device)
    #print(corpus_embeddings.shape)
    #query_embedding = query_embedding.squeeze(0)
    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    top_results = torch.topk(cos_scores, k=top_k)

    print("\nTop {} most similar articles in the corpus:".format(top_k))
    for score, idx in zip(top_results[0], top_results[1]):
        title, url = get_info(corpus_ids[idx])
        print("Title: {} (Score: {:.4f}) \n url: {} \n".format(title, score, url))





def main():
    corpus_path = 'data_articlev2/corpus.csv'
    model_path = "C:/Users/hasse/OneDrive - Danmarks Tekniske Universitet/Data_DL_02546/output/train_bi-encoder-margin_mse-bert-base-uncased-batch_size_16-2023-11-12_14-04-21"
    corpus_embeddings_path = os.path.join(os.getcwd(), 'data_articlev2/embeddings')
    corpus_path = os.path.join(os.getcwd(), corpus_path)
    model = load_model(model_path)
    corpus_ids = []
    with open(corpus_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split(";")
            corpus_ids.append(pid)
            
    if not os.path.exists(corpus_embeddings_path + '.npy'):
        embed(corpus_path, model, corpus_embeddings_path)
  
    corpus_embeddings = torch.from_numpy(numpy.load('C:/Users/hasse/Skrivebord/02456_DL_SBERT/data_articlev2/embeddings.npy'))
    

    while True:
        user_query = input("Enter your research query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        search_articles(user_query, model, corpus_embeddings,corpus_ids)

if __name__ == "__main__":
    main()

