import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import os
import csv, numpy
from preprocessing import preprocess_text, embedding_text
from scraper import scrape_article  
from tqdm import tqdm
from torch import Tensor
from postprocessing import embed, get_info, get_keywords


def load_model(model_path):
    # Function for loading trained S-Bert model
    
    model = SentenceTransformer(model_path)
    return model

def load_corpus_embeddings(corpus_embeddings_path):
    # Load the precomputed embeddings for the corpus
    with open(corpus_embeddings_path, 'rb') as f:
        corpus_embeddings = pickle.load(f)
    return corpus_embeddings

def jaccard_custom(list1,list2):
    a = []
    if len(list1) <= len(list2):
        for i in list1:
            b = 1 if i in list2 else 0
            a.append(b)
        return sum(a) / len(list1)
    else:
        for i in list2:
            b = 1 if i in list1 else 0
            a.append(b)
        return sum(a) / len(list2)

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

def pytorch_jaccard_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    c = []
    for i in b:
        e = []
        for j in a:
            d = 1 if j in b else 0
            e.append(d)
        c.append(sum(e) / len(i))

def jaccard_sim(a: Tensor, b: Tensor):
    """
    Computes the Jaccard similarity between sets a and b.
    :return: Matrix with res[i][j]  = Jaccard similarity(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    # Convert to sets
    set_a = set(a.tolist()[0])
    set_b = [set(b_i.tolist()) for b_i in b]

    # Compute Jaccard similarity
    #jaccard_sim_matrix = [i for i in set_a = 1 if i in set_b_i for set_b_i in set_b]
    jaccard_sim_matrix = [sum(1 if i in set_b_i else 0 for i in set_a) / len(set_a) for set_b_i in set_b]

    return torch.tensor(jaccard_sim_matrix)


def search_articles(query, model, corpus_embeddings, corpus_ids, top_k=5):
    
    # Preprocess query to same format as the trained data
    user_input = preprocess_text(query)
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    query_embedding = query_embedding.to(corpus_embeddings.device)
    #print(corpus_embeddings.shape)
    #query_embedding = query_embedding.squeeze(0)
    # Compute cosine similarities
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
   
  
    top_results = torch.topk(cos_scores, k=top_k)
    #pos_key = corpus_embeddings[int(top_results[1][0])].numpy()
    
    print("\nTop {} most similar articles in the corpus:".format(top_k))
    for score, idx in zip(top_results[0], top_results[1]):
        title, url = get_info(corpus_ids[idx])
        #score = jaccard_custom(pos_key, corpus_embeddings[int(idx)].numpy())
        print("Title: {} \n (Score: {:.4f}) \n url: {} \n".format(title, score, url))





def main():
    corpus_path = 'datav2/train/train_corpus.csv'
    model_path = "C:/Users/hasse/Skrivebord/02456_DL_SBERT/train_bi-encoder-margin_mse-bert-base-uncased-batch_size_64-2023-11-30_17-14-58"
    corpus_embeddings_path = os.path.join(os.getcwd(), 'datav2/embeddings')
    corpus_path = os.path.join(os.getcwd(), corpus_path)
    model = load_model(model_path)
    corpus_ids = []
    with open(corpus_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split(";")
            corpus_ids.append(pid)
            
    if not os.path.exists(corpus_embeddings_path + '.npy'):
        embed(corpus_path, model, corpus_embeddings_path)
  
    corpus_embeddings = torch.from_numpy(numpy.load('C:/Users/hasse/Skrivebord/02456_DL_SBERT/datav2/embeddings.npy'))
    

    while True:
        user_query = input("Enter your research query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        search_articles(user_query, model, corpus_embeddings,corpus_ids)

if __name__ == "__main__":
    main()

