from matplotlib import pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import transformers
from sentence_transformers import SentenceTransformer
from postprocessing import embed
import os
import pandas as pd
import numpy
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import seaborn as sns
sns.set(style="whitegrid")
import torch

def plot_attention_map(attention_map, queries_labels, keys_labels, print_values:bool=False, ax=None, color_bar:bool=True):
    """Plot the attention weights as a 2D heatmap"""
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,6), dpi=300)
    else:
        fig = plt.gcf()
    im = ax.imshow(attention_map, cmap=sns.color_palette("viridis", as_cmap=True))
    ax.grid(False)
    ax.set_ylabel("$\mathbf{Q}$")
    ax.set_xlabel("$\mathbf{K}$")
    ax.set_yticks(np.arange(len(queries_labels)))
    ax.set_yticklabels(queries_labels)
    ax.set_xticks(np.arange(len(keys_labels)))
    ax.set_xticklabels(keys_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if print_values:
        for i in range(len(queries_labels)):
            for j in range(len(keys_labels)):
                text = ax.text(j, i, f"{attention_map[i, j]:.2f}",
                            ha="center", va="center", color="w")

    if color_bar:
      fig.colorbar(im, fraction=0.02, pad=0.04)
    fig.tight_layout()
    
def show_attention_map(model, corpus, keywords):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model = transformers.AutoModel.from_pretrained(model).eval()
    
    # Define some documents and questions
    #documents = [f"{city} is the capital of {country}" for city, country in corpus]
    #questions =  [f"What is the capital city of {country}?" for country in keywords]

    # Tokenizer
    tokenizer_args = {"padding":True, "truncation":True, "return_tensors": "pt"}
    documents_ = tokenizer(corpus, **tokenizer_args)
    questions_ = tokenizer(keywords, **tokenizer_args)

    # Compute the question and document vectors
    with torch.no_grad():
        document_vectors = model(**documents_).pooler_output
        question_vectors = model(**questions_).pooler_output

    # compute the inner product (attention map) between all pairs of questions and documents
    attention_map = question_vectors @ document_vectors.T
    attention_map = attention_map.softmax(dim=1)

    # plot as an attention map
    plot_attention_map(attention_map, keywords, corpus)
    
#### Read the corpus files, that contain all the passages. Store them in the corpus dict
train_corpus = []         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join('data/test/test_corpus.csv')

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split(";")
        train_corpus.append(passage)


train_keywords = []
keywords_filepath = os.path.join('data/test/test_keywords.csv')

with open(keywords_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        row = line.strip().split(";")
        pid, qid, keywordss = row[0], row[1], row[2:]
        train_keywords.append(keywordss)
        
model = 'test'


def plot_accuracies(filepath):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath, sep=',')

    # Set the Seaborn style
    sns.set(style="whitegrid")

    # Plot the accuracies using Seaborn
    plt.figure(figsize=(10, 6))
    
    # Euclidean accuracy
    sns.lineplot(x='epoch', y=100*df['accuracy_euclidean'], data=df, label='Euclidean', linewidth=3)
    
    # Uncomment the following lines if you want to include other accuracy metrics
    # sns.lineplot(x='epoch', y=100*df['accuracy_manhattan'], data=df, label='Manhattan', linewidth=3)
    # sns.lineplot(x='epoch', y=100*df['accuracy_cosinus'], data=df, label='Cosine', linewidth=3)

    # Set plot limits and labels
    plt.ylim([0, 100])
    plt.ylabel('Accuracy', fontsize=24)
    plt.xlabel('Epochs', fontsize=24)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Show the plot
    plt.show()





def plot_nicestuff():
    corpus_path = 'data/train/train_corpus.csv'
    model_path = "C:/Users/hasse/Skrivebord/02456_DL_SBERT/test"
    corpus_embeddings_path = os.path.join(os.getcwd(), 'data/embeddings')
    corpus_path = os.path.join(os.getcwd(), corpus_path)
    model =  SentenceTransformer(model_path)
    corpus_ids = []
    corpus = []
    with open(corpus_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split(";")
            corpus_ids.append(pid)
            corpus.append(passage)
            
    if not os.path.exists(corpus_embeddings_path + '.npy'):
        embed(corpus_path, model, corpus_embeddings_path)
  
    corpus_embeddings = torch.from_numpy(numpy.load('C:/Users/hasse/Skrivebord/02456_DL_SBERT/data/embeddings.npy'))
    
    
    root_path = os.getcwd()
    keywords_filepath = os.path.join(root_path, 'data/train/train_keywords.csv')
    keywords = {}
    with open(keywords_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            row = line.strip().split(";")
            pid, qid, keywordss = row[0], row[1], row[2:]
            qid = int(qid)
            pid = int(pid)
            keywords[qid] = keywordss
    
   
    """queries = {}        #dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(root_path, 'data/train/train_queries.csv')

    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            try:
                qid, query, title = line.strip().split(";")
                qid = int(qid)
                queries[qid] = title
            except:
                continue
    
    ids = queries.keys()
    queries_list = []
    for id in ids:
        a = ' '.join(keywords[id]).replace('"',' ')
        text = queries[id] + a
        
        queries_list.append(text)"""
    
    top_1 = []
    top_3 = []
    top_5 = []
    top_10 = []
    for query in tqdm(corpus, total=len(corpus)):
      
        query_embedding = model.encode(query, convert_to_tensor=True)
        query_embedding = query_embedding.to(corpus_embeddings.device)
        distances = [euclidean(query_embedding, validation_embedding) for validation_embedding in corpus_embeddings]
        # Reshape distances to a 2D array (required by StandardScaler)
        maxd = numpy.max(distances)
        mind = numpy.min(distances)
        distances = distances / maxd
       
        top_5_indices = numpy.argsort(distances)[:10]
        top_distances = [distances[i] for i in top_5_indices]
        
        for score in top_distances:
            score = 1 - score
            top_10.append(score)
        
        for score in top_distances[:5]:
            score = 1 - score
            top_5.append(score)
            
        for score in top_distances[:3]:
            score = 1 - score 
            top_3.append(score)
            
        score = 1 - top_distances[0]
        top_1.append(score)

    print("This is mean accuracy for top 1:", numpy.mean(top_1))
    print("This is mean accuracy for top 3:", numpy.mean(top_3))
    print("This is mean accuracy for top 5:", numpy.mean(top_5))
    print("This is mean accuracy for top 10:", numpy.mean(top_10))
    
#plot_nicestuff()



plot_accuracies('C:/Users/hasse/Skrivebord/02456_DL_SBERT/test/eval/triplet_evaluation_results.csv')