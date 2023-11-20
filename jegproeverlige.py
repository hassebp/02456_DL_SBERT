from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import os
import csv
import pickle
import time
import sys


model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
num_candidates = 500
cross_encoder_model = CrossEncoder('cross-encoder/stsb-roberta-base')

dataset_path = "data_generic_filename_20231115T113324/corpus.csv"
max_corpus_size = 20000


# Some local file to cache computed embeddings
embedding_cache_path = 'corpus-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)


#Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("Download dataset")
        util.http_get(url, dataset_path)

    # Get all unique sentences from the file
    corpus_sentences = set()
    with open(dataset_path, encoding='utf8') as fIn:
        reader = csv.reader(fIn, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        next(reader, None)  # Skip the header
        for row in reader:
            if row:  # Check if the row is not empty
                corpus_sentences.add(row[1])
            if len(corpus_sentences) >= max_corpus_size:
                break

            """
            corpus_sentences.add(row[1])
            if len(corpus_sentences) >= max_corpus_size:
                break
            """

    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_tensor=True)

    print("Store file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences'][0:max_corpus_size]
        corpus_embeddings = cache_data['embeddings'][0:max_corpus_size]

###############################
print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))