"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.

Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""

from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import logging
import sys
import os
import tarfile
from itertools import islice
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Name of the SBERT model
model_name = sys.argv[1]

# You can limit the approx. max size of the corpus. Pass 100 as second parameter and the corpus has a size of approx 100k docs
corpus_max_size = int(sys.argv[2])*1000 if len(sys.argv) >= 3 else 0


####  Load model

model = SentenceTransformer(model_name)

### Data files
data_folder = 'data_articlev2'
os.makedirs(data_folder, exist_ok=True)

collection_filepath = os.path.join(data_folder, 'corpus.csv')
dev_queries_file = os.path.join(data_folder, 'queries.csv')
qrels_filepath = os.path.join(data_folder, 'keywords.csv')

### Load data

corpus = {}             #Our corpus pid => passage
dev_queries = {}        #Our dev queries. qid => query
dev_rel_docs = {}       #Mapping qid => set with relevant pids
needed_pids = set()     #Passage IDs we need
needed_qids = set()     #Query IDs we need

# Load the 6980 dev queries
with open(dev_queries_file, encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split(";")
        dev_queries[qid] = query.strip()


# Load which passages are relevant for which queries
with open(qrels_filepath, encoding='utf8') as fIn:
    for line in fIn:
        row = line.strip().split(';')
        pid, qid = row[0], row[1]
        
        if qid not in dev_queries:
            continue


        if qid not in dev_rel_docs:
            dev_rel_docs[qid] = set()
        dev_rel_docs[qid].add(pid)

        needed_pids.add(pid)
        needed_qids.add(qid)


# Read passages
with open(collection_filepath, encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split(";")
        passage = passage

        if pid in needed_pids or corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
            corpus[pid] = passage.strip()




#corpus = dict(islice(corpus.items(), 5000))


## Run evaluator
logging.info("Queries: {}".format(len(dev_queries)))
logging.info("Corpus: {}".format(len(corpus)))

ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                        show_progress_bar=True,
                                                        corpus_chunk_size=5000,
                                                        precision_recall_at_k=[10, 100],
                                                        name="msmarco dev")

ir_evaluator(model)