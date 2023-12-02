import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
from torch.utils.data import Dataset
import random
from shutil import copyfile
import pickle
import argparse
import torch, numpy
from preprocessing import preprocess_text

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=500, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--eval_steps", default=500, type=int)
args = parser.parse_args()

logging.info(str(args))



# The  model we want to fine-tune
train_batch_size = args.train_batch_size          #Increasing the train batch size improves the model performance, but requires more GPU memory
model_name = args.model_name
max_passages = args.max_passages
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory

num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs         # Number of epochs we want to train

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name, device='cuda')
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda')

model_save_path = f'output/train_bi-encoder-margin_mse-{model_name.replace("/", "-")}-batch_size_{train_batch_size}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


# Write self to path
os.makedirs(model_save_path, exist_ok=True)

train_script_path = os.path.join(model_save_path, 'train_script.py')
copyfile(__file__, train_script_path)
with open(train_script_path, 'a') as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))



"""
DATA VALIDATION LOADING
"""
valid_data_folder = 'data/valid'

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
val_corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
val_corpus_path = os.path.join(valid_data_folder, 'valid_corpus.csv')

logging.info("Loading validation: validation corpus")
with open(val_corpus_path, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, passage = line.strip().split(";")
        qid = int(qid)
        val_corpus[qid] = passage
        
#### Read the corpus files, that contain all the passages. Store them in the corpus dict
val_queries = {}         #dict in the format: passage_id -> passage. Stores all existent passages
val_queries_path =  os.path.join(valid_data_folder, 'valid_queries.csv')

logging.info("Loading validation: validation queries")
with open(val_queries_path, 'r', encoding='utf8') as fIn:
    for line in fIn:
        try:
            qid, passage, q = line.strip().split(";")
            qid = int(qid)
            val_queries[qid] = passage
        except:
            continue
      

val_keywords = {}
keywords_filepath =  os.path.join(valid_data_folder, 'valid_keywords.csv')
logging.info("Loading validation: validation keywords")
with open(keywords_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        row = line.strip().split(";")
        pid, qid, keywordss = row[0], row[1], row[2:]
        qid = int(qid)
        val_keywords[qid] = keywordss


"""
DATA TRAINING LOADING
"""

train_data_folder = 'data/train'

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
train_corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(train_data_folder, 'train_corpus.csv')

logging.info("Loading trianing: training corpus")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split(";")
        pid = int(pid)
        train_corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(train_data_folder, 'train_queries.csv')
logging.info("Loading training: training queries")
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        try:
            qid, passage, q = line.strip().split(";")
            qid = int(qid)
            queries[qid] = passage
        except:
            continue
       
        

train_keywords = {}
keywords_filepath = os.path.join(train_data_folder, 'train_keywords.csv')
logging.info("Loading training: keywords corpus")
with open(keywords_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        row = line.strip().split(";")
        pid, qid, keywordss = row[0], row[1], row[2:]
        qid = int(qid)
        train_keywords[qid] = keywordss

    
train_queries = {}
hard_negatives_filepath = os.path.join('data', 'hard_negs.jsonl.gz')
logging.info("Read hard negatives train file")

train_keys = queries.keys()

negs_to_use = None
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn):
        if max_passages > 0 and len(train_queries) >= max_passages:
            break
        
        data = json.loads(line)
        
        if not data['qid'] in queries:
            continue
        
        #Get the positive passage ids
        pos_pids = data['pos']
        
        #Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if args.negs_to_use is not None:    #Use specific system for negatives
                negs_to_use = args.negs_to_use.split(",")
            else:   #Use all systems
                negs_to_use = list(data['neg'].keys())
            logging.info("Using negatives from the following systems:", negs_to_use)

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break
                    
        if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}
                

"""
SETUP DATASET
"""

logging.info("Train queries: {}".format(len(train_queries)))

# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, keywords):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.keywords = keywords
        self.corpus = corpus
        
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])
            
    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']
        qid = query['qid']
        keywords = ' '.join(self.keywords[qid])
        query_text = query_text + keywords.replace('"',' ')
        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)    #Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)
        else:   #We only have negatives, use two negs
            pos_id = query['neg'].pop(0)    #Pop negative and add at end
            pos_text = self.corpus[pos_id]
            query['neg'].append(pos_id)

        #Get a negative passage
        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)
       
        return InputExample(texts=[query_text, pos_text, neg_text], label=1)
    

    def __len__(self):
        return len(self.queries)


"""
SETUP VALIDATION
"""
valid_queries = {}
hard_negatives_filepath = os.path.join('data', 'valid_hard_negs.jsonl.gz')
logging.info("Read hard negatives valid file")

negs_to_use = None
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn):
        if max_passages > 0 and len(valid_queries) >= max_passages:
            break
        
        data = json.loads(line)
        
        if not data['qid'] in val_queries:
            continue
        
        #Get the positive passage ids
        pos_pids = data['pos']
        
        #Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if args.negs_to_use is not None:    #Use specific system for negatives
                negs_to_use = args.negs_to_use.split(",")
            else:   #Use all systems
                negs_to_use = list(data['neg'].keys())
            logging.info("Using negatives from the following systems:", negs_to_use)

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break
                    
        if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            valid_queries[data['qid']] = {'qid': data['qid'], 'query': val_queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}
                


class GenerateValidationTriplets():
    def __init__(self, validation_queries, validation_corpus, validation_keywords):
        self.queries = validation_queries
        self.queries_ids = list(self.queries.keys())
        self.keywords = validation_keywords
        self.corpus = validation_corpus
   
        
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])
            
    def getdata(self):
        val_anchor = []
        pos_sentence = []
        neg_sentence = []
        for item in range(len(self.queries)):
            query = self.queries[self.queries_ids[item]]
            query_text = query['query']
            qid = query['qid']
            keywords = ' '.join(self.keywords[qid])
            query_text = query_text + keywords.replace('"',' ')
            if len(query['pos']) > 0:
                pos_id = query['pos'].pop(0)    #Pop positive and add at end
                pos_text = self.corpus[pos_id]
                query['pos'].append(pos_id)
            else:   #We only have negatives, use two negs
                pos_id = query['neg'].pop(0)    #Pop negative and add at end
                pos_text = self.corpus[pos_id]
                query['neg'].append(pos_id)

            #Get a negative passage
            neg_id = query['neg'].pop(0)    #Pop negative and add at end
            neg_text = self.corpus[neg_id]
            query['neg'].append(neg_id)
            
            val_anchor.append(query_text)
            pos_sentence.append(pos_text)
            neg_sentence.append(neg_text)
        
        return val_anchor, pos_sentence, neg_sentence
    
    
val_anchor, pos_sentence, neg_sentence = GenerateValidationTriplets(valid_queries, val_corpus, val_keywords).getdata()
"""
TRAIN MODEL
"""
# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(queries=train_queries, corpus=train_corpus, keywords=train_keywords)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.TripletLoss(model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluation.TripletEvaluator(anchors=val_anchor, positives=pos_sentence, negatives=neg_sentence),
          #evaluation_steps=args.eval_steps,
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          #checkpoint_path=model_save_path,
          #checkpoint_save_steps=500,
          optimizer_params = {'lr': args.lr},
          output_path='test/'
          )

# Train latest model
model.save(model_save_path)