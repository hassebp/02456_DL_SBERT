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


### Now we read the MS Marco dataset
data_folder = 'data'


val_corpus = {}

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'train/train_corpus.csv')

logging.info("Read corpus: corpus.csv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split(";")
        pid = int(pid)
        corpus[pid] = passage


ce_scores_file = os.path.join(data_folder, 'jaccard_scores.pkl')
logging.info("Load Jaccard scores dict")
with open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)
    
logging.info("Train corpus': {}".format(len(corpus)))

class MSMARCODataset(Dataset):
    def __init__(self, corpus, ce_scores):
        self.corpus_ids = list(corpus.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores
        self.scores = []
        for pid in self.corpus_ids:
            sentence1 = self.corpus[pid]
            
            for passage_id, score in zip(self.ce_scores[pid].keys(), self.ce_scores[pid].values()):
                sentence2 = self.corpus[passage_id]
                score = score
                self.scores.append(InputExample(texts=[sentence1, sentence2], label=score))

    def __getitem__(self, item):
        output = self.scores[item]
        
        return output


    def __len__(self):
        return len(self.corpus)*2 ## Try multiplying with 50 future

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(corpus=corpus, ce_scores=ce_scores)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.CosineSimilarityLoss(model=model)


### Load and prepare validation
val_corpus = {}
  
val_collection_filepath = os.path.join(data_folder, 'valid/valid_corpus.csv')
logging.info("Read corpus: corpus.csv")
with open(val_collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split(";")
        pid = int(pid)
        val_corpus[pid] = passage

logging.info("Validation corpus': {}".format(len(val_corpus)))
   
val_scores_file = os.path.join(data_folder, 'val_jaccard_scores.pkl')
logging.info("Load Jaccard scores dict validation")
with open(val_scores_file, 'rb') as fIn:
     val_scores_load = pickle.load(fIn)

val_sentence1 = []
val_sentence2 = []
val_scores = []
val_corpus_ids = list(val_corpus.keys())
for pid in val_corpus_ids:
    sentence1 = val_corpus[pid]
    for passage_id, score in zip(val_scores_load[pid].keys(), val_scores_load[pid].values()):
        sentence2 = val_corpus[passage_id]
        val_sentence1.append(sentence1)
        val_sentence2.append(sentence2)
        val_scores.append(score)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          evaluator=evaluation.EmbeddingSimilarityEvaluator(val_sentence1, val_sentence2, val_scores),
          evaluation_steps=args.eval_steps,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=10000,
          optimizer_params = {'lr': args.lr},
          output_path='output/test'
          )

# Train latest model
model.save(model_save_path)