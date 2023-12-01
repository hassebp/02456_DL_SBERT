import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.readers import InputExample
from torch import nn, optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse, logging, sys, json, gzip, os, random, pickle
from datetime import datetime
from torch.utils.data import Dataset
from shutil import copyfile


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=500, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
args = parser.parse_args()

logging.info(str(args))

# The  model we want to fine-tune
train_batch_size = args.train_batch_size          #Increasing the train batch size improves the model performance, but requires more GPU memory
model_name = args.model_name
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
num_epochs = args.epochs         # Number of epochs we want to train
num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
max_passages = args.max_passages
# Define your custom dataset and DataLoader
# You may need to implement your own dataset class and DataLoader based on your data format

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
data_folder = 'data_articlev2'

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'corpus.csv')

logging.info("Read corpus: corpus.csv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split(";")
        pid = int(pid)
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, 'queries.csv')

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split(";")
        qid = int(qid)
        queries[qid] = query

### Read the train queries, store in queries dict
mapping_q_p = {}        #dict in the format: query_id -> query. Stores all training keywords
mapping_p_q = {}        #dict in the format: query_id -> query. Stores all training keywords
keywords_filepath = os.path.join(data_folder, 'keywords.csv')

with open(keywords_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        row = line.strip().split(";")
        pid, qid, keywords = row[0], row[1], row[2:]
        qid = int(qid)
        pid = int(pid)
        mapping_q_p[qid] = pid
        mapping_p_q[pid] = qid

ce_scores_file = os.path.join(data_folder, 'jaccard_scores.pkl')
logging.info("Load Jaccard scores dict")
with open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)
    

hard_negatives_filepath = os.path.join(data_folder, 'hard_negs.jsonl.gz')
logging.info("Read hard negatives train file")
train_queries = {}
negs_to_use = None
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in tqdm(fIn):
        if max_passages > 0 and len(train_queries) >= max_passages:
            break
        data = json.loads(line)
       
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
                


logging.info("Train queries: {}".format(len(train_queries)))


# Example InputExample:
# input_example = InputExample(texts=[text1, text2], label=1.0)

# Example DataLoader:
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, ce_scores):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[int(qid)]['pos'])
            self.queries[qid]['neg'] = list(self.queries[int(qid)]['neg'])
            random.shuffle(self.queries[qid]['neg'])
        
    def __len__(self):
        return len(self.queries_ids)
    
    def __getitem__(self, item):
        """
        Get corpus texts based on pos_id and neg_id as input for text and their score 
        from ce_scores based on (qid, pid) -> pos_text from mapping
        mapping_p_q = pid to qid
        mapping_q_p = qid to pid
        """
        
        query = self.queries[self.queries_ids[item]]
        examples = []
        ### For more than one positive against all negs. 
        for pos_id in query['pos']:
            neg_ids = self.ce_scores[mapping_p_q[pos_id]]
            pos_text = corpus[int(pos_id)]
            for neg_id in neg_ids:
                neg_text = corpus[int(neg_id)]
                score = self.ce_scores[mapping_p_q[pos_id]][neg_id]
                return InputExample(texts=[pos_text, neg_text], label=score)
        
  

def collate_fn(batch):
    examples = []
    for example_data in batch:
        texts, label = example_data
        examples.append(InputExample(texts=texts, label=label))
    
    return examples

dataset = MSMARCODataset(queries=train_queries, corpus=corpus, ce_scores=ce_scores)
#train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)

# Create separate DataLoaders for training and validation
train_dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
loss_function = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, loss_function)], epochs=1)
model.save('fine_tuned_sbert_model')

p.p

# Load pre-trained SBERT model
model = SentenceTransformer(model_name)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_function = losses.CosineSimilarityLoss(model)

# Number of training steps and evaluation steps
evaluation_steps = 100

def calculate_validation_loss(model, validation_data_loader):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for val_batch in validation_data_loader:
        sentences1, sentences2, labels = val_batch
        with torch.no_grad():
            # Forward pass
            embeddings1 = model(sentences1)
            embeddings2 = model(sentences2)
            # Calculate loss
            loss = loss_function(embeddings1, embeddings2, labels)
            total_loss += loss.item()
            num_batches += 1

    # Calculate average loss
    average_loss = total_loss / num_batches
    return average_loss

# Training loop
for step in tqdm(range(num_epochs), desc="Training"):
    # Set the model to training mode
    model.train()
    
    # Fetch a batch of data
    for data_batch in train_data_loader:
        print(data_batch)
        p.p
        
        
        sentences1 = [" ".join(example.texts[0]) for example in data_batch]
        sentences2 = [" ".join(example.texts[1]) for example in data_batch]
        labels = [example.label for example in data_batch]
        print(labels)
        p.p
        for example in data_batch:
            # Extract data for the current example
            sentences, label = example[0], example[1]

            embeddings1 = model(str(sentences[0]))
            embeddings2 = model(str(sentences[1]))
            label = torch.tensor(label, dtype=torch.float32).to(model.device)
            # Calculate loss
            loss = loss_function(embeddings1, embeddings2, label)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    """ # Evaluate every x steps
    if step % evaluation_steps == 0:
        # Set the model to evaluation mode
        model.eval()
        # Perform evaluation on your validation set
        with torch.no_grad():
            # Calculate validation loss and other metrics
            validation_loss = calculate_validation_loss(model, val_data_loader)
            # Print or log the evaluation results
            print(f"Step {step}, Validation Loss: {validation_loss}")"""

# Save the fine-tuned model
model.save('fine_tuned_sbert_model')