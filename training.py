import torch
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from data_loader import load_json_data
from tqdm import tqdm

data = load_json_data('C:/Users/hasse/Skrivebord/02456_DL_SBERT/News_Category_Dataset_v3.json')

### Hyperparams 
hparams = {
    'epochs': 100,
    'step': 0,
    'num_steps':100,
    'batch_size': 16,
    'num_workers': 1,
}

class ArticleDataset(Dataset):
    def __init__(self, data, model):
        self.sentences = [article.short_description for article in data]
        self.labels = [article.headline for article in data]
        self.model = model
        self.label_map = {label: idx for idx, label in enumerate(set(self.labels))}
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return {'sentence': sentence, 'label': self.label_map[label]}

def collate_batch(batch):
    sentences = [sample['sentence'] for sample in batch]
    labels = [sample['label'] for sample in batch]
    return {'sentences': sentences, 'labels': labels}

def train_sbert(data, model_path='sbert_model', epochs=100, batch_size=32):
    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model, just took one from https://www.sbert.net/
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    dataset = ArticleDataset(data, model)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
 
    # CosineSimilarityLoss from article
    train_loss = losses.CosineSimilarityLoss(model)

    # SBERT training loop
    
    model.train()
    total_loss = 0
    step =  hparams['step']

    with tqdm(total=hparams['num_steps']) as pbar:
        while hparams['step'] < hparams['num_steps']:
            for batch in train_loader:
              
                sentences_batch = batch['sentences']
                # Ensure sentences_batch is a list of sentences
                if not isinstance(sentences_batch, list):
                    sentences_batch = [sentences_batch]
                    
                embeddings = model.encode(sentences_batch, convert_to_tensor=True).to(device)
                labels_tensor = torch.tensor(batch['labels'], dtype=torch.long).to(device)

                loss = train_loss(embeddings, labels_tensor)
                
                total_loss += loss.item()
                loss.backward()

                # Update model parameters
                model.optimizer.step()
                model.zero_grad()
                
                step += 1
                pbar.update(1)

                # Report
                if step % 5 ==0 :
                    loss = loss.detach().device()
                    pbar.set_description(f"epoch={1}, step={step}, loss={loss:.1f}")

            average_loss = total_loss / len(train_loader)
            print(f'Epoch {1+1}/{epochs}, Average Loss: {average_loss}')

    # Save the trained model
    model.save(model_path)
    
def test_training():
    train_sbert(data=data, model_path='sbert_model', epochs=hparams['epochs'], batch_size=hparams['batch_size'])
    
    
    
