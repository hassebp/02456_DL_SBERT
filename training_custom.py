import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy
from data_loader import load_json_data
from tqdm import tqdm

# Sample dataset class
class SentenceDataset(Dataset):
    def __init__(self, data, bert_model, tokenizer, max_length=128):
        self.sentences = list(map(lambda data: data.short_description, data))[:100]
        self.url = list(map(lambda data: data.link, data))[:100]
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_length = max_length
      

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence1 = self.sentences[idx]
        inputs1 = self.tokenizer(sentence1, return_tensors='pt', truncation=True, padding=True, max_length=128)
        attention_mask1 = inputs1['attention_mask']
        with torch.no_grad():
            outputs1 = self.bert_model(**inputs1)
        embeddings1 = outputs1.last_hidden_state.mean(dim=1).squeeze(0)  # Average pooling
        url1 = self.url[idx]

        # Randomly choose another sentence as a negative example
        idx2 = numpy.random.choice(len(self.sentences)) 
        idx2 if idx2 != idx else numpy.random.choice(len(self.sentences)) 
        sentence2 = self.sentences[idx2] 
        inputs2 = self.tokenizer(sentence2, return_tensors='pt', truncation=True, padding=True, max_length=128)
        attention_mask2 = inputs2['attention_mask']
        with torch.no_grad():
            outputs2 = self.bert_model(**inputs2)
        embeddings2 = outputs2.last_hidden_state.mean(dim=1).squeeze(0)  # Average pooling
        url2 = self.url[idx2]

        return {
            "input_ids1": embeddings1,
            "attention_mask1": attention_mask1.squeeze(0),
            "input_ids2": embeddings2,
            "attention_mask2": attention_mask2.squeeze(0),
            "url1": url1,
            "url2": url2
        }


# Siamese network model
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(SiameseNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.fc = nn.Linear(self.embedding_size, 128)
        self.pooling = nn.AdaptiveMaxPool1d(self.embedding_size*4)
        self.fc2 = nn.Linear(self.embedding_size*4,128)
      

    def forward(self, x1, x2):
        x1 = x1.to(self.fc.weight.dtype)
        x2 = x2.to(self.fc.weight.dtype)
        print(numpy.shape(x1),numpy.shape(x2))
        
        # Apply linear transformation to both input tensors
        x1 = self.fc(x1)
        x2 = self.pooling(x2)
        x2 = self.fc2(x2)

   
        return x1, x2

# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2) +
                                      torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

# Helper function to get BERT embeddings
def get_bert_embeddings(sentences, model, tokenizer):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Training loop
def train_siamese_network(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    with tqdm() as pbar:
        for epoch in range(num_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                output1, output2 = model(torch.cat([batch['input_ids1'], batch['input_ids2']],dim=1),torch.cat([batch['attention_mask1'], batch['attention_mask2']],dim=1))
                print(numpy.shape(output1),numpy.shape(output2))
                p.p
                loss = criterion(output1, output2)  
    
                loss.backward()
                optimizer.step()
                pbar.update(1)
                # Report
                if pbar.n % 5 == 0:
                    loss = loss.detach().cpu().item()
                    pbar.set_description(f"epoch={epoch+1}, step={pbar.n}, loss={loss:.2f}")

        torch.save(model.state_dict(), 'C:/Users/hasse/Skrivebord/02456_DL_SBERT/tester.pth')
# Sample usage
embedding_size = 768  # BERT hidden size
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Assuming you have a list of sentences for the dataset
sentences = [
    {"sentence": "Your sentence 1 here.", "url": "https://example.com/1"},
    {"sentence": "Your sentence 2 here.", "url": "https://example.com/2"},
    {"sentence": "aaaa.", "url": "https://example.com/3"},
    {"sentence": "bbbb", "url": "https://example.com/4"},
]

data = load_json_data('C:/Users/hasse/Skrivebord/02456_DL_SBERT/News_Category_Dataset_v3.json')

dataset = SentenceDataset(data, bert_model, tokenizer)
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

# Modify the SiameseNetwork to accept BERT embeddings
siamese_model = SiameseNetwork(embedding_size)
criterion = ContrastiveLoss()
optimizer = optim.Adam(siamese_model.parameters(), lr=0.001)





def train_model():
    train_siamese_network(siamese_model, dataloader, criterion, optimizer)
    
    
def test_model():
    # Load the saved model
    model_path = 'C:/Users/hasse/Skrivebord/02456_DL_SBERT/tester.pth'
    loaded_model = torch.load(model_path)

    # New sentence
    new_sentence = "bb"

    with torch.no_grad():
        new_sentence_embedding = loaded_model.encode([new_sentence], convert_to_tensor=True)

   
    # Calculate cosine similarity with the training embeddings
    similarities = torch.nn.functional.cosine_similarity(new_sentence_embedding, loaded_model.encode([]))  # Provide empty list as no additional training sentences are needed

    # Find the index of the training sentence with the highest similarity
    index_of_max_similarity = torch.argmax(similarities).item()
    print(index_of_max_similarity)
    p.p
    # Retrieve the corresponding URL
    closest_url = training_urls[index_of_max_similarity]


    # Get the corresponding URL
    closest_url = urls[index_of_max_similarity]
    print(closest_url)