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
from matplotlib import pyplot
import os
from datasets import ArticleDataset
from loss_functions import CosineSimilarityLoss, ContrastiveLoss
from models import SiameseNetwork

# Helper function to get BERT embeddings
def get_bert_embeddings(sentences, model, tokenizer):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Training loop
def train_siamese_network(model, dataloader, criterion, optimizer, num_epochs=10):
    num_epochs = 100
    epoch_losses = []
    with tqdm(total=num_epochs*10000) as pbar:
        for epoch in range(num_epochs):
            model.train()
            batch_losses = []
            for batch in dataloader:
                optimizer.zero_grad()
                output1, output2 = model(batch['input_ids1'],batch['attention_mask1'], batch['input_ids2'],batch['attention_mask2'])
    
                loss = criterion(output1, output2)  
    
                loss.backward()
                optimizer.step()
                pbar.update(1)
                batch_losses.append(loss.detach().cpu().item())
                # Report
                if pbar.n % 5 == 0:
                    loss = loss.detach().cpu().item()
                    pbar.set_description(f"epoch={epoch+1}, step={pbar.n}, loss={loss:.2f}")
            avg_epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(avg_epoch_loss)
        torch.save(model.state_dict(), 'C:/Users/hasse/Skrivebord/02456_DL_SBERT/tester.pth')
        
        
    # Plot the training curve
    pyplot.plot(range(1, num_epochs + 1), epoch_losses, label='Average Loss')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.title('Training Curve')
    pyplot.grid()
    pyplot.legend()
    pyplot.show()
        
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

data = load_json_data(os.path.join(os.getcwd(),'News_Category_Dataset_v3.json'))

dataset = ArticleDataset(data, bert_model, tokenizer)
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

# Modify the SiameseNetwork to accept BERT embeddings
siamese_model = SiameseNetwork(embedding_size)
criterion = ContrastiveLoss()
optimizer = optim.Adam(siamese_model.parameters(), lr=0.001)





def train_model():
    train_siamese_network(siamese_model, dataloader, criterion, optimizer)
    
    
def test_model():
    # Load the saved model
    model_path = os.path.join(os.getcwd(), 'tester.pth')
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