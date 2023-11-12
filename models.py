from sentence_transformers import SentenceTransformer
from pelutils import TT 
from torch import nn
import torch


def SBERT():
    
    TT.profile("Loading model")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    TT.end_profile()
    sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
    TT.profile("Encoding sentences")
    sentence_embeddings = model.encode(sentences)
    TT.end_profile()
    TT.profile("Printing embeddings")
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
    TT.end_profile()
    print(TT)
    
    
# Siamese network model
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(SiameseNetwork, self).__init__()
        pooling_size = 64
        self.embedding_size = embedding_size
        self.fc = nn.Linear(self.embedding_size + pooling_size, 128)
        self.pooling = nn.AdaptiveMaxPool1d(pooling_size)
      

    def forward(self, input1, att1, input2, att2):
        input1 = input1.to(self.fc.weight.dtype)
        input2 = input2.to(self.fc.weight.dtype)
        att1 = att1.to(self.fc.weight.dtype)
        att2 = att2.to(self.fc.weight.dtype)
        
        att1 = self.pooling(att1)
        att2 = self.pooling(att2)
        
        x1 = torch.cat([input1, att1], dim=1)
        x2 = torch.cat([input2, att2], dim=1)
   
        # Apply linear transformation to both input tensors
        x1 = self.fc(x1)
        x2 = self.fc(x2)

   
        return x1, x2