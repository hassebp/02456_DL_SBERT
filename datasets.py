from torch.utils.data import Dataset
import numpy
import torch

# Sample dataset class
class ArticleDataset(Dataset):
    def __init__(self, data, bert_model, tokenizer, max_length=128):
        self.sentences = list(map(lambda data: data.short_description, data))[:10000]
        self.url = list(map(lambda data: data.link, data))[:10000]
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
        
        
class MSMarcoDataset(Dataset):
    def __init__(self) -> None:
        super().__init__(Dataset)