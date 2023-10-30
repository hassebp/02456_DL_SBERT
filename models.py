from sentence_transformers import SentenceTransformer
from pelutils import TT 

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