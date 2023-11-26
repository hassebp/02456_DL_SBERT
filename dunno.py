from sentence_transformers import SentenceTransformer, util

# Load your custom SentenceTransformer model
model_path = 'C:/Users/hasse/OneDrive - Danmarks Tekniske Universitet/Data_DL_02546/output/train_bi-encoder-margin_mse-bert-base-uncased-batch_size_16-2023-11-12_14-04-21'
custom_model = SentenceTransformer(model_path)
model = SentenceTransformer('bert-base-uncased')

input_sentence = "Input sentence for similarity."


input_embedding = model.encode(input_sentence, convert_to_tensor=True)
custom_embedding = custom_model.encode(input_sentence, convert_to_tensor=True)
# Calculate cosine similarities
similarities = util.pytorch_cos_sim(input_embedding, custom_embedding)[0]

most_similar_index = similarities.argmax()


#most_similar_sentence = all_sentences[most_similar_index]


print(f"Similarity: {most_similar_index}")
