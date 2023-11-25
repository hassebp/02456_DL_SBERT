# preprocessing.py
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re, csv, random, os

def preprocess_text(text):
    ps = PorterStemmer()
    words = [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
    tokens = word_tokenize(" ".join(words))
    stemmed = [ps.stem(token) for token in tokens]
    cleaned_text = re.sub(r'[^\w\s]|_|\d', ' ', ' '.join(stemmed))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def embedding_text(preprocessed_text):
    # Transforms the preprocessed text to embeddings
    embeddings = model.encode(preprocessed_text, convert_to_tensor=True)
    return embeddings


def split_queries(query_filename):
    
    # Specify the paths for the two output CSV files
    name = query_filename.split('.')[0]
    output_csv_70_percent_path = os.path.join(os.getcwd(), name + '_train.csv')
    output_csv_30_percent_path = os.path.join(os.getcwd(), name + '_validate.csv')

    # Read the entire CSV into a list of rows
    with open(query_filename, 'r', encoding='utf-8') as input_csv:
        csv_reader = csv.reader(input_csv)
        header = next(csv_reader)  # Assuming the first row is the header
        rows = list(csv_reader)

    # Shuffle the rows randomly
    random.shuffle(rows)

    # Calculate the split indices based on the desired percentages
    split_index = int(0.7 * len(rows))

    # Split the rows into two parts
    rows_70_percent = rows[:split_index]
    rows_30_percent = rows[split_index:]

    # Write the 70% portion to the output CSV file
    with open(output_csv_70_percent_path, 'w', encoding='utf-8', newline='') as output_csv_70_percent:
        csv_writer_70_percent = csv.writer(output_csv_70_percent)
        csv_writer_70_percent.writerow(header)
        csv_writer_70_percent.writerows(rows_70_percent)

    # Write the 30% portion to the output CSV file
    with open(output_csv_30_percent_path, 'w', encoding='utf-8', newline='') as output_csv_30_percent:
        csv_writer_30_percent = csv.writer(output_csv_30_percent)
        csv_writer_30_percent.writerow(header)
        csv_writer_30_percent.writerows(rows_30_percent)

    print("CSV files split successfully.")
    
def generate_binary_answer_file(keywords_file):
        
    # Specify the path for the output TSV file
    output_tsv_path = os.path.join(os.getcwd(), 'test.tsv')

    # Read the CSV file and create a list of tuples with the desired format
    data_tuples = []
    with open(keywords_file, 'r',encoding='utf-8') as input_csv:
        csv_reader = csv.reader(input_csv)
        for row in csv_reader:
          
            row = row[0].split(";")
         
            data_tuples.append((row[0], '0', row[1], '1'))

    # Write the data to the output TSV file
    with open(output_tsv_path, 'w', newline='', encoding='utf-8') as output_tsv:
        tsv_writer = csv.writer(output_tsv, delimiter='\t')
       
        tsv_writer.writerows(data_tuples)

    print("TSV file created successfully.")