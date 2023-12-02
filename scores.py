import os, logging, json, pickle, argparse, random, csv, gzip
from tqdm import tqdm
import heapq
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import pandas
from sentence_transformers import SentenceTransformer, models

parser = argparse.ArgumentParser()
parser.add_argument("--max_value_hard_neg", default=0.05, type=float)
parser.add_argument("--max_hard_negs", default=10, type=int)
parser.add_argument("--max_nb_scores", default=150, type=int)
parser.add_argument("--model_name", default='distilbert-base-uncased', type=str)
args = parser.parse_args()

logging.info(str(args))





def jaccard_set(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def jaccard_custom(list1,list2):
    a = []
    if list1 < list2:
        for i in list1:
            b = 1 if i in list2 else 0
            a.append(b)
        return sum(a) / len(list1)
    else:
        for i in list2:
            b = 1 if i in list1 else 0
            a.append(b)
        return sum(a) / len(list2)



filepath_corpus = os.path.join(os.getcwd(), "data/corpus.csv")
filepath_query = os.path.join(os.getcwd(), "data/queries.csv")
filepath_keywords = os.path.join(os.getcwd(), 'data/valid/valid_keywords.csv')
def add_numbers_to_neg(my_dict, model_key, numbers):
    if model_key in my_dict['neg']:
        my_dict['neg'][model_key].extend(numbers)
    else:
        my_dict['neg'][model_key] = numbers
        
def append_dict_to_list(main_dict, sub_dict):
    main_dict.setdefault().append(sub_dict)



def generate_pos_neg(filename):
    """_summary_

    Args:
        filename (_type_): _description_
    """
    print('here')
    data = []
    jacc_custom_scores = {}
    with open(filename, 'r', newline='',encoding='utf-8') as csvfile:
        reader = list(csv.reader(csvfile))
        
        # Store the results for each row
        rows = list(reader)  # Store rows in a list
        # Loop over each row in the CSV file
        for _, row in tqdm(enumerate(reader)):
            a = [item.split(';') for item in row]
            id, qid, keywords = a[0][0], a[0][1], a[0][2:]
            # Extract the values from the third column of the current row
         
            sub_data = {'qid': int(qid),
                        'pos': [],
                        'neg': {}
            }
            # Select x random rows and extract values from the third column
            
         
            
            random_rows = [r for r in random.sample(rows, args.max_nb_scores)]
        
            jacc_custom_scores_rows = {}
            
            ### Adding the pos
            """jacc_custom_scores_rows.update({
                        int(id): float(jaccard_custom(keywords,keywords))
                    })"""
            
            evals = []
            for rand_row in random_rows:
                b = [item.split(';') for item in rand_row]
                rand_id, rand_qid, rand_keywords = b[0][0], b[0][1], b[0][2:]
                rand_keywords = [item.strip('"') for item in rand_keywords]
                #print(rand_keywords)
                
                eval_1 = jaccard_set(keywords,rand_keywords)
                eval_2 = jaccard_custom(keywords,rand_keywords)
                
                evals.append({int(rand_id): float(eval_2)})
              
            
            
            top_two = heapq.nlargest(2, evals, key=lambda x: list(x.values())[0]) # Just taking the two largest numbers for pos ID
            list_lows = [d for d in evals if d not in top_two]
            ids_low = [int(list(d.keys())[0]) for d in list_lows][:25] # Taking the smallest 25 for neg_id
            values_low = [float(list(d.values())[0]) for d in list_lows][:25]
            ids_high = [int(list(d.keys())[0]) for d in top_two]
            values_high = [float(list(d.values())[0]) for d in top_two]
            
            for id_low, value_low in zip(ids_low, values_low):
                add_numbers_to_neg(sub_data, 'jaccard_custom', [int(id_low)])
                jacc_custom_scores_rows.update({
                        int(id_low): float(value_low)
                    })
                
            for id_high, value_high in zip(ids_high, values_high):
                sub_data['pos'].extend([int(id_high)])
                jacc_custom_scores_rows.update({
                        int(id_high): float(value_high)
                    })
            #sub_data['pos'].extend([int(id)])
            
            jacc_custom_scores[int(qid)] = jacc_custom_scores_rows
            data.append(sub_data)
            
  
   
    hard_negs_path = os.path.join(os.getcwd(), 'data/valid_hard_negs.json')
    hard_negs_path_gz = os.path.join(os.getcwd(), 'data/valid_hard_negs.jsonl.gz')
    jacc_scores_path = os.path.join(os.getcwd(), 'data/valid_jaccard_scores.pkl')
    jacc_scores_paths = os.path.join(os.getcwd(), 'data/valid_jaccard_scores.json')
    
    with gzip.open(hard_negs_path_gz, 'wt') as jsonl_gz_file:
        for my_dict in data:
            json_line = json.dumps(my_dict)
            jsonl_gz_file.write(json_line + '\n')
    
    with open(hard_negs_path, 'w') as json_file:
        json.dump(data, json_file)

    # Open the file in binary read mode and load the dictionary
    with open(jacc_scores_path, 'wb') as pickle_file:
        pickle.dump(jacc_custom_scores, pickle_file)
    
    with open(jacc_scores_paths, 'w') as json_file:
        json.dump(jacc_custom_scores, json_file)
        

def generate_pos_neg_pairwise(filename):
    """_summary_

    Args:
        filename (_type_): _description_
    """
   
    jacc_custom_scores = {}
    with open(filename, 'r', newline='',encoding='utf-8') as csvfile:
        reader = list(csv.reader(csvfile))
        
        # Store the results for each row
        rows = list(reader)  # Store rows in a list
        # Loop over each row in the CSV file
        for _, row in tqdm(enumerate(reader)):
            a = [item.split(';') for item in row]
            pid, qid, keywords = a[0][0], a[0][1], a[0][2:]
            # Extract the values from the third column of the current row
         
            sub_data = {'pid': int(pid),
                        'pids': {}
            }
            
            # Select x random rows and extract values from the third column
            random_rows = [r for r in random.sample(rows, args.max_nb_scores)]
            jacc_custom_scores_rows = {}
            for rand_row in random_rows:
                b = [item.split(';') for item in rand_row]
                rand_id, rand_qid, rand_keywords = b[0][0], b[0][1], b[0][2:]
                rand_keywords = [item.strip('"') for item in rand_keywords]
               
                eval = jaccard_custom(keywords,rand_keywords)
                
                jacc_custom_scores_rows.update({
                        int(rand_id): float(eval)
                    })
           
            
            jacc_custom_scores[int(pid)] = jacc_custom_scores_rows
       
            
  
   
    jacc_scores_path = os.path.join(os.getcwd(), 'datav2/jaccard_scores.pkl')
    jacc_scores_paths = os.path.join(os.getcwd(), 'datav2/jaccard_scores.json')
    

    # Open the file in binary read mode and load the dictionary
    with open(jacc_scores_path, 'wb') as pickle_file:
        pickle.dump(jacc_custom_scores, pickle_file)
    
    with open(jacc_scores_paths, 'w') as json_file:
        json.dump(jacc_custom_scores, json_file)
        
generate_pos_neg(filepath_keywords)





def generate_ce_scores(corpus, query):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    def get_embedding(text):
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def compute_similarity_score(embedding1, embedding2):
        return 1 - cosine(embedding1, embedding2)

    
    
    data = []
    ce_scores = {}
    with open(corpus, 'r', newline='',encoding='utf-8') as ce_scores_file, open(query, 'r', newline='',encoding='utf-8') as queries_file:
        abstracts = list(csv.reader(ce_scores_file))
        titles = list(csv.reader(queries_file))
        
        for abstract, title in tqdm(zip(abstracts, titles), total=len(abstracts), desc="Processing Rows"):
            a = [item.split(';') for item in abstract]
            b = [item.split(';') for item in title]
            abs_id, corpus = a[0][0], a[0][1]
            qid, query = b[0][0], b[0][1]
           
          
            sub_data = {'qid': int(qid),
                        'pos': [],
                        'neg': {}
            }
            
            random_rows = [r for r in random.sample(list(abstracts), args.max_nb_scores)]
         
            ce_scores_rows = {}

            evals = []
            for rand_row in random_rows:
                c = [item.split(';') for item in rand_row]
                rand_abs_id, rand_corpus = c[0][0], c[0][1]
             
                embedding1 = get_embedding(corpus)
                embedding2 = get_embedding(rand_corpus)

                similarity_score = compute_similarity_score(embedding1, embedding2)
              
                evals.append({int(rand_abs_id): float(similarity_score)})
         
            
            
            top_two = heapq.nlargest(2, evals, key=lambda x: list(x.values())[0])
            list_lows = [d for d in evals if d not in top_two]
            ids_low = [int(list(d.keys())[0]) for d in list_lows][:25]
            values_low = [float(list(d.values())[0]) for d in list_lows]
            ids_high = [int(list(d.keys())[0]) for d in top_two]
            values_high = [float(list(d.values())[0]) for d in top_two]
            
            for id_low, value_low in zip(ids_low, values_low):
                add_numbers_to_neg(sub_data, 'jaccard_custom', [int(id_low)])
                ce_scores_rows.update({
                        int(id_low): float(value_low)
                    })
                
            for id_high, value_high in zip(ids_high, values_high):
                sub_data['pos'].extend([int(id_high)])
                ce_scores_rows.update({
                        int(id_high): float(value_high)
                    })
            
            ce_scores[int(qid)] = ce_scores_rows
            data.append(sub_data)
    
    hard_negs_path = os.path.join(os.getcwd(), 'data/hard_negs.json')
    hard_negs_path_gz = os.path.join(os.getcwd(), 'data/hard_negs.jsonl.gz')
    jacc_scores_path = os.path.join(os.getcwd(), 'data/jaccard_scores.pkl')
    jacc_scores_paths = os.path.join(os.getcwd(), 'data/jaccard_scores.json')
    
    with gzip.open(hard_negs_path_gz, 'wt') as jsonl_gz_file:
        for my_dict in data:
            json_line = json.dumps(my_dict)
            jsonl_gz_file.write(json_line + '\n')
    
    with open(hard_negs_path, 'w') as json_file:
        json.dump(data, json_file)

    # Open the file in binary read mode and load the dictionary
    with open(jacc_scores_path, 'wb') as pickle_file:
        pickle.dump(ce_scores, pickle_file)
    
    with open(jacc_scores_paths, 'w') as json_file:
        json.dump(ce_scores, json_file)
           

#generate_pos_neg(filepath_keywords)

#generate_ce_scores(filepath_corpus,filepath_query)