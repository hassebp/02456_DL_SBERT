import os, logging, tqdm, json, pickle, argparse, random, csv, gzip


parser = argparse.ArgumentParser()
parser.add_argument("--max_value_hard_neg", default=0.05, type=float)
parser.add_argument("--max_hard_negs", default=10, type=int)
parser.add_argument("--max_nb_scores", default=50, type=int)
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


### Now we read the MS Marco dataset
file_path = os.path.join(os.getcwd(), "data_article/keywords.csv")

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
    data = []
    jacc_custom_scores = {}
    with open(filename, 'r', newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        
        # Store the results for each row

        # Loop over each row in the CSV file
        for _, row in tqdm.tqdm(enumerate(reader)):
            a = [item.split(';') for item in row]
            id, qid, keywords = a[0][0], a[0][1], a[0][2:]
            # Extract the values from the third column of the current row
            sub_data = {'qid': int(qid),
                        'pos': [int(id)],
                        'neg': {}
            }
            # Select x random rows and extract values from the third column
            
            
            
            random_rows = [r for r in random.sample(list(reader), args.max_nb_scores)]
            cnt1 = 0
            cnt2 = 0
            cnt1_ce = 0
            jacc_custom_scores_rows = {}
            
            ### Adding the pos
            jacc_custom_scores_rows.update({
                        int(id): float(jaccard_custom(keywords,keywords))
                    })
            
            
            for rand_row in random_rows:
                b = [item.split(';') for item in rand_row]
                rand_id, rand_qid, rand_keywords = b[0][0], b[0][1], b[0][2:]
                eval_1 = jaccard_set(keywords,rand_keywords)
                eval_2 = jaccard_custom(keywords,rand_keywords)
                if (eval_1 < args.max_value_hard_neg and cnt1 < args.max_hard_negs):
                    add_numbers_to_neg(sub_data, 'jaccard', [int(rand_id)])
                    cnt1 += 1
                   
                if (eval_2 < args.max_value_hard_neg and cnt2 < args.max_hard_negs):
                    add_numbers_to_neg(sub_data, 'jaccard_custom', [int(rand_id)])
                    jacc_custom_scores_rows.update({
                        int(rand_id): float(eval_2)
                    })
                    cnt2 += 1
                    
                """if cnt1_ce < args.max_nb_scores:
                    jacc_custom_scores_rows.update({
                        int(rand_id): float(eval_2)
                    })
                    cnt1_ce += 1"""
            jacc_custom_scores[int(qid)] = jacc_custom_scores_rows
            data.append(sub_data)
            
            #append_dict_to_list(data, sub_data)

    
    hard_negs_path = os.path.join(os.getcwd(), 'hard_negs.json')
    hard_negs_path_gz = os.path.join(os.getcwd(), 'hard_negs.jsonl.gz')
    jacc_scores_path = os.path.join(os.getcwd(), 'jaccard_scores.pkl')
    jacc_scores_paths = os.path.join(os.getcwd(), 'jaccard_scores.json')
    
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
        
        

        
        
    
generate_pos_neg(file_path)



    