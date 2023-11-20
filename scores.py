from sentence_transformers import util
import os, gzip, tqdm, json, pickle, numpy, random, csv
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
    main_dict.setdefault('sub_dicts', []).append(sub_dict)

def generate_pos_neg(filename):
    """_summary_

    Args:
        filename (_type_): _description_
    """
    data = {}
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
            random_rows = [r for r in random.sample(list(reader), 900)]
            cnt1 = 0
            cnt2 = 0
            for rand_row in random_rows:
                b = [item.split(';') for item in rand_row]
                rand_id, rand_qid, rand_keywords = b[0][0], b[0][1], b[0][2:]
                eval_1 = jaccard_set(keywords,rand_keywords)
                eval_2 = jaccard_custom(keywords,rand_keywords)
                if (eval_1 < 0.05 and cnt1 < 10):
                    add_numbers_to_neg(sub_data, 'jaccard', [int(rand_id)])
                    cnt1 += 1
                   
                if (eval_2 < 0.05 and cnt2 < 10):
                    add_numbers_to_neg(sub_data, 'jaccard_custom', [int(rand_id)])
                    cnt2 += 1
                    
            
            append_dict_to_list(data, sub_data)

    
    test = os.path.join(os.getcwd(), 'hard_negs.json')
    with open(test, 'w') as json_file:
        json.dump(data, json_file)
    
    
generate_pos_neg(file_path)



    