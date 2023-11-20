from sentence_transformers import util
import os, gzip, tqdm, json, pickle, numpy
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
data_folder = 'data_generic_filename_20231115T113324'

import random

x = 100  # Replace 10 with your desired length
range_of_numbers = range(1, 1000000)  # Adjust the range as needed

random_numbers = random.sample(range_of_numbers, x)

print(random_numbers)

### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, 'queries.csv')

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split(";")
        qid = int(qid)
        queries[qid] = query

print(numpy.shape(queries))


    