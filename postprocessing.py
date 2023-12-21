from tqdm import tqdm
import torch, numpy, os, csv
def embed(corpus_embeddings_path, model , save_path):
    passages = []
    ### This should be made a seperate function :)))
    with open(corpus_embeddings_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split(";")
            #pid = int(pid)
            passages.append(passage)
    corpus_embeddings = []
    # Use tqdm to create a progress bar
    for passage in tqdm(passages, desc="Encoding sentences"):
        # Get embeddings for each sentence and append to the list
        embedding = model.encode(passage, convert_to_tensor=True)
        corpus_embeddings.append(embedding)
        
    corpus_embeddings = torch.stack(corpus_embeddings, dim=0)
    numpy.save(save_path + '.npy', corpus_embeddings.cpu().numpy())
    
    
def get_info(corpus_id):
    """
    Get title and url from corpus id
    """
    root_path = os.getcwd()
    mapping_p_q = {}        #dict in the format: query_id -> query. Stores all training keywords
    keywords_filepath = os.path.join(root_path, 'data/test/test_keywords.csv')

    with open(keywords_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            row = line.strip().split(";")
            pid, qid = row[0], row[1]
            qid = int(qid)
            pid = int(pid)
            mapping_p_q[pid] = qid
    
   
    queries = {}        #dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(root_path, 'data/test/test_queries.csv')

    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            try:
                qid, query, title = line.strip().split(";")
                qid = int(qid)
                queries[qid] = title
            except:
                continue
       
    
    url_list = {}
    with open(os.path.join(root_path, 'generic_filename_20231114T202027.csv'), 'r') as file:
        for line in file:
            id, url = line.strip().split(",")
            id = int(id)
            url_list[id] = url
        
    title_qid = mapping_p_q[int(corpus_id)]
    title = queries[title_qid]
    
    return title, url_list[int(corpus_id)]

def get_keywords(corpus_id):
    """
    Get title and url from corpus id
    """
    root_path = os.getcwd()
    keywords = {}        #dict in the format: query_id -> query. Stores all training keywords
    keywords_filepath = os.path.join(root_path, 'data_articlev2/keywords.csv')

    with open(keywords_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            row = line.strip().split(";")
            pid, qid, keyword = row[0], row[1], row[2:]
            keywords[int(pid)] = keyword
    
    return keywords[int(corpus_id)]