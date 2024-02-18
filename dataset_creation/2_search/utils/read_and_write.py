
import os
import pickle
import numpy as np
import random
import json


def read_doc_enc_from_pickle(dataset_name, model_name, log_dir, user_id=""):
    """
    Read the embedding from a pickled file
    :return: [array, array]: document embeddings and associated document ids
    """

    name = "{}_{}.pkl".format(dataset_name, model_name)
    # Example: ./embeddings/nfcorpus/nfcorpus_contriever.pkl
    log_dir = os.path.join(log_dir, user_id, dataset_name)

    embeddings, docids = [], []
    with open(os.path.join(log_dir, name), 'rb') as f:
        while 1:
            try:
                emb, ids = pickle.load(f)
                embeddings.append(emb)
                docids.append(ids)
            except EOFError:
                break
    embeddings, docids = np.concatenate(embeddings, axis=0), np.concatenate(docids, axis=0)
    return embeddings, docids


def get_embedding_subset(doc_embeds, subsample_size=1000000):
    if len(doc_embeds) > subsample_size:
        random_indx = random.sample(range(len(doc_embeds)), subsample_size)
    else:
        random_indx = list(range(len(doc_embeds)))
    return doc_embeds[random_indx]


def save_enc_to_pickle(embeddings, docids, dataset_name, model_name, log_dir, batch_num,
                       user_id=""):
    name = "{}_{}.pkl".format(dataset_name, model_name)
    # Example: ./log_dir/nfcorpus/nfcorpus_contriever.pkl
    log_dir = os.path.join(log_dir, user_id, dataset_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # If file exists - remove it first
    if os.path.exists(os.path.join(log_dir, name)) and batch_num == 0:
        os.remove(os.path.join(log_dir, name))

    if batch_num == 0:
        writing_type = 'wb'
    else:
        writing_type = 'ab+'

    with open(os.path.join(log_dir, name), writing_type) as f:
        pickle.dump((embeddings, docids), f)
    return True


def save_search_results(queries, doc_ids, scores, indices,
                        where_to_save, run_id="test"):
    """
    :return: results in a format of a dictionary,
             suitable for EvaluateRetrieval function from BEIR lib
    """

    # if queries is a dictionary, then we need to extract the keys
    if isinstance(queries, dict):
        query_ids = list(queries.keys())
    else:
        query_ids = queries
    assert len(query_ids) == len(scores)
    res = {}

    with open(where_to_save, 'w') as f:
        # header: qid Q0 docid rank score run_name
        # f.write("qid Q0 docid rank score run_name\n")
        for i, ind_query in enumerate(query_ids):
            res[ind_query] = {}
            rank = 0
            for score, indice in zip(scores[i], indices[i]):
                doc_name = doc_ids[indice]
                res[ind_query][doc_name] = float(score)
                # qid Q0 docid rank score run_name
                line = f"{ind_query} Q0 {doc_name} {rank} {float(score)} {run_id}"
                f.write(f"{line}\n")
                rank += 1

    return res


def transform_search_results(query_ids, doc_ids, scores, indices):
    """
    :return: results in a format of a dictionary,
             suitable for EvaluateRetrieval function from BEIR lib
    """
    assert len(query_ids) == len(scores)
    res = {}

    for i, ind_query in enumerate(query_ids):
        res[ind_query] = {}
        for score, indice in zip(scores[i], indices[i]):
            doc_name = doc_ids[indice]
            res[ind_query][doc_name] = float(score)
    return res


def load_search_results(log_dir, dataset_name, user_id="", model_name=None, special_id=None):
    """

    :return: A dictionary containing search results in a format suitable for EvaluateRetrieval function from BEIR lib
    """
    log_dir = os.path.join(log_dir, user_id)
    if model_name is None:
        file_name = f"{special_id}.txt"
    else:
        file_name = "sch_{}_{}.txt".format(dataset_name, model_name)

    path_to_file = os.path.join(log_dir, file_name)
    results = {}
    with open(path_to_file, 'r') as f:
        for line in f:
            # Assuming the format of each line is: qid Q0 docid rank score run_name

            # dset none model_name model_rank evalustion_score(rbo/ndcg) test
            parts = line.strip().split()
            qid, _, docid, rank, score, run_name = parts
            if qid not in results:
                results[qid] = {}
            results[qid][docid] = float(score)

    return results


def load_eval_results(log_dir, dataset_name, model_name):
    eval_log_dir = os.path.join(log_dir, "eval_results", dataset_name)
    eval_name = "eval_{}_{}.txt".format(dataset_name, model_name)
    with open(os.path.join(eval_log_dir, eval_name), "r") as f:
        eval_results = json.load(f)
        ndsg10 = eval_results[0]['NDCG@10']
    return ndsg10


def save_result_to_pickle(result, result_dir, dataset_name, model_name, experiment_index, user_id=""):
    # Save results
    log_dir = os.path.join(result_dir, user_id, dataset_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # bentropy
    name = "{}_{}_{}.pkl".format(experiment_index, dataset_name, model_name)
    path_to_file = os.path.join(log_dir, name)
    with open(path_to_file, 'wb') as f:
        pickle.dump(result, f)
    return True


def read_queries(path, file_id):
    with open(os.path.join(path, f"{file_id}.queries.tsv"), "r") as f:
        qids, queries = [], []
        for line in f:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)
    return qids, queries


def read_qrels(path, file_id):
    with open(os.path.join(path, f"{file_id}.qrels"), "r") as f:
        qrels = {}
        for line in f:
            qid, _, doc, score = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc] = int(score)
    return qrels

# def load_search_results(log_dir, dataset_name, model_name):
#     log_dir = os.path.join(log_dir, dataset_name)
#     file_name = "sch_{}_{}.pkl".format(dataset_name, model_name)
#     path_to_file = os.path.join(log_dir, file_name)
#     with open(path_to_file, 'rb+') as f:
#         search_results = pickle.load(f)
#     return search_results



