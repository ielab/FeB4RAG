
from utils import read_and_write
import pickle, json

from data.dataset_collection import Datasets
from beir.retrieval.evaluation import EvaluateRetrieval
from model.model_zoo import CustomModel, BeirModels
import numpy as np
import faiss
from utils.read_and_write import read_doc_enc_from_pickle, save_search_results, load_search_results
import gc, os
from utils.get_args import get_args
import torch



def tokenize_and_save(args, models, model_names,
                      corpus):

    """
    Encodes the corpus and saves the encoding into a file
    :param args: arguments from the input
    :param beir_models: BEIR MODELs class
    :param corpus: corpus
    :return: name where to save the file
    """

    # Sort the documents by its size
    corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
                        reverse=True)
    corpus = [corpus[cid] for cid in corpus_ids]
    itr = range(0, len(corpus), args.corpus_chunk_size)

    for model_name in model_names:
        print("Model name", model_name)

        model = models.load_model(model_name, model_name_or_path=None, cuda=True) # args.model_dir)

        #  Encoding
        for batch_num, corpus_start_idx in enumerate(itr):
            corpus_end_idx = min(corpus_start_idx + args.corpus_chunk_size, len(corpus))
            # Returns numpy arrays
            if "instructor" in model_name:
                sub_corpus_embeddings = model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx], batch_size=args.batch_size,
                    convert_to_tensor=False, dataset_name=args.dataset_name)
            else:
                sub_corpus_embeddings = model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],  batch_size=args.batch_size,
                    convert_to_tensor=False)
            # Save results in a file
            read_and_write.save_enc_to_pickle(sub_corpus_embeddings, corpus_ids[corpus_start_idx:corpus_end_idx],
                                              dataset_name=args.dataset_name,  #args.dataset_name+"_train",
                                              model_name=model_name,
                                              log_dir=args.embedding_dir,
                                              batch_num=batch_num,
                                              user_id=args.user_id)

            print("Saved batch", batch_num, "of", len(itr), "batches")
    return True


def search(query_embeddings, doc_embeddings, top_k=1000, score_function="dot"):
    """
    Extracts top_k documents based on the queries.
    Saves the scores and the ids of the extracted documents in a file

    Implemented with faiss library

    :param query_embeddings:
    :param doc_embeddings:
    :param top_k: How many docs to extract per query
    :param score_function: "dot" or "cos_sim"retrieve_and_eval.py
    :return: Scores and Associated document indices
    """
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    # If normalized - becomes Cosine similarity
    if score_function == "cos_sim":
        faiss.normalize_L2(doc_embeddings)
        faiss.normalize_L2(query_embeddings)

    elif score_function != "dot":
        raise "Unknown score function"

    # Otherwise - Dot product
    index.add(doc_embeddings)
    # To save the index -> faiss.write_index()
    # Search for query embeddings
    scores, indices = index.search(query_embeddings, top_k)
    return scores, indices


def save_eval_results(qrels, results, where_to_save):
    eval_retrieval = EvaluateRetrieval()
    # Can add 1000 here if needed
    eval_results = eval_retrieval.evaluate(qrels=qrels, results=results, k_values=[1, 3, 5, 10, 100, 1000])
    print(eval_results)

    with open(where_to_save, 'w') as f:
        json.dump(eval_results, f)
    return eval_results


def run_evaluation(args, models, names):
    # for model_name in model_names:
    if torch.cuda.is_available():
        cuda = True
        print("Cuda is available")
    else:
        cuda = False
    for model_name in names:

        model = models.load_model(model_name, model_name_or_path=None, cuda=cuda)
        dataset = Datasets(args.dataset_dir)
        if args.dataset_name == "msmarco":
            # query_dname = "msmarco"
            query_dname = "msmarco"
            split = "dev"
        else:
            query_dname = args.dataset_name
            split = "test"

        if args.fake_queries:
            # read from qrels
            queries, qrels = {}, {}
            # BRING THIS BACK
            #data_temp = f"{args.fake_data_dir}/{args.dataset_name}/"
            data_temp = f"{args.fake_data_dir}/fedbeir/"
            #with open(os.path.join(data_temp, "{}-{}.qrels".format(args.dataset_name, args.fake_id_qrels)), 'r') as f:
            with open(os.path.join(data_temp, "fedbeir-1.qrels"), 'r') as f:
                for line in f:
                    qid, _, doc_id, _ = line.split()
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][doc_id] = 1
            #with open(os.path.join(data_temp, f"{args.dataset_name}-{args.fake_id_queries}.queries.tsv"), 'r') as f:
            with open(os.path.join(data_temp, "fedbeir-1.queries.tsv"), 'r') as f:
                for line in f:
                    qid, query_text = line.split("\t")
                    queries[qid] = query_text
        else:

            _, queries, qrels = dataset.load_dataset(query_dname,  # args.dataset_name,
                                                     load_corpus=False,
                                                     split=split,
                                                     user_id=args.user_id)
        # # # Get document embeddings
        doc_embeds, doc_ids = read_doc_enc_from_pickle(args.dataset_name, model_name, args.embedding_dir,
                                                       user_id=args.user_id)
        # read_doc_enc_from_pickle(args.dataset_name, model_name, args.embedding_dir)

        # Get query embeddings
        query_list = [queries[qid] for qid in queries]

        # if fusion => load the search result as it is the same as q10
        if "fusion" in args.fake_id_qrels:
            results = load_search_results(args.log_dir+"/search_results_fake/",
                                          args.dataset_name,
                                          special_id="sch_{}_{}_{}".format(args.dataset_name,
                                                                           model_name, args.fake_id_queries))

            pass
        else:
            if "instructor" in model_name:
                query_embeds = model.encode_queries(query_list,  batch_size=args.batch_size,
                                                    show_progress_bar=True,
                                                    convert_to_tensor=False, dataset_name=args.dataset_name)
            else:
                query_embeds = model.encode_queries(query_list,  batch_size=args.batch_size,
                                                    show_progress_bar=True,
                                                    convert_to_tensor=False)

            scores, indices = search(query_embeds, doc_embeds, top_k=1000,
                                     score_function=models.score_function[model_name])
            # ================================== Save results  ====================================
            # Example: ./log_dir/nfcorpus/sch_nfcorpus_contriever.pkl

            if args.fake_queries:
                log_dir = os.path.join(args.log_dir, "search_results_fake")
                name = "sch_{}_{}_{}.txt".format(args.dataset_name, model_name, args.fake_id_qrels)
            else:
                log_dir = os.path.join(args.log_dir, "search_results")
                name = "sch_{}_{}.txt".format(args.dataset_name, model_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            path_to_file = os.path.join(log_dir, name)
            # print(f"Saving search results to {path_to_file}")
            results = save_search_results(queries, doc_ids, scores, indices, where_to_save=path_to_file)
    return results


def run_encoding(args,  models, names):

    dataset = Datasets(args.dataset_dir)
    split = "test"
    corpus, queries, qrels = dataset.load_dataset(args.dataset_name,
                                                  user_id=args.user_id,
                                                  split=split,
                                                  load_corpus=True)
    if args.little_corpus:
        corp2_keys = ['MED-10', 'MED-14', 'MED-118', 'MED-301', 'MED-306',
                      'MED-329', 'MED-330', 'MED-332', 'MED-334', 'MED-335',
                  'MED-398', 'MED-557', 'MED-666', 'MED-691', 'MED-692', 'MED-1130']
        corpus_2 = {key: corpus[key] for key in corp2_keys}
        corpus = corpus_2
    tokenize_and_save(args, models, names, corpus)


def run_encoding_or_eval():
    args = get_args()

    if args.task == "eval":
        # fix path for demo: /log_dir/user_id/dataset_name
        log_dir = args.log_dir
        log_dir = os.path.join(log_dir, args.user_id, args.dataset_name)
        os.makedirs(log_dir, exist_ok=True)
        args.log_dir = log_dir


    if args.old_models_only:
        MODELS = [BeirModels(args.model_dir, old_models_only=True),
                  CustomModel(model_dir=args.model_dir, old_models_only=True)]
    elif args.special_token:
        MODELS = [BeirModels(args.model_dir, special_token=True),
                  CustomModel(model_dir=args.model_dir, special_token=True)]
        MODELS[0].download_models()
    # else:
    elif args.model_type == "beir":
        MODELS = [BeirModels(args.model_dir, special_token=args.special_token,
                             specific_model=args.specific_model)]
    else:
        MODELS = [CustomModel(model_dir=args.model_dir,
                              special_token=args.special_token,
                              specific_model=args.specific_model)]
    # MODELS = [BeirModels(args.model_dir, specific_model="gte-tiny")]
    for models in MODELS:
        # models.download_models()
        for model_name in models.names:
            args.model_name = model_name
            print("Start with model", model_name)
            if args.task == "encode":
                if not torch.cuda.is_available():
                    raise "Cuda is required for encoding"
                run_encoding(args, models, [args.model_name])
            elif args.task == "eval":
                run_evaluation(args, models, [args.model_name])
            else:
                raise "Unknown task"


if __name__ == "__main__":
    run_encoding_or_eval()
