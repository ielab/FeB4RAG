from typing import Dict, List

import torch
import os
import logging
from collections import defaultdict
import numpy as np

from beir.retrieval.evaluation import EvaluateRetrieval


class CustomDEModel:
    def __init__(self, **kwargs):

        # Re-init this in the constructor
        self.query_encoder = None
        self.doc_encoder = None
        self.tokenizer = None
        self.config = None

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        pass

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        pass


class ModelClass:
    def __init__(self, model_dir):
        """
        The super-class for models to test.
        Each Model Class might consist of several models, that come from the same category
        For example, if the models have the same way of encoding the queries and the documents


        names - the list of names of this type of models.
                it can be, for example, different versions of the same model, e.g.
                ["msmarco-distilbert-base-v2", "msmarco-distilbert-base-v3"]

        score_function - dictionary, that defines the type of the scoring function for each model.
                right now, we support "dot" and "cos_sim"
                Example:  self.score_function = [{"msmarco-distilbert-base-v2":"dot"},
                                                 {"msmarco-distilbert-base-v3":"cos_sim"}]

        source_datasets - the dataset each model was trained on. For most of the datasets, it will be msmarco
                Example: self.source_datasets = [{"msmarco-distilbert-base-v2":"msmarco"},
                                                 {"msmarco-distilbert-base-v3":"msmarco"}]

        :param model_dir: where to store the model
        """

        # The next lines should be inherited from this class
        self.model_dir = model_dir
        self.metrics = defaultdict(list)  # store final results

        # The following variables need to be re-initialized in the constructor of each model class
        self.names = []
        self.score_function = {}
        self.source_datasets = {}

    def download_models(self):
        """
        Downloads all the models and stores them in self.model_dir

        :return: True if the download is successful
        """

        raise "Must be implemented by a subclass"

    def load_model(self, name, **kwargs) -> CustomDEModel:
        """
        Loads the model by its name.


        :param name: the name of the model
        :return: the model, that has two main functions:
            encode_queries and encode_corpus
        """

        assert name in self.names
        model = CustomDEModel()

        return model

    def model_eval(self, model, corpus, queries, qrels,
                   score_function="dot", model_name=None):

        retriever = EvaluateRetrieval(model, score_function=score_function)

        #### Retrieve dense results (format of results is identical to qrels)
        logging.info("Prior to model retrieval")

        # You want to replace this!
        # Takes a very long time. Encodes the documents + retrieves them
        results = retriever.retrieve(corpus, queries)
        #print("RESULTS", results)

        #### Evaluate your retrieval using NDCG@k, MAP@K ...

        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))

        # This method is static
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        #### Print top-k documents retrieved ####
        top_k = 10
        self.metrics["model_name"].append(model_name)
        for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
            if isinstance(metric, str):
                metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
            for key, value in metric.items():
                self.metrics[key].append(value)
        return

    def save_metrics(self, log_file="./main_logs/metrics.pt"):
        if os.path.isfile(log_file):
            existing = torch.load(log_file)
            for i, m_name in enumerate(self.metrics["model_name"]):
                if m_name not in existing['model_name']:
                    # Add all the stats for this model to the result file
                    for key in self.metrics.keys():
                        existing[key].append(self.metrics[key][i])
                else:
                    # Replace with the new value
                    indx = existing['model_name'].index(m_name)
                    for key in self.metrics.keys():
                        existing[key][indx] = self.metrics[key][i]
            self.metrics = existing
        torch.save(self.metrics, log_file)



