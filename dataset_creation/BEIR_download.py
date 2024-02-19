from beir import util
import os


def download_one(data_dir, dataset_name):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    dataset_dir = os.path.join(data_dir, dataset_name)
    util.download_and_unzip(url, dataset_dir)

dataset_names = ["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever", "nfcorpus", "fiqa", "trec-covid", "nq", "hotpotqa", "robust04", "msmarco", "webis-touche2020"]
data_dir = "original_dataset"

for dataset_name in dataset_names:
    download_one(data_dir, dataset_name)
