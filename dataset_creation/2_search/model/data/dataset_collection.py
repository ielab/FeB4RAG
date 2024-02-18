
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader


class Datasets:
    def __init__(self, data_dir="/opt/data/IR_datasets/"):
        self.data_dir = data_dir

    def download_one(self, dataset_name):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
        dataset_dir = os.path.join(self.data_dir, dataset_name)
        util.download_and_unzip(url, dataset_dir)
        return True

    def load_dataset(self, dataset_name, user_id="", load_corpus=True, split="test"):
        data_path = os.path.join(self.data_dir, user_id, dataset_name)
        dataloader = GenericDataLoader(data_folder=data_path)
        # Change this bit if you want to disable the loading of the large corpus to a memory
        if not load_corpus:
            dataloader.corpus = ["blabla"]
        corpus, queries, qrels = dataloader.load(split=split)
        return corpus, queries, qrels