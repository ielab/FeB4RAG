
import os

from transformers import AutoConfig, AutoTokenizer

from sentence_transformers import SentenceTransformer
from beir.retrieval import models

from model.CustomModels import AnglEModel, E5Model, SentenceTransformerSpec, InstructorModel
from model.model_collection import ModelClass


class CustomModel(ModelClass):
    def __init__(self, model_dir="/opt/data/IR_models/", old_models_only=False,
                 special_token=False, specific_model=None):
        super().__init__(model_dir)

        if specific_model is not None:
            self.names = [specific_model]
        else:
            self.names = [
                "SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                "e5-large", "e5-base", "UAE-Large-V1",
                "instructor-xl", "multilingual-e5-large"]

        self.score_function = {
            "SGPT-5.8B-weightedmean-msmarco-specb-bitfit": "cos_sim",
            "e5-large": "cos_sim",
            "e5-base": "cos_sim",
            "UAE-Large-V1": "cos_sim",
            "instructor-xl": "cos_sim",
            "multilingual-e5-large": "cos_sim",
        }

    def load_model(self, name, cuda=True, model_name_or_path=None):
        assert name in self.names

        if name == "UAE-Large-V1":
            model = AnglEModel("WhereIsAI/UAE-Large-V1", cache_dir=self.model_dir, cuda=cuda)
        elif "e5-" in name:
            # "e5-base", "e5-small", "e5-large-v2"
            # "e5-small-v2", "e5-base-v2", "e5-large-v2"
            # "multilingual-e5-small", "multilingual-e5-base", "multilingual-e5-large"
            model = E5Model("intfloat/" + name, cache_dir=self.model_dir, cuda=cuda)
        elif "weightedmean-msmarco-specb-bitfit" in name:
            model = SentenceTransformerSpec(f"Muennighoff/{name}",
                                            cache_dir=self.model_dir, cuda=cuda)
        elif "instructor-" in name:
            model = InstructorModel(f"hkunlp/{name}", cache_dir=self.model_dir, cuda=cuda)
        else:
            raise "Unknown model name"
        return model


class BeirModels(ModelClass):
    def __init__(self, model_dir, old_models_only=False, special_token=False,
                 specific_model=None):
        super().__init__(model_dir)

        if specific_model is not None:
            self.names = [specific_model]
        else:
            self.names = [
                "all-mpnet-base-v2", "ember-v1", "gte-large", "gte-base" ]

        model_name_or_path = []
        for name in self.names:
            if "gte-" in name:
                model_name_or_path.append(f"thenlper/{name}")
            elif name == "ember-v1":
                model_name_or_path.append(f"llmrails/{name}")
            else:
                model_name_or_path.append(f"sentence-transformers/{name}")
        self.model_name_or_path = model_name_or_path
        score_function = {}

        score_function["all-mpnet-base-v2"] = "dot"
        names_with_cos = ["ember-v1", "gte-large", "gte-base"]
        for name in names_with_cos:
            score_function[name] = "cos_sim"
        self.score_function = score_function

    def download_models(self):

        for name in self.model_name_or_path:
            # print("Downloading", name)
            SentenceTransformer(model_name_or_path=name,
                                cache_folder=self.model_dir)

            # print("Finished loading")

    def load_model(self,  model_name, cuda=True, model_name_or_path=None):

        # find model_name in self.model_name_or_path
        for name in self.model_name_or_path:
            if model_name in name:
                model_name = name
                break

        # replace "/" with "_"
        model_name = model_name.replace("/", "_")
        model_dir = os.path.join(self.model_dir, model_name)
        model = models.SentenceBERT(model_dir)

        model.config = AutoConfig.from_pretrained(model_dir)
        model.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if cuda:
            model.q_model = model.q_model.cuda()
            model.doc_model = model.doc_model.cuda()
        return model


