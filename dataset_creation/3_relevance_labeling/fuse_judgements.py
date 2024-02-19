#fuse the labeling file
import argparse
import json
import os
from tqdm import tqdm


def convert_label_to_score(label):
    if label == "nr":
        return 0
    elif label == "r":
        return 1
    elif label == "hr":
        return 2
    elif label == "key":
        return 3
    else:
        return 0

def convert_score_to_label(score):
    if score == 0:
        return "nr"
    elif score == 1:
        return "r"
    elif score == 2:
        return "hr"
    elif score == 3:
        return "key"
    else:
        return "nr"


collections =["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever", "nfcorpus", "fiqa", "trec-covid", "nq", "hotpotqa", "robust04", "msmarco", "webis-touche2020"]
def main():
    parser = argparse.ArgumentParser(description='Fuse the labeling file')
    parser.add_argument('--labeling_folder', type=str, required=True, help='Path to the relevance label file 1')
    #llms, it could be multiple
    parser.add_argument('--llms', type=str, nargs='+', required=True, help='llms')
    args = parser.parse_args()
    labeling_folder = args.labeling_folder
    llms = args.llms
    #output_type = args.output_type

    for collection in tqdm(collections):
        label_dict = {}
        for llm in llms:
            label_file = os.path.join(labeling_folder, collection, llm + ".jsonl")
            with open(label_file, 'r') as f:
                for line in f:
                    label = json.loads(line)
                    qid = label["qid"]
                    docid = label["docid"]
                    label_conv = label["label"]
                    if qid not in label_dict:
                        label_dict[qid] = {}
                    #null case
                    if label is None:
                        continue
                    if docid not in label_dict[qid]:
                        label_dict[qid][docid] = []
                    label_dict[qid][docid].append(label_conv)
        with open(os.path.join(labeling_folder, collection, "fused-" + "-".join(llms) + ".jsonl"), 'w') as f:
            for qid in label_dict:
                for docid in label_dict[qid]:
                    label_list = label_dict[qid][docid]
                    label_average = sum([convert_label_to_score(label) for label in label_list])/len(label_list)
                    #take the closest from predefined list
                    label_average = min([0, 1, 2, 3], key=lambda x: abs(x - label_average))
                    #print(label_average)

                    f.write(json.dumps({"qid": qid, "docid": docid, "label": convert_score_to_label(label_average)}) + "\n")


if __name__ == "__main__":
    main()


