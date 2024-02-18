import argparse
import os
import json

def convert_rel(rel):
    if rel == "nr":
        return 0
    elif rel == "r":
        return 1
    elif rel == "hr":
        return 2
    elif rel == "key":
        return 3



def main(input_folder, llm, out_file):
    sub_collections = ["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever", "nfcorpus", "fiqa", "trec-covid", "nq", "hotpotqa", "robust04", "msmarco", "webis-touche2020"]

    with open(out_file, "w") as fw:
        for sub_collection in sub_collections:
            input_file = os.path.join(input_folder, sub_collection, llm+".jsonl")
            with open(input_file, "r") as f:
                for line in f:
                    label = json.loads(line)
                    qid = label["qid"]
                    docid = label["docid"]
                    rel = label["label"]
                    fw.write(f"{qid} Q0 {docid} {convert_rel(rel)}\n")







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the judgement file')
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    parser.add_argument('--llm', type=str, required=True, help='llm_name')
    args = parser.parse_args()

    input_folder = args.input_folder
    out_file = args.out_file

    llm = args.llm
    main(input_folder, llm, out_file)