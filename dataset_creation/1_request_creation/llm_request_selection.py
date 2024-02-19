import json
import os
from tqdm import tqdm
import argparse
datasets = ["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever"]

args = argparse.ArgumentParser()
args.add_argument("--original_dataset_folder", type=str)
args = args.parse_args()
dataset_folder=args.original_dataset_folder
for dataset in datasets:
    print(dataset)
    input_file = os.path.join(dataset_folder, dataset, "conversational_queries.jsonl")
    output_file = os.path.join(dataset_folder, dataset, "conversational_queries_final.jsonl")
    qid_parsed = set()

    if os.path.exists(output_file):
        for line in open(output_file):
            query = json.loads(line)
            query_id = query["_id"]
            qid_parsed.add(query_id)
    with open(output_file, "a") as fw:
        with open(input_file) as f:
            for line in tqdm(f):
                query = json.loads(line)
                query_id = query["_id"]
                if query_id in qid_parsed:
                    continue
                query_texts = query["conversational_queries"]
                print(f"Queries generated for {query_id}")
                for q_rank, query_text in enumerate(query_texts):
                    #how to the user
                    print(f"{q_rank+1}: {query_text}")

                selection = input("Which query do you want to select? (1, 2, 3)\n")
                if selection == "1":
                    query_text = query_texts[0]
                elif selection == "2":
                    query_text = query_texts[1]
                elif selection == "3":
                    query_text = query_texts[2]
                else:
                    print("Invalid selection")
                    continue
                new_dict = {"_id": query_id, "text": query_text, "metadata": {}}
                fw.write(json.dumps(new_dict) + "\n")






