import argparse
import glob
import math
import os
import json
import matplotlib.pyplot as plt
import numpy as np
aggregate_dict = {

        "nr": 0,
        "r": 0.25,
        "hr": 0.5,
        "key": 1,
}
predefined_num = 10
collections = ["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever", "nfcorpus", "fiqa", "trec-covid", "nq", "hotpotqa", "robust04", "msmarco", "webis-touche2020"]


collection_dict = {
    "msmarco": "MS MARCO",
    "trec-covid": "TREC-COVID",
    "nfcorpus": "NFCorpus",
    "scidocs": "SCIDOCS",
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "fiqa": "FIQA-2018",
    "signal1m": "Signal-1M",
    "trec-news": "TREC-NEWS",
    "robust04": "Robust04",
    "arguana": "ArguAna",
    "webis-touche2020": "Touché-2020",
    "dbpedia-entity": "DBPedia ",
    "fever": "FEVER",
    "climate-fever": "Climate-FEVER",
    "scifact": "SciFact"
}
def main(input_folder, queries_file, llm, out_file):
    qid_set = set()



    resouce_selected_dict = {}
    with open(queries_file, 'r') as f:
        for line in f:
            current_dict = json.loads(line)
            qid = current_dict["_id"]
            qid_set.add(qid)
    final_aggregate_dict = {}
    for collection in collections:
        #collection is the collection level label, now need to aggregate the resource level label to get this
        input_file = os.path.join(input_folder, collection, llm + ".jsonl")

        with open(input_file, 'r') as f:
            for line in f:
                current_dict = json.loads(line)
                qid = current_dict["qid"]
                docid = current_dict["docid"]
                if qid in qid_set:
                    if qid not in final_aggregate_dict:
                        final_aggregate_dict[qid] = {}
                    if collection not in final_aggregate_dict[qid]:
                        final_aggregate_dict[qid][collection] = []

                    if qid not in resouce_selected_dict:
                        resouce_selected_dict[qid] = []
                    label = current_dict["label"]
                    if (label == "key") or (label=="hr") or (label=="r"):
                        if collection not in resouce_selected_dict[qid]:
                            resouce_selected_dict[qid].append(collection)


                    final_aggregate_dict[qid][collection].append(aggregate_dict[current_dict["label"]])
    statistics = {}
    with open(out_file, 'w') as f:
        for qid in final_aggregate_dict:
            current_dict = final_aggregate_dict[qid]
            collection_nums_dict = {}
            for collection in current_dict:
                collection_num = math.ceil((sum(current_dict[collection])/predefined_num)*100)
                collection_nums_dict[collection] = collection_num

            #statistics would append number of non-zero collections
            statistics[qid]  = list(collection_nums_dict.values())

            #now sort from the collection_nums_dict
            sorted_collection_nums_dict = dict(sorted(collection_nums_dict.items(), key=lambda item: item[1], reverse=True))
            for collection, collection_num in sorted_collection_nums_dict.items():

                f.write(f"{qid} 0 {collection} {collection_num}\n")

    total_selected_collections = 0

    # Iterate through each query and count non-zero relevance labels
    for qid in statistics:
        total_selected_collections += sum(1 for rel in statistics[qid] if rel > 25)

    # Calculate the average number of collections selected per query
    average_collections_selected = total_selected_collections / len(statistics)
    print("Average Number of Collections Selected Per Query:", average_collections_selected)


    #also then need to print the average number, meaning the relevance label for each query, then average across all queries
    average_relevances = []

    # Iterate through each query and calculate its average relevance
    for qid in statistics:
        # Filter out zero values before calculating average
        non_zero_rels = [rel for rel in statistics[qid] if rel > 0]
        if non_zero_rels:  # Check if there are any non-zero values
            average_relevances.append(sum(non_zero_rels) / len(non_zero_rels))
        else:
            average_relevances.append(0)  # If all values are zero, the average is zero

    # Calculate overall average, median, min, and max across all queries
    overall_average_relevance = np.mean(average_relevances)
    median_relevance = np.median(average_relevances)
    min_relevance = min(average_relevances)
    max_relevance = max(average_relevances)


    #calculate for resource_selected, the average num
    #print the average number of resources selected

    print("Average Number of Resources Selected Per Query:", np.mean([len(res) for res in resouce_selected_dict.values()]))







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the judgement file')
    parser.add_argument('--request_file', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--llm', type=str, required=True, help='llm type')
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    args = parser.parse_args()
    queries_file = args.queries_file
    qrrel_file = args.qrel_file
    input_folder = args.input_folder
    out_file = args.out_file
    llm = args.llm

    main(input_folder, queries_file, llm, out_file)


