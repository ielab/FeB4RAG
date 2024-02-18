import argparse

import os
import json

import matplotlib.pyplot as plt

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

def plot_kappa_values_increasing(kappa_dict, kappa):
    # Sorting the dictionary by its values (kappa values)
    sorted_kappa_dict = dict(sorted(kappa_dict.items(), key=lambda item: item[1]))

    # Extracting sorted collection names and kappa values
    collections = list(sorted_kappa_dict.keys())
    collections = [collection_dict[collection] for collection in collections]
    kappa_values = list(sorted_kappa_dict.values())

    # Creating the bar plot
    plt.figure(figsize=(14, 6))
    plt.bar(collections, kappa_values,  color='black')

    plt.ylabel('Kappa', fontsize=22)
    plt.ylim(0, 1.1)
    #plt.title('Kappa Values bewteen two LLM-based judges', fontsize=22)
    plt.axhline(kappa, color='red', linestyle='-.')
    plt.text(0, kappa + 0.06, f'Kappa={kappa:.2f}', fontsize=20, color='red')
    plt.yticks(fontsize=17)
    plt.xticks(rotation=60, fontsize=17)   # Rotating the x-axis labels for better readability
    #plt.grid(True)  # Rotating the x-axis labels for better readability
    plt.tight_layout()
    plt.savefig('kappa_values_between_models.pdf')


def read_mapping(mapping_file):
    # format: qid dataset docid
    mapping_dataset = {}
    mapping_id = {}

    with open(mapping_file, 'r') as f:
        for line in f:
            qid, dataset, docid = line.strip().split()
            mapping_dataset[qid] = dataset
            mapping_id[qid] = docid
    return mapping_dataset, mapping_id
def convert_rel(rel):
    if int(rel)> 0:
        return "r"
    else:
        return "nr"

def convert_rel_string(rel):
    if rel =="r":
        return "r"
    elif rel == "nr":
        return "nr"
    elif rel == "hr":
        return "r"
    elif rel == "key":
        return "r"
    else:
        return "nr"


def calculate_kappa(list1, list2):
    po = sum([1 for i in range(len(list1)) if list1[i] == list2[i]]) / len(list1)
    pe = sum([1 for i in range(len(list1)) if list1[i] == "r"]) / len(list1) * sum([1 for i in range(len(list1)) if list2[i] == "r"]) / len(list1)

    # Handling the special case where po and pe are both 1
    if po == 1 and pe == 1:
        return 1

    # Handling the case where po and pe are very close to avoid indeterminate 0/0
    if abs(po - pe) < 1e-10:
        return 1

    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
    return kappa



def main():
    args = argparse.ArgumentParser(description='Calculate agreement between two judges')
    args.add_argument('--label_folder', type=str, required=True, help='Path to the judgement file')
    args = args.parse_args()
    label_folder = args.label_folder

    sub_collections = ["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever", "nfcorpus", "fiqa", "trec-covid", "nq", "hotpotqa", "robust04", "msmarco", "webis-touche2020"]
    #sub_collections =["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever"]

    overall_label_list = []
    overall_label_2_list = []
    kappaca_dict = {}
    overall_count = 0

    for sub_collection in sub_collections:
        label_list = []
        label_2_list = []
        label_file = os.path.join(label_folder, sub_collection, "solar-11b.jsonl")
        label_file_2 = os.path.join(label_folder, sub_collection, "lgs-13b.jsonl")
        #only assess qids that mapping dataset is the same as sub_collection
        if not os.path.exists(label_file):
            continue
        if not os.path.exists(label_file_2):
            continue

        label_1_dict = {}
        with open(label_file, 'r') as f:
            for line in f:
                label = json.loads(line)
                qid = label["qid"]
                doc_id = label["docid"]
                label_1_dict[qid+ " " + doc_id] = label["label"]

        label_2_dict = {}
        with open(label_file_2, 'r') as f:
            for line in f:
                label = json.loads(line)
                qid = label["qid"]
                doc_id = label["docid"]
                label_2_dict[qid+ " " + doc_id] = label["label"]
        for qid_doc_id in label_1_dict:
            if qid_doc_id in label_2_dict:
                actual_label = convert_rel_string(label_1_dict[qid_doc_id])
                actual_label_2 = convert_rel_string(label_2_dict[qid_doc_id])
                #label_list.append(actual_label)
                #label_2_list.append(actual_label_2)
                label_list.append(label_1_dict[qid_doc_id])
                label_2_list.append(label_2_dict[qid_doc_id])
                if actual_label != actual_label_2:
                    overall_count += 1
                # if label_1_dict[qid_doc_id] != label_2_dict[qid_doc_id]:
                #     overall_count += 1


        if len(label_list) == 0:
            continue
        print(len(label_list))
        #calculate cohen's kappa
        kappa = calculate_kappa(label_list, label_2_list)
        print(f"{sub_collection} {kappa}")
        overall_label_list += label_list
        overall_label_2_list += label_2_list
        kappaca_dict[sub_collection] = kappa
    print("Overall")
    kappa = calculate_kappa(overall_label_list, overall_label_2_list)
    print(f"{kappa}")

    #next, draw a plot, showing increasingly the kappa value, then x axis is the collection name
    plot_kappa_values_increasing(kappaca_dict, kappa)
    print(overall_count)
    print(overall_count/len(overall_label_list))




if __name__ == "__main__":
    main()