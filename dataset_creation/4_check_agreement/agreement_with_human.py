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
def fuse_judgements(j1, j2):
    # The fused judgement is the majority vote of the individual judgements
    convert_dict = {"key": 3, "hr": 2, "r": 1, "nr": 0}
    if j1 is None:
        return convert_rel_string(j2)
    if j2 is None:
        return convert_rel_string(j1)

    j1_num = convert_dict[j1]
    j2_num = convert_dict[j2]
    #then pick the one with the average value
    #print(j1_num, j2_num)
    #in case of none value:
    alpa = 0.5
    j_avg = j1_num*alpa + j2_num*(1-alpa)

    #note that

    if j_avg >= 1:
        return "r"
    else:
        return "nr"

def plot_kappa_values_increasing(kappa_dict, name, kappa):
    # Sorting the dictionary by its values (kappa values)
    sorted_kappa_dict = dict(sorted(kappa_dict.items(), key=lambda item: item[1]))

    # Extracting sorted collection names and kappa values
    collections = list(sorted_kappa_dict.keys())
    collections = [collection_dict[collection] for collection in collections]
    kappa_values = list(sorted_kappa_dict.values())

    # Creating the bar plot
    plt.figure(figsize=(14, 6))
    plt.bar(collections, kappa_values, color='black')
    #plt.xlabel('Collection Name', fontsize=15 )
    plt.ylabel('Kappa', fontsize=22)
    #y axis 0 to 1
    plt.ylim(0, 1.1)
    #plt.title(f'Kappa Values by Collection from qrel to {name}', fontsize=22)
    # draw a dotted horizontal line at y=kappa
    plt.axhline(kappa, color='red', linestyle='-.')
    #next to it, write the kappa value
    plt.text(0, kappa+0.06, f'Kappa={kappa:.2f}', fontsize=20, color='red')
    plt.xticks(rotation=60, fontsize=17)  # Rotating the x-axis labels for better readability
    plt.yticks(fontsize=17)
    #plt.grid(True)  # Rotating the x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(f'{name}_kappa_values_increasing.pdf')


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

negative_unjudged = 0
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


def read_qrel_file(qrel_file):
    qrel_dict = {}
    with open(qrel_file, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid not in qrel_dict:
                qrel_dict[qid] = {}
            qrel_dict[qid][docid] = convert_rel(rel)
    return qrel_dict

def main():
    args = argparse.ArgumentParser(description='Calculate agreement between two judges')
    args.add_argument('--label_folder', type=str, required=True, help='Path to the judgement file')

    args.add_argument('--qrel_file', type=str, required=True, help='qrel file')
    args.add_argument('--mapping_file', type=str, required=True, help='output file')
    args.add_argument('--llm', type=str, required=True, help='llm_name')
    args.add_argument('--llm_2', type=str, default=None, help='llm_name')

    args = args.parse_args()
    label_folder = args.label_folder
    qrel_file = args.qrel_file
    mapping_file = args.mapping_file

    sub_collections = ["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever", "nfcorpus", "fiqa", "trec-covid", "nq", "hotpotqa", "robust04", "msmarco", "webis-touche2020"]
    #sub_collections =["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever"]
    mapping_dataset, mapping_id = read_mapping(mapping_file)
    qrel_dict = read_qrel_file(qrel_file)
    overall_label_list = []
    overall_qrel_list = []
    kappaca_dict = {}

    for sub_collection in sub_collections:
        label_list = []
        qrel_list = []
        label_file = os.path.join(label_folder, sub_collection, args.llm+".jsonl")

        label_2_dict ={}
        if not os.path.exists(label_file):
            continue
        if args.llm_2 is not None:
            label_file_2 = os.path.join(label_folder, sub_collection, args.llm_2 + ".jsonl")
            with open(label_file_2, 'r') as f:
                for line in f:
                    label = json.loads(line)
                    qid = label["qid"]
                    docid = label["docid"]
                    label_conv = label["label"]
                    label_2_dict[qid+ " " + docid] = label_conv

        #only assess qids that mapping dataset is the same as sub_collection
        for line in open(label_file):
            label = json.loads(line)
            qid = label["qid"]
            docid = label["docid"]
            if qid in mapping_dataset and mapping_dataset[qid] == sub_collection:
                label_conv = convert_rel_string(label["label"])
                current_tuple = qid + " " + docid
                if args.llm_2 is not None:
                    if current_tuple in label_2_dict:
                        label_conv_2 = convert_rel_string(label_2_dict[current_tuple])
                        # if label_conv == label_conv_2:
                        #     label_list.append(label_conv)
                        #     if qid in qrel_dict:
                        #         if docid in qrel_dict[qid]:
                        #             #print(f"{qid} {docid} {label_conv} {qrel_dict[qid][docid]}")
                        #             qrel_list.append(qrel_dict[qid][docid])
                        #         else:
                        #             qrel_list.append("nr")
                        # else:
                        label_list.append(fuse_judgements(label["label"], label_2_dict[current_tuple]))
                        if qid in qrel_dict:
                            if docid in qrel_dict[qid]:
                                #print(f"{qid} {docid} {label_conv} {qrel_dict[qid][docid]}")
                                qrel_list.append(qrel_dict[qid][docid])
                            else:
                                qrel_list.append("nr")

                else:
                    if qid in qrel_dict:
                        if docid in qrel_dict[qid]:
                            label_list.append(label_conv)
                            #print(f"{qid} {docid} {label_conv} {qrel_dict[qid][docid]}")
                            qrel_list.append(qrel_dict[qid][docid])
                        #else:
                            #qrel_list.append("nr")

        if len(label_list) == 0:
            continue
        #print(label_list)
        #print(qrel_list)
        #calculate cohen's kappa
        kappa = calculate_kappa(label_list, qrel_list)
        print(f"{sub_collection}: has {len(label_list)} judgement with kappa value: {kappa}")
        overall_label_list += label_list
        overall_qrel_list += qrel_list
        kappaca_dict[sub_collection] = kappa

    kappa = calculate_kappa(overall_label_list, overall_qrel_list)
    print(f"Overall: has {len(overall_label_list)} judgement with kappa value: {kappa}")

    #next, draw a plot, showing increasingly the kappa value, then x axis is the collection name
    if args.llm_2 is not None:
        plot_kappa_values_increasing(kappaca_dict, f"{args.llm}_{args.llm_2}", kappa)
    else:
        plot_kappa_values_increasing(kappaca_dict, f"{args.llm}", kappa)





if __name__ == "__main__":
    main()




