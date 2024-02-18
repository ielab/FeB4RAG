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
def main(input_folder, queries_file, llm, out_file, qrel_file):
    qid_set = set()
    qrel_dict = {}
    qrel_max_label_dict = {}
    with open(qrel_file, 'r') as f:
        for line in f:
            qid, _, docid, label = line.strip().split(" ")
            if qid not in qrel_dict:
                qrel_dict[qid] = []
                qrel_max_label_dict[qid] = int(label)
            if int(label) > qrel_max_label_dict[qid]:
                qrel_max_label_dict[qid] = int(label)
                qrel_dict[qid] = [docid]
            elif int(label) == qrel_max_label_dict[qid]:
                qrel_dict[qid].append(docid)

    #print(qrel_dict)
    qid_variability_plot = {}

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
                    elif docid in qrel_dict[qid]:
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

    #for statistics, need to print, average number of collections selected (means when they are non zero)
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

    ## fo a barplot on the number of collections selected per query
    # Draw a histogram of the number of collections selected per query
    # using resource_selected_dict
    plt.figure(figsize=(12, 6))
    plt.hist([len(res) for res in resouce_selected_dict.values()], bins=range(0, 17), align='left', rwidth=0.8, color='black')
    plt.xlabel('Number of search engines with relevant content', fontsize=26)
    plt.ylabel('Number of Queries', fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)




    plt.xticks(range(0, 17))
    plt.tight_layout()
    plt.savefig('collections_selected_histogram.pdf')


    # Next, draw a bar plot, on which x axis is the collection name, and y axis is the number of times the collection is selected
    # Draw a bar plot of the number of times each collection was  using resource_selected_dict, remember, that key is qid

    # selected per query
    plt.figure(figsize=(12, 6))
    plt.bar(collections, [sum(1 for res in resouce_selected_dict.values() if collection in res) for collection in collections], color='black')
    #plt.xlabel('Collections', fontsize=18)
    plt.ylabel('Number of Times Selected', fontsize=18)
    #plt.title('Number of Times Each Collection was Selected')
    plt.xticks(rotation=60)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    plt.savefig('collections_selected_barplot.pdf')



    #
    # #print for resource selected
    # #print the average number of resources selected
    # print("Average Number of Resources Selected Per Query:", np.mean([len(res) for res in resouce_selected_dict.values()]))
    # #print the max
    # print("Maximum Number of Resources Selected Per Query:", max([len(res) for res in resouce_selected_dict.values()]))
    # #print the min
    # print("Minimum Number of Resources Selected Per Query:", min([len(res) for res in resouce_selected_dict.values()]))
    # # print number of queirs that selected 0 resources
    # print("Number of Queries that Selected 0 Resources:", sum(1 for res in resouce_selected_dict.values() if len(res) == 0))
    #
    # #plot per query selected resources
    # # Draw a histogram of the number of resources selected per query
    # plt.figure(figsize=(12, 6))
    # plt.hist([len(res) for res in resouce_selected_dict.values()], bins=range(0, 17), align='left', rwidth=0.8)
    # plt.xlabel('Number of Resources Selected')
    # plt.ylabel('Number of Queries')
    # plt.title('Histogram of Resources Selected per Query')
    # plt.xticks(range(0, 17))
    # plt.tight_layout()
    # plt.savefig('resources_selected_histogram.png')
    #
    #
    #
    #
    # #finally, draw a histogram of the number of queries that selected for each collection
    # highest_selected_counts = {collection: 0 for collection in collections}
    #
    # # For each query, identify the highest-scoring collection and increment its count
    # for qid in statistics:
    #     if statistics[qid]:  # Ensure there is at least one score
    #         max_score = max(statistics[qid])
    #         if max_score > 0:  # Check if the max score is greater than 0
    #             max_index = statistics[qid].index(max_score)
    #             highest_selected_counts[collections[max_index]] += 1
    #
    # # Sort collections based on how many times they were the highest-scoring
    # sorted_collections_by_count = sorted(highest_selected_counts, key=highest_selected_counts.get, reverse=True)
    # sorted_counts = [highest_selected_counts[collection] for collection in sorted_collections_by_count]
    #
    # # Draw the histogram
    # plt.figure(figsize=(12, 6))
    # plt.bar(sorted_collections_by_count, sorted_counts)
    # plt.xlabel('Collections')
    # plt.ylabel('Number of Queries with Collection as Highest Scoring')
    # plt.title('Histogram of Collections Selected as Highest Scoring per Query')
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig('highest_selected_collections_histogram.png')
    #
    #
    #
    #
    #
    # #draw a boxplot of the relevance labels per collection
    # relevance_labels = {collection: [] for collection in collections}
    #
    # # Collect relevance labels for each collection
    # for qid in statistics:
    #     for i, rel in enumerate(statistics[qid]):
    #         relevance_labels[collections[i]].append(rel)
    #
    # # Calculate average relevance for each collection
    # average_relevance_per_collection = {
    #     collection: sum(relevance_labels[collection]) / len(relevance_labels[collection])
    #     if relevance_labels[collection] else 0
    #     for collection in collections}
    #
    # # Sort collections based on their average relevance
    # sorted_collections = sorted(collections, key=lambda x: average_relevance_per_collection[x], reverse=True)



    # Prepare sorted data for boxplot
    # sorted_data = [relevance_labels[collection] for collection in sorted_collections]
    #
    #
    #
    #
    #
    #
    # # Draw the boxplot
    # plt.figure(figsize=(15, 8))
    # plt.boxplot(sorted_data, labels=sorted_collections)
    # plt.xlabel('Collections')
    # plt.ylabel('Relevance Labels')
    # plt.title('Boxplot of Relevance Labels per Collection (Sorted by Average Relevance)')
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig('sorted_relevance_labels_per_collection.png')
    #
    #
    #
    # #draw a boxplot of the highest relevance labels per collection
    #
    #
    # highest_relevance_per_collection = {collection: [] for collection in collections}
    #
    # # For each query, find the collection with the highest relevance score and record the score
    # for qid in statistics:
    #     if statistics[qid]:  # Ensure there is at least one score
    #         max_score = max(statistics[qid])
    #         if max_score > 0:  # Check if the max score is greater than 0
    #             max_index = statistics[qid].index(max_score)
    #             highest_relevance_per_collection[collections[max_index]].append(max_score)
    #
    # # Calculate average of the highest relevance for each collection
    # median_highest_relevance = {collection: np.median(scores) for collection, scores in highest_relevance_per_collection.items()}
    #
    #    # {collection: sum(scores) / len(scores) if scores else 0
    #                              #for collection, scores in highest_relevance_per_collection.items()}
    #
    # # Sort collections by their average highest relevance
    # sorted_collections = sorted(highest_relevance_per_collection.keys(), key=lambda x: median_highest_relevance[x],
    #                             reverse=True)
    #
    # # Prepare sorted data for boxplot
    # sorted_data = [highest_relevance_per_collection[collection] for collection in sorted_collections]
    #
    # sorted_collections = [collection_dict[collection] for collection in sorted_collections]
    #
    # # Draw the boxplot
    # plt.figure(figsize=(11, 6))
    # plt.boxplot(sorted_data, labels=sorted_collections)
    # plt.ylabel('Graded Precision of Best Resources', fontsize=16)
    # plt.yticks(fontsize=13)
    # plt.xticks(fontsize=13)
    # plt.ylim(0, 100)
    # #plt.title('Boxplot of Highest Relevance Labels per Collection (Sorted by Average)')
    # plt.xticks(rotation=60)
    # plt.tight_layout()
    # plt.savefig('sorted_highest_relevance_labels_per_collection.pdf')
    #
    #
    #
    #
    # # draw a boxplot of the highest relevance labels per collection type
    # colection_type_mapping_dict = {
    #     "arguana": "General",
    #     "scidocs": "Scientific",
    #     "scifact": "Scientific",
    #     "dbpedia-entity": "General",
    #     "signal1m": "Social",
    #     "trec-news": "News",
    #     "robust04": "News",
    #     "webis-touche2020": "Debate",
    #     "fever": "Wiki",
    #     "climate-fever": "Wiki",
    #     "nfcorpus": "Biomedical",
    #     "fiqa": "Finance",
    #     "trec-covid": "Biomedical",
    #     "nq": "Wiki",
    #     "hotpotqa": "Wiki",
    #     "msmarco": "General"
    # }
    #
    # highest_relevance_per_type = {type_: [] for type_ in set(collection_dict.values())}
    #
    # # For each query, find the collection with the highest relevance score and record the score in the respective type
    # for qid in statistics:
    #     if statistics[qid]:  # Ensure there is at least one score
    #         max_score = max(statistics[qid])
    #         if max_score > 0:  # Check if the max score is greater than 0
    #             max_index = statistics[qid].index(max_score)
    #             collection_type = colection_type_mapping_dict[collections[max_index]]
    #             highest_relevance_per_type[collection_type].append(max_score)
    #
    # # Calculate median of the highest relevance for each collection type
    # median_highest_relevance = {type_: np.median(scores) for type_, scores in highest_relevance_per_type.items()}
    #
    # # Sort types by their median highest relevance
    # sorted_types = sorted(highest_relevance_per_type.keys(), key=lambda x: median_highest_relevance[x], reverse=True)
    #
    # # Prepare sorted data for boxplot
    # sorted_data = [highest_relevance_per_type[type_] for type_ in sorted_types]
    #
    # # Draw the boxplot
    # plt.figure(figsize=(10, 6))
    # plt.boxplot(sorted_data, labels=sorted_types)
    # plt.ylabel('Graded Precision of Best Vertical', fontsize=16)
    # plt.yticks(fontsize=13)
    # plt.xticks(fontsize=13)
    # plt.ylim(0, 100)
    # plt.xticks(rotation=60)
    # plt.tight_layout()
    # plt.savefig('sorted_highest_relevance_labels_per_collection_type.pdf')
    #
    #
    #
    #











if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the judgement file')
    parser.add_argument('--queries_file', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--qrel_file', type=str, required=True, help='qrel file')
    parser.add_argument('--llm', type=str, required=True, help='llm type')
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    args = parser.parse_args()
    queries_file = args.queries_file
    qrrel_file = args.qrel_file
    input_folder = args.input_folder
    out_file = args.out_file
    llm = args.llm

    main(input_folder, queries_file, llm, out_file, qrrel_file)


