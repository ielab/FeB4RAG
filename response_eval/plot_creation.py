# Description: This file contains the functions to create the plots for the data analysis
import json
import matplotlib.pyplot as plt
import numpy as np

def calculating_final_judgement(judgement_list):
    #calculate the final judgement based on the input list, providing list like ["naive-fed", "none", "naive-fed", "none"]
    #the final judgement will be the most chosen source, if there is a tie, the final judgement will be none

    count_naive_fed = 0
    count_best_fed = 0
    for source in judgement_list:
        if source == "naive-fed":
            count_naive_fed += 1
        elif source == "best-fed":
            count_best_fed += 1
    if count_naive_fed > count_best_fed:
        return "naive-fed"
    elif count_naive_fed < count_best_fed:
        return "best-fed"
    else:
        return "none"


dense_patterns = ['////', '\\\\\\\\', '...']

def create_plot_overall():
    #create a bar plot from the manual judgdement, with input as {"qid": "114", "chosen_sources": ["naive-fed", "none", "naive-fed", "none"]}
    #the chosen sources corresponde to Coverage, Consistency, Correctness, Clarity
    #the plot will show the number of times each source was chosen for each category, either naive-fed, best-fed or none
    #the plot will be saved as a pdf file
    #also there should be a final judgement

    #read the file
    result_count_dict = {"Coverage": {"naive-fed": 0, "best-fed": 0, "none": 0}, "Consistency": {"naive-fed": 0, "best-fed": 0, "none": 0}, "Correctness": {"naive-fed": 0, "best-fed": 0, "none": 0}, "Clarity": {"naive-fed": 0, "best-fed": 0, "none": 0}, "Final": {"naive-fed": 0, "best-fed": 0, "none": 0}}

    with open("selections_1.jsonl") as f:
        for line in f:
            current_dict = json.loads(line)
            qid = current_dict["qid"]
            chosen_sources = current_dict["chosen_sources"]
            #calculate the final judgement
            final_judgement = calculating_final_judgement(chosen_sources)
            #calculate the count
            converage_source = chosen_sources[0]
            consistency_source = chosen_sources[1]
            correctness_source = chosen_sources[2]
            clarity_source = chosen_sources[3]
            result_count_dict["Coverage"][converage_source] += 1
            result_count_dict["Consistency"][consistency_source] += 1
            result_count_dict["Correctness"][correctness_source] += 1
            result_count_dict["Clarity"][clarity_source] += 1
            result_count_dict["Final"][final_judgement] += 1

    categories = list(result_count_dict.keys())
    sources = ["naive-fed", "best-fed", "none"]
    labels = ["naive-fed", "best-fed", "equal"]

    percentage_data = []
    total_counts = 80
    for source in sources:
        percentage_data.append(
            [100* result_count_dict[category][source] / total_counts for category in categories])

    # Plotting
    x = np.arange(len(categories))  # the label locations
    width = 0.2  # the width of the bars
    #dimention of bar plot

    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(len(sources)):
        ax.bar(x + i * width, percentage_data[i], width, label=labels[i], hatch=dense_patterns[i], color='gray', edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_xlabel('Criteria')
    ax.set_ylabel('Preference Percentage (%)', fontsize=15)
    #ax.set_title('Winning Cases by Criteria', fontsize=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, fontsize=15)  # Increase font size for x-ticks
    #rotate the x axis
    plt.xticks(rotation=45)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)

    ax.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig('generation_judgement.pdf')

    #create the bar plot,

collections =["msmarco", "trec-covid", "nfcorpus", "scidocs", "nq", "hotpotqa", "fiqa", "signal1m", "trec-news",  "robust04",  "arguana",  "webis-touche2020",  "dbpedia-entity",  "fever", "climate-fever", "scifact"]

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


def plot_line(data, ax, title, show_y_labels):
    labels = ["naive-fed", "best-fed", "equal"]
    source_labels = ["naive-fed", "best-fed", "none"]
    dense_patterns = ['///', '\\\\\\', '..']
    colors = 'gray'
    edgecolors = 'black'
    fontsize = 20  # Increase font size for readability

    # Convert counts to percentages
    total = sum(data.values())
    percentages = [100 * data[source_label] / total for source_label in source_labels]

    bars = ax.bar(labels, percentages, color=colors, edgecolor=edgecolors)
    for bar, pattern in zip(bars, dense_patterns):
        bar.set_hatch(pattern)

    ax.set_title(title, fontsize=fontsize)
    if show_y_labels:
        ax.set_ylabel('Percentage (%)', fontsize=fontsize)
    ax.set_ylim(0, 100)
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.tick_params(axis='y', labelsize=fontsize)  # Increase y-axis label size



def create_plot_individual_collection():
    #this will be a line plot, showing the percentage of times each source was chosen for each category, either naive-fed, best-fed or none
    #However, it will only account for final judgements
    #but it will on each x axis be one collection, based on the mapping file
    mapping_file = "fedbier/qid_mapping.tsv"
    mapping_dict = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            qid, collection,_ = line.split()
            mapping_dict[qid] = collection

    # read the file
    result_count_dict = {}
    for collection in collections:
        result_count_dict[collection] = {"naive-fed": 0, "best-fed": 0, "none": 0}
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    with open("selections_1.jsonl") as f:
        for line in f:
            current_dict = json.loads(line)
            qid = current_dict["qid"]
            chosen_sources = current_dict["chosen_sources"]
            #calculate the final judgement
            final_judgement = calculating_final_judgement(chosen_sources)
            qid_collection = mapping_dict[qid]
            result_count_dict[qid_collection][final_judgement] += 1

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
    axes = axes.flatten()  # Flatten the 2D array of axes

    for idx, collection in enumerate(collection_dict.keys()):
        plot_line(result_count_dict[collection], axes[idx], title=collection_dict[collection],
                  show_y_labels=(idx % 4 == 0))

    # Add a legend at the top
    labels = ["naive-fed", "best-fed", "equal"]
    dense_patterns = ['///', '\\\\\\', '..']
    bars = [plt.bar([0], [0], color='gray', edgecolor='black', hatch=pattern)[0] for pattern in dense_patterns]
    legend_fontsize = 24  # Increase legend font size
    fig.legend(bars, labels, loc='lower center', bbox_to_anchor=(0.5, 0.06), fancybox=True, shadow=True, ncol=3, fontsize=legend_fontsize)


    #plt.tight_layout()

    plt.savefig('collection_subplots.pdf')


create_plot_overall()
create_plot_individual_collection()

