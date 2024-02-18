import argparse
import os
import json
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

timeout_duration = 120.0
collections = ["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever", "nfcorpus", "fiqa", "trec-covid", "nq", "hotpotqa", "robust04", "msmarco", "webis-touche2020"]

system_prompt = "You are a helpful assistant helping to answer user requests based on the provided search result.\n" \
                "Your responses should directly address the user's request and must be based on the information obtained from the provided search results.\n" \
                "You are forbidden to create new information that is not supported by these results. \n" \
                "You must attribute your response to the source from the search results by including citations, for example, [1].\n"

user_prompt = "User Request:\n" \
              "{query}\n" \
               "Search Results:\n"\
               "{search_results}\n"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

open_source_prompt = f"{BOS}{B_INST} {B_SYS}\n" \
                f"{system_prompt}\n" \
                f"{E_SYS}\n\n" \
                 f"{user_prompt}\n" \
                 f"{E_INST}Response:\n"



def load_model(api_key):
    model_kwargs = {"seed": 1}
    chat_model = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, max_tokens=4096, api_key=api_key, request_timeout=timeout_duration,
                            model_kwargs=model_kwargs)
    model = chat_model
    return model


def model_openai(model, query, search_results):
    search_results_combined = ""
    for rank, result in enumerate(search_results):
        search_results_combined += "[" + str(rank+1) + "]:\n" + result

    prompt = user_prompt.format(query=query, search_results=search_results_combined)
    prompt_gpt = [SystemMessage(content=system_prompt),
                  HumanMessage(content=prompt),
                  AIMessage(content="Response:")]
    print(prompt_gpt)

    response = model.generate([prompt_gpt]).generations[0][0].text
    return response

def model_opensource(model, query, search_results, max_per_passage):

    response = model.generate_answer(open_source_prompt, query, search_results, max_per_passage)
    return response




def load_corpus(corpus_file):
    corpus_dict = {}
    with open(corpus_file, 'r') as file:
        for line in file:
            current_json = json.loads(line)
            if "title" in current_json:
                corpus_dict[current_json['_id']] = f"Title: {current_json['title']}\nText:{current_json['text']}\n"
            else:
                corpus_dict[current_json['_id']] = current_json['text']

    return corpus_dict

def load_queries(corpus_file):
    corpus_dict = {}
    with open(corpus_file, 'r') as file:
        for line in file:
            current_json = json.loads(line)

            corpus_dict[current_json['_id']] = current_json['text']

    return corpus_dict


def main():
    ## Parse the arguments
    parser = argparse.ArgumentParser(description='generate final response using search result')
    parser.add_argument('--original_data_folder', type=str, required=True, help='Path to the original dataset file')
    parser.add_argument('--query_file', type=str, required=True, help='Path to the query file')
    parser.add_argument('--out_file', type=str, required=True, help='generation_out')
    parser.add_argument('--llm', type=str, help='llm')
    parser.add_argument('--model_path', type=str, default='llama2-7b-chat', help='the path of the model')
    parser.add_argument('--type', type=str, choices=['naive-fed', 'gold', "best-fed"], required=True, help='type of generation')

    parser.add_argument('--labeling_folder', type=str, help='Path to the labeling folder (only for gold type)')
    parser.add_argument('--label_type', type=str, help='Label type (only for gold type)')

    # Add arguments that are specific to the 'non-fed' type
    parser.add_argument('--search_folder', type=str, help='Path to the original dataset file (only for non-fed type)')
    parser.add_argument('--top', type=int, help='Top results (only for non-fed type)')


    # Parse the arguments
    args = parser.parse_args()



    ## Read the original corpus
    original_data_folder = args.original_data_folder
    query_file = args.query_file
    search_folder = args.search_folder
    out_file = args.out_file
    llm = args.llm
    model_path = args.model_path
    top = args.top
    print(top)
    #first load corpus and search results

    if llm=="gpt4":
        api_key_path = "/scratch/project/neural_ir/dylan/LLM_FS/api_key"
        api_key = open(api_key_path, 'r').read()
        model = load_model(api_key)
    else:
        from models.solar_models import SOLARModel
        import random
        random.seed(0)
        model = SOLARModel()
        model.load_model(model_path=model_path)
        max_len_passages = 3584
        max_per_passage = int(max_len_passages // top)

    result_dict = {}


    # qrel_file = args.qrel_file
    # with open(qrel_file, 'r') as file:
    #     highest_labels = {}
    #     qrel_dict = {}
    #     for line in file:
    #         qid, _, docid, label_str = line.strip().split()
    #         label = int(label_str)
    #
    #         # Initialize if qid is encountered for the first time
    #         if qid not in highest_labels:
    #             highest_labels[qid] = label
    #             qrel_dict[qid] = [docid]
    #         else:
    #             # Update if a new higher label is found
    #             if label > highest_labels[qid]:
    #                 highest_labels[qid] = label
    #                 qrel_dict[qid] = [docid]
    #             # Add the docid to the list if it has the same highest label
    #             elif label == highest_labels[qid]:
    #                 qrel_dict[qid].append(docid)

    for collection in tqdm(collections):
        corpus_file = os.path.join(original_data_folder, collection, "corpus.jsonl")
        corpus_dict = load_corpus(corpus_file)

        if args.type == "best-fed":
            print("loading gold")
            label_file = os.path.join(args.labeling_folder, collection, args.label_type + ".jsonl")

            with open(label_file, 'r') as f:
                for line in f:
                    label = json.loads(line)
                    qid = label["qid"]
                    docid = label["docid"]
                    label_conv = label["label"]
                    if qid not in result_dict:
                        result_dict[qid] = {}
                    if collection not in result_dict[qid]:
                        result_dict[qid][collection] = {}
                    if label_conv != "nr":
                        if label_conv not in result_dict[qid][collection]:
                            result_dict[qid][collection][label_conv] = []
                        result_dict[qid][collection][label_conv].append((docid, corpus_dict[docid]))
        else:
            tem_search_folder = os.path.join(search_folder, collection)
            search_file = os.listdir(tem_search_folder)[0]
            search_file = os.path.join(tem_search_folder, search_file)
            with open(search_file, 'r') as file:
                for line in file:
                    #trec format
                    qid, _, docid, rank, score, _ = line.strip().split()
                    if qid not in result_dict:
                        result_dict[qid] = {}
                    if collection not in result_dict[qid]:
                        result_dict[qid][collection] = []
                    rank = int(rank)
                    if rank<10:
                        #if docid in qrel_dict[qid]:
                            #if corpus_dict[docid] not in result_dict[qid][collection]:
                            #if docid not in result_dict[qid][collection]:
                            #result_dict[qid][collection].append(docid)
                            #if corpus_dict[docid] not in result_dict[qid][collection]:
                        result_dict[qid][collection].append((docid, corpus_dict[docid]))
                        #result_dict[qid][collection].append(corpus_dict[docid])

    result_dict_real = {}
    docid_dict_real = {}

    if args.type == "best-fed":
        for qid in result_dict:
            result_dict_real[qid] = []
            docid_dict_real[qid] = []
            # rank based on from all collections, [key, hr, r]
            for label in ["key", "hr", "r"]:
                for collection in result_dict[qid]:
                    if label in result_dict[qid][collection]:
                        for docid, passage in result_dict[qid][collection][label]:
                            result_dict_real[qid].append(passage)
                            docid_dict_real[qid].append(docid)
    else:
        for qid in result_dict:
            result_dict_real[qid] = []
            docid_dict_real[qid] = []
            for i in range(0, len(result_dict[qid][collections[0]])):
                #only append based on index of i
                for collection in result_dict[qid]:
                    docid, passage = result_dict[qid][collection][i]
                    result_dict_real[qid].append(passage)
                    docid_dict_real[qid].append(docid)


    #load queries

    query_dict = load_queries(query_file)

    qid_parsed = set()
    if os.path.exists(out_file):
        #first read these that
        with open(out_file, 'r') as file:
            for line in file:
                current_json = json.loads(line)
                qid_parsed.add(current_json["qid"])

        fw = open(out_file, 'a')

    else:
        fw = open(out_file, 'w')
    #only use the first 500 queries
    #query_dict = dict(list(query_dict.items())[500:])

    for qid in tqdm(query_dict):
        query = query_dict[qid]
        if qid in qid_parsed:
            continue
        print(qid)
        print(query)
        current_result = result_dict_real[qid]
        #search_result_list = []
        if len(current_result) < top:
            top = len(current_result)
        search_result_list = current_result[:top]
        search_ids_list = docid_dict_real[qid][:top]
        print(search_ids_list)
        if llm=="gpt4":
            try:
                response = model_openai(model, query, search_result_list)
            except:
                response = "Error in getting response"
        else:
            response = model_opensource(model, query, search_result_list, max_per_passage)
        print(response)
        fw.write(json.dumps({"qid": qid, "query": query, "response": response, "search_results": search_result_list, "search_ids": search_ids_list}) + "\n")



if __name__ == "__main__":
    main()








