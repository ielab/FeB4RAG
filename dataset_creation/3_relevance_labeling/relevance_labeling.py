import argparse
import os
import json
from tqdm import tqdm


api_key_path = "api_key"
pseudo_output = '{"M": 2, "T": 1,"O": 1}'

labeling_prompt = "Given a user request and a search result, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n\n" \
                    "3 = key, this search result contains relevant, diverse, informative and correct answers to the user request; the user request can be fulfilled by relying only on this search result.\n" \
                    "2 = high relevance, this search result contains relevant, informative and correct answers to the user request; however, it does not span diverse perspectives, and including another perspective can help with a better answer to the user request.\n" \
                    "1 = minimal relevance, this search result contains relevant answers to the user request. However, it is impossible to answer the user request based solely on the search result. \n" \
                    "0 = not relevant, this search result does not contain any relevant answer to the user request.\n" \
                    'Assume that you are collecting all the relevant search results to write a final answer for the user request.\n' \
                    'User Request:\n' \
                    'A user typed the following request.\n' \
                    '{query}.\n' \
                    'Result:\n' \
                    'Consider the following search result:\n' \
                    '—BEGIN Search Result CONTENT—\n' \
                    '{snippet}\n' \
                    '—END Search Result CONTENT—\n' \
                    'Instructions:\n' \
                    'Split this problem into steps:\n' \
                    'Consider the underlying intent of the user request.\n' \
                    'Measure how well the search result matches a likely intent of the request (M)\n' \
                    'Measure how trustworthy the search result is (T).\n' \
                    'Consider the aspects above and the relative importance of each, and decide on a final score (O).\n' \
                    'Produce a JSON of scores without providing any reasoning. Example:{"M": 2, "T": 1, "O": 1}\n' \
                    'Results:\n'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"
assistant_prompt = "Given a user request and a search result, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n\n" \
                    "3 = key, this search result contains relevant, diverse, informative and correct answers to the user request; the user request can be fulfilled by relying only on this search result.\n" \
                    "2 = high relevance, this search result contains relevant, informative and correct answers to the user request; however, it does not span diverse perspectives, and including another perspective can help with a better answer to the user request.\n" \
                    "1 = minimal relevance, this search result contains relevant answers to the user request. However, it is impossible to answer the user request based solely on the search result. \n" \
                    "0 = not relevant, this search result does not contain any relevant answer to the user request.\n" \
                    'Assume that you are collecting all the relevant search results to write a final answer for the user request.\n' \

user_prompt = 'User Request:\n' \
                    'A user typed the following request.\n' \
                    '{query}.\n' \
                    'Result:\n' \
                    'Consider the following search result:\n' \
                    '—BEGIN Search Result CONTENT—\n' \
                    '{snippet}\n' \
                    '—END Search Result CONTENT—\n' \
                    'Instructions:\n' \
                    'Split this problem into steps:\n' \
                    'Consider the underlying intent of the user request.\n' \
                    'Measure how well the search result matches a likely intent of the request (M)\n' \
                    'Measure how trustworthy the search result is (T).\n' \
                    'Consider the aspects above and the relative importance of each, and decide on a final score (O).\n' \
                    'Produce a JSON of scores without providing any reasoning. Example:{"M": 2, "T": 1, "O": 1}\n'

labeling_prompt_llama = f"{BOS}{B_INST} {B_SYS}\n" \
                f"{assistant_prompt}\n" \
                f"{E_SYS}\n\n" \
                 f"{user_prompt}\n" \
                 f"{E_INST}Results:\n"



def main(search_file, query_file, corpus_file, out_file, llm, model_path):

    queries_dict = load_queries(query_file)
    corpus_dict = load_corpus(corpus_file)
    result_dict = load_trec_search(search_file)
    print(out_file)
    if "flan" in llm:
        from models.flan_models import FLANModel
        import random
        random.seed(0)
        model = FLANModel()
        model.load_model(model_path=model_path)
        process_data_opensource(queries_dict, corpus_dict, labeling_prompt, result_dict, model, out_file)
    elif ("solar" in llm) or ("lgs-13b" in llm):
        from models.solar_models import SOLARModel
        import random
        random.seed(0)
        model = SOLARModel()
        model.load_model(model_path=model_path)
        print("Model loaded")
        process_data_opensource(queries_dict, corpus_dict, labeling_prompt_llama, result_dict, model, out_file )


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

def load_trec_search(search_file):
    #trec format file, load with qid be the key, then value if rank<10, it should be loaded
    result_dict ={}
    with open(search_file, 'r') as file:
        for line in file:
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) >= 10:
                continue
            if qid not in result_dict:
                result_dict[qid] = {}
            result_dict[qid][docid] = score
    return result_dict

def process_data_opensource(queries, corpus, prompt, result_dict, model, output_file):
    finished_set = set()
    if os.path.exists(output_file):
        finished_set = check_already_labeled(output_file)
        output = open(output_file, 'a')
    else:
        output = open(output_file, 'w')

    for qid in tqdm(result_dict):
        query = queries[qid]
        for docid in result_dict[qid]:
            try:
                if qid + "_" + docid in finished_set:
                    continue
                snippet = corpus[docid]

                text = model.generate(prompt, query, snippet)
                try:
                    current_json = json.loads(text)
                    converted_score = convert_score_to_label(current_json['O'])
                    #print(current_json)
                except:
                    current_json = extract_first_json(text)
                    #print(current_json)
                    if current_json is None:
                        print("Error occurred on the following text:")
                        print(f"{qid}, {docid}")
                        output.write(json.dumps({"qid": qid, "docid": docid, "label": 'nr'}) + "\n")
                        #print(text)
                        continue
                    try:
                        converted_score = convert_score_to_label(current_json['O'])
                    except:
                        print("Error occurred on the following text:")
                        print(f"{qid}, {docid}")
                        print(text)
                        output.write(json.dumps({"qid": qid, "docid": docid, "label": 'nr'}) + "\n")
                        continue
                output.write(json.dumps({"qid": qid, "docid": docid, "label": converted_score}) + "\n")
            except:
                output.write(json.dumps({"qid": qid, "docid": docid, "label": 'nr'}) + "\n")
                print("Error occurred on the following text:")
                print(f"{qid}, {docid}")
                continue


import re
def extract_first_json(text):
    """
    Extract the first JSON object from a given string.

    :param text: The string containing the JSON object.
    :return: The first JSON object if found, otherwise None.
    """
    try:
        # Regex pattern to find the first JSON object
        pattern = r'\{.*?\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def convert_score_to_label(score):
    # The following weights are given to the relevance levels of documents: wNon = 0, wRel = 0.25, wHRel =0.5, wKey = rNav = 1
    if score == 0:
        return "nr"
    elif score == 1:
        return "r"
    elif score == 2:
        return "hr"
    elif score == 3:
        return "key"

def check_already_labeled(out_file):
    check_already_labeled = set()
    with open(out_file, 'r') as file:
        for line in file:
            current_json = json.loads(line)
            representation = current_json['qid'] + "_" + current_json['docid']
            check_already_labeled.add(representation)
    return check_already_labeled



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--search_folder', type=str, required=True, help='Path to the search trec file')
    parser.add_argument('--query_file', type=str, required=True, help='Path to the query file')
    parser.add_argument('--corpus_file', type=str, required=True, help='corpus file')
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    parser.add_argument('--llm', type=str, help='llm')
    parser.add_argument('--model_path', type=str, default='llama2-7b-chat', help='the path of the model')
    args, _ = parser.parse_known_args()

    print(args)
    search_folder = args.search_folder
    search_file = os.listdir(search_folder)[0]
    search_file = os.path.join(search_folder, search_file)

    main(search_file, args.query_file, args.corpus_file, args.out_file, args.llm, args.model_path)