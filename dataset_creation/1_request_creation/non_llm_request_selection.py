import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from tqdm import tqdm

datasets = ["nfcorpus", "fiqa", "trec-covid", "nq", "hotpotqa", "robust04", "msmarco", "webis-touche2020"]



def main():
    dataset_folder="original_dataset"
    for dataset in datasets:
        print(dataset)
        query_file = os.path.join(dataset_folder, dataset, "queries.jsonl")
        output_file = os.path.join(dataset_folder, dataset, "conversational_queries_final.jsonl")
        qid_parsed = set()
        if os.path.exists(output_file):
            for line in open(output_file):
                query = json.loads(line)
                query_id = query["_id"]
                qid_parsed.add(query_id)
        result_num = len(qid_parsed)

        qrel_file = os.path.join(dataset_folder, dataset, "qrels", "test.tsv")
        rel_dict = {}
        #select the top 50 qids in the qrel file that has the most relevant documents
        for line in open(qrel_file):
            #jump the first line
            if line.startswith("query-id"):
                continue
            qid, docid, rel = line.strip().split("\t")
            if qid not in rel_dict:
                rel_dict[qid] = 0
            if int(rel) > 0:
                rel_dict[qid] += 1

        sorted_rel_dict = sorted(rel_dict.items(), key=lambda x: x[1], reverse=True)
        qid_set = []
        if len(sorted_rel_dict) < 300:
            qid_set = [qid for qid, rel in sorted_rel_dict]
        else:
            for qid, rel in sorted_rel_dict[:300]:
                qid_set.append(qid)

        query_dict = {}
        with open(query_file, 'r') as f:
            for line in f:
                query = json.loads(line)
                query_id = query["_id"]
                if query_id not in qid_set:
                    continue
                if query_id in qid_parsed:
                    continue
                query_text = query["text"]
                if len(query_text.split()) <=2:
                    qid_parsed.add(query_id)
                    continue
                query_dict[query_id] = query_text
        if len(query_dict) == 0:
            continue
        with open(output_file, 'a') as fw:
            for query_id in tqdm(qid_set):
                if result_num >= 50:
                    break
                if query_id in qid_parsed:
                    continue
                #generate three times per query
                query = query_dict[query_id]
                #now user input, if 1, then the query is conversational
                user_input = input("input response for query \n{}:\n".format(query))
                if int(user_input) != 1:
                    print("excluded")
                    continue
                result_num += 1
                print("current result num: ", result_num)

                current_line = {"_id": query_id, "text": query, "metadata": {}}
                fw.write(json.dumps(current_line) + "\n")

if __name__ == "__main__":
    main()




