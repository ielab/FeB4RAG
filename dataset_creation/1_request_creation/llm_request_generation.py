import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from tqdm import tqdm
import argparse

datasets = ["arguana", "scidocs", "scifact", "dbpedia-entity", "signal1m", "trec-news", "fever", "climate-fever"]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

system_prompt = "You are an helpful assistant helping to reformulate the provided text, that describes a need of a user, into a conversational question that expresses the user need. " \
                "The generated text will be used for a user to ask a chatbot for a direct response." \
                "Therefore, it should not include information about the retrieval step."

user_prompt_dict = {
    "arguana": "Consider the following text. "
               "The text provides an argument with claims. I want you to formulate a question that asks to find counter-arguments to the main claim in the text. "
               "In your question, specify clearly what that claim is, but do not refer explicitly to the text. "
               "Please only include the formulated question in your response. \n"
                "Text:\n {query}\n",

    "scidocs": "Consider the following text. "
               "The text is the title of a research article. I want you to formulate a question that asks to find related articles to the one provided in the text."
               "Please only include the formulated question in your response. \n"
                "Text:\n {query}\n",

    "scifact": "Consider the following text. "
               "The text is a scientific claim. I want you to formulate a question that asks to find evidence that supports or refutes the claim made in the text."
               "In your question, do not specify that you want to find evidence."
               "Please only include the formulated question in your response. \n"
                "Text:\n {query}\n",

    "dbpedia-entity": "Consider the following text. "
               "The text provides an entity. I want you to formulate a question that asks to find relevant information about the entity."
               "In your question, do not mention that the text is an entity."
               "Please only include the formulated question in your response. \n"
                "Text:\n {query}\n",

    "signal1m": "Consider the following text. "
               "The text is the title of a news article. I want you to formulate a question that asks to find relevant Tweet messages about the provided news article."
               "In your question, do not mention that the text is the title of the news article."
               "Please only include the formulated question in your response. \n"
                "Text:\n {query}\n",

    "trec-news": "Consider the following text. "
               "The text is a topic, The text is a topic, I want you to formulate a question that asks to find relevant news based on the topic."
               "Please only include the formulated question in your response. \n"
                "Text:\n {query}\n",

    "fever": "Consider the following text. "
               "The text is a claim. I want you to formulate a question that asks to find evidence that supports or refutes the claim made in the text."
               "In your question, do not specify that you want to find evidence."
               "Please only include the formulated question in your response. \n"
                "Text:\n {query}\n",

    "climate-fever": "Consider the following text. "
               "The text is a claim. I want you to formulate a question that asks to find evidence that supports or refutes the claim made in the text."
               "In your question, do not specify that you want to find evidence."
               "Please only include the formulated question in your response. \n"
                "Text:\n {query}\n",
}

user_message_2 = "Please formulate again so that it is different from the previous response.\n"



prompt_template_solar = \
    f"{BOS}{B_INST} {B_SYS}\n" \
    f"{system_prompt}\n" \
    f"{E_SYS}\n\n" \
    "{user_prompt}" \
    f"{E_INST}Response:\n"

prompt_template_gpt = [SystemMessage(content=system_prompt),
                       HumanMessage(content="{user_prompt}"),
                       AIMessage(content="Response:")]

def load_model(api_key):
    model_kwargs = {"seed": 1}
    chat_model = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, max_tokens=2048, api_key=api_key,
                            model_kwargs=model_kwargs)
    model = chat_model
    return model


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--original_dataset_folder", type=str)
    args.add_argument("--api_key", type=str)
    args = args.parse_args()


    api_key = args.api_key

    model = load_model(api_key=api_key)
    dataset_folder=args.original_dataset_folder
    for dataset in datasets:
        print(dataset)
        query_file = os.path.join(dataset_folder, dataset, "queries.jsonl")
        output_file = os.path.join(dataset_folder, dataset, "conversational_queries.jsonl")
        qid_parsed = set()

        if os.path.exists(output_file):
            for line in open(output_file):
                query = json.loads(line)
                query_id = query["_id"]
                qid_parsed.add(query_id)

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
        qid_set = set()
        for qid, rel in sorted_rel_dict[:50]:
            qid_set.add(qid)
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
                query_dict[query_id] = query_text

        if len(query_dict) == 0:
            continue
        with open(output_file, 'a') as fw:
            for query_id in tqdm(query_dict):
                #generate three times per query
                query = query_dict[query_id]
                user_prompt = user_prompt_dict[dataset].format(query=query)
                prompt_gpt = [SystemMessage(content=system_prompt),
                              HumanMessage(content=user_prompt),
                              AIMessage(content="Response:")]
                all_responses = []
                for time in range(0, 3):
                    response = model.generate([prompt_gpt]).generations[0][0].text
                    all_responses.append(response)
                    response_aug = "Response: " + response

                    prompt_gpt[-1] = AIMessage(content=response_aug)
                    prompt_gpt.append(HumanMessage(content=user_message_2))
                print(query_id, all_responses)
                current_line = {"_id": query_id, "conversational_queries": all_responses}
                fw.write(json.dumps(current_line) + "\n")

if __name__ == "__main__":
    main()









