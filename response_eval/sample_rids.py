import json

# File paths

mapping_file = "fedbier/qid_mapping.tsv"
gpt4best = "response_generated/best-fed/gpt4.jsonl"
gpt4naive = "response_generated/naive-fed/gpt4.jsonl"

# Read mapping file
mapping_dict = {}
with open(mapping_file) as f:
    for line in f:
        qid, collection, _ = line.strip().split("\t")
        if collection not in mapping_dict:
            mapping_dict[collection] = set()
        mapping_dict[collection].add(qid)



# Read GPT-4 responses and calculate lengths
gpt4LEN_DATA = {}
with open(gpt4best) as f:
    for line in f:
        current_dict = json.loads(line)
        gpt4LEN_DATA[current_dict["qid"]] = len(current_dict["response"])

with open(gpt4naive) as f:
    for line in f:
        current_dict = json.loads(line)
        gpt4LEN_DATA[current_dict["qid"]] += len(current_dict["response"])

# Sample 5 queries from each collection
output = []
for collection in mapping_dict:
    # Select the shortest 5 responses from GPT-4
    qids = list(mapping_dict[collection])
    gpt_sorted = sorted(qids, key=lambda x: gpt4LEN_DATA[x])
    sampled_queries = gpt_sorted[:5]

    for qid in sampled_queries:
        output.append(qid)

# Write to output file
with open("response_generated/sample_queries.txt", "w") as f:
    for qid in output:
        f.write(f"{qid}\n")


