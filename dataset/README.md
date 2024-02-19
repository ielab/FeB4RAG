# This part is for the details FeB4RAG dataset

### Prepare original BEIR collection
First, please download the original BEIR collection for the dataset, to do this, you can use the code below:

```bash
python3 BEIR_download.py
```

You can also refer to [BEIR URL](https://github.com/beir-cellar/beir) for more information about BEIR dataset

### Dataset details

#### Request/Queries
- [queries](queries/requets.jsonl), or using [tsv](queries/requests.tsv) format, there are overall 790 requests in the dataset
- [rid mapping file](queries/rid_mapping.jsonl) is also provided, which maps the request id to the original BEIR dataset id and query id

#### Engines
- [engines](engines/engines.csv), there are overall 16 engines in the dataset, each paired with name,model,Description,vertical,Description Source,Num_queries_original,Task,Objective (original),Is the task make sense in chatbot (RAG),Need LLM to generate query?,Can select manually?,Devise manually?,Note

#### Relevance judgdements
- [Resource Selection qrels](qrels/BEIR-QRELS-RS.txt), the relevance judgements for the resource selection task
- [Resource Merging qrels](qrels/BEIR-QRELS-RM.txt), the relevance judgements for the resource merging task

#### Search results
- [search results](search_results/), the search results for each engine, which is contained in each subfolder, with it's highest performming dense retrievers

#### Response generated
- [response](response_generated/), the response generated using two mode best-fed or naive fed using gpt4.

#### Evaluation
- [evaluation](eval_script/), the evaluation script for result selection and merging.
    - [Resource Selection](eval_script/FW_eval_RS.py), the evaluation script for resource selection
