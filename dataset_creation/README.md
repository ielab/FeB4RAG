# This folder is for creating the dataset, or expanding it.

This folder contains the code for creating or reproducing the dataset, or expanding it. The dataset is organized as follows:

- `1_request_creation/`: contains the code for creating the requests, or expanding them.
- `2_search/`: contains the code for searching the requests on search engines, or expanding the search engines.
- `3_response_labeling/`: contains the code to label search result with respect to the requests, labelling models can be expanded.
- `4_check_agreement/`: contains the code to check the agreement between labeling models to human judge, or between two LLMs
- `5_response_generation/`: contains the code for generating responses, or expanding the response generation models.


---------

## Preparation
If you have not yet download BEIR collection, please do so by using the following code:

```bash
python3 BEIR_download.py
```
---------

## 1. Request creation

```bash
cd request_creation
```

First for BEIR dataset that could directly use their query as request
    
```bash
python3 non_llm_request_selection.py \
    --original_dataset_folder ../original_dataset
```

Then for generating requests using LLM, use 

```bash
python3 llm_request_creation.py
    --original_dataset_folder ../original_dataset \
    --api_key <api_key> \ this is the api key for openai
```

Then for selecting the best requests, use 

```bash
python3 llm_request_selection.py
    --original_dataset_folder ../original_dataset
```

Finally, you can just combine all the generated requests using the code below:

```bash
cat original_dataset/*/conversational_queries_final.jsonl > original_dataset/requests.jsonl 
```

---------
## 2. Search

```bash
cd search
```

For searching the requests on search engines, use the script

```bash
python3 encoding_and_eval.py \
    --dataset_dir ../original_dataset \
    --model_dir ../models/ \
    --embedding_dir ../encoding/ \
    --log_dir ../logs/  \
    --task encode  \
    --special_token \
    --feb4rag_dir ../../dataset \
    --fake_queries \
    --model_type custom \
    --specific_model <model_name> \
    --dataset_name <beir_dataset_name>

python3 encoding_and_eval.py \
    --dataset_dir ../original_dataset \
    --model_dir ../models/ \
    --embedding_dir ../encoding/ \
    --log_dir ../logs/  \
    --task eval  \
    --special_token \
    --feb4rag_dir ../../dataset \
    --fake_queries \
    --model_type custom \
    --specific_model <model_name> \
    --dataset_name <beir_dataset_name> 
```
---------


## 3. Response labeling

```bash
cd response_labeling
```

Use the following script to label the search results with respect to the requests, first, note that below script corresponds to one BEIR dataset, here we use nfcorpus as an example

```bash
python3 relevance_labeling_bier.py \
    --request_file ../../dataset/queries/requests.jsonl \
    --corpus_file ../original_dataset/nfcorpus/corpus.jsonl \
    --search_folder ../../dataset/search_results \
    --llm <solar-11b or lgs-13b> \
    --model_path <path of llm> \
    --out_file ../labels/nfcorpus/<llm>.jsonl 
```

The code above will then create a jsonl judgement file for the nfcorpus dataset, for the corresponding llm.

Then if you want to fuse the judgements from different llms, use the following code:

```bash
python3 fuse_judgements.py \
    --labeling_folder ../labels/ \
    --llms <llm1> <llm2> 
```
This, will create a fused-<llm1>-<llm2>.jsonl file in the same folder from the judgements of llm1 and llm2.

Finally, if you want to create the qrels file, use the following code:

```bash
For Resource selection

python3 LABEL2RS.py \
    --input_folder ../labels/ \
    --queries_file ../../dataset/queries/requests.jsonl \
    --llm fused-<llm1>-<llm2> \
    --out_file BEIR-QRELS-RS.txt
    
For Resource merging

python3 LABEL2RM.py \
    --input_folder ../labels/ \
    --llm fused-<llm1>-<llm2> \
    --out_file BEIR-QRELS-RM.txt
```
---------

## 4. Check agreement
```bash
cd check_agreement
```

You can check the agreement between labeling models to human judge using the following code:

```bash
python3 agreement_with_human.py \
    --label_folder ../labels_pre/ \ Note for this you have to judge from the labels comming form the qrels
    --llm <llm> \
    --qrel_file ../../dataset/qrels/original_qrel.tsv \
    --mapping_file ../../dataset/queries/rid_mapping.jsonl
```

You can also check the agreement between two LLMs using the following code:

```bash
python3 agreement_between_llms.py \
    --label_folder ../labels/
```

---------

## 5. Response generation

```bash
cd response_generation
```

To generate responses, use the following code:

```bash
python3 RAG_generation.py \
    --original_data_folder ../../original_dataset \
    --query_file ../../dataset/queries/requests.jsonl \
    --llm gpt4 \
    --type <naive-fed or best-fed> \
    --search_folder ../../dataset/search_results \
    --top 10 \
    --labeling_folder ../../dataset/labels \
    --label_type <name of the file in labels, like solar-11b>
    --out_folder generated_responses
```

This will generate the responses for the requests, and save them in the `generated_responses` folder.




    
