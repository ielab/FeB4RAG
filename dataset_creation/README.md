# This folder is for creating the dataset, or expanding it.

This folder contains the code for creating or reproducing the dataset, or expanding it. The dataset is organized as follows:

- `1_request_creation/`: contains the code for creating the requests, or expanding them.
- `2_search/`: contains the code for searching the requests on search engines, or expanding the search engines.
- `3_response_labeling/`: contains the code to label search result with respect to the requests, labelling models can be expanded.
- `4_check_agreement/`: contains the code to check the agreement between labeling models to human judge, or between two LLMs
- `5_response_generation/`: contains the code for generating responses, or expanding the response generation models.

---------
## 1. Request creation

```bash
cd request_creation
```

First for BEIR dataset that could directly use their query as request
    
```bash
python3 non_llm_request_selection.py
```

Then for generating requests using LLM, use 

```bash
python3 llm_request_creation.py
```

Then for selecting the best requests, use 

```bash
python3 llm_request_selection.py
```

---------

## 2. Search

```bash
cd search
```

For searching the requests on search engines, use the script

```bash
python3 encoding_and_eval.py
```

---------


## 3. Response labeling

```bash
cd response_labeling
```






    
