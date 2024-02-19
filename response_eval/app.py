import json
import streamlit as st
import random
import os
import html
import re
import argparse
st.set_page_config(layout="wide")


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def process_response_with_citations(response, search_results):
    # Regular expression to find patterns like [7]
    matches = re.finditer(r'\[\d+\]', response)
    citation_texts = []

    for match in matches:
        # Extracting the citation number
        citation_number = int(match.group()[1:-1]) - 1  # Convert to zero-based index

        if 0 <= citation_number < len(search_results):
            # Extract the citation text
            citation_text = search_results[citation_number]
            citation_entry = f"[{match.group()[1:-1]}] {citation_text}"
            citation_texts.append(citation_entry)

    # Combine the original response with a well-formatted citation section
    full_text = response
    if citation_texts:
        full_text += "\n\nCitations:\n\n" + "\n".join(citation_texts)
    return full_text


def display_pairwise_comparison(query, response1, response2, search_results1, search_results2, unique_id):
    col0, col1, col3 = st.columns([2, 6, 2])
    with col1:
        st.markdown(f"### User Request: {query}")

    # Process responses to append citations
    processed_response1 = process_response_with_citations(response1, search_results1)
    processed_response2 = process_response_with_citations(response2, search_results2)

    # Check if order already exists in session state
    if unique_id not in st.session_state:
        # Randomize the order of responses if not already saved
        responses = [('A', processed_response1), ('B', processed_response2)]
        random.shuffle(responses)
        st.session_state[unique_id] = responses
    else:
        # Retrieve saved order
        responses = st.session_state[unique_id]

    # Create two columns for the responses
    col0, col1, col2, col3 = st.columns([2, 3, 3, 2])
    count_label = {}
    for count, (label, response) in enumerate(responses, start=1):
        with col1 if count == 1 else col2:
            count_label[label] = f"Response {count}"
            st.markdown(f"#### Response {count}:")
            st.markdown(response, unsafe_allow_html=True)

    return count_label


def save_selection(selection, file_path="selections.jsonl"):
    with open(file_path, 'a') as file:
        # Convert the selection dictionary to a JSON string
        json_string = json.dumps(selection)
        # Write the JSON string as a new line in the file
        file.write(json_string + '\n')

def load_selections(file_path="selections.jsonl"):
    parsed_ids = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                # Parse each line as a JSON object
                current_dict = json.loads(line)
                # Add the query ID to the set
                qid = current_dict["qid"]
                parsed_ids.add(qid)
    else:
        open(file_path, 'w').close()  # Create the file if it doesn't exist

    return parsed_ids


def main():
    args = argparse.ArgumentParser(description='input args')
    args.add_argument('--naive_fed_file', type=str, required=True, help='naive fed file')
    args.add_argument('--best_fed_file', type=str, required=True, help='best fed file')
    args.add_argument('--rid_file', type=str, required=True, help='Path to the rid file')
    # output
    args.add_argument('--output_file', type=str, required=True, help='output file')
    args = args.parse_args()


    parsed_ids = load_selections()

    sample_qid_file ="sample_rids.txt"
    if not os.path.exists(sample_qid_file):
        print("No sample queries file found")
        return
    sample_qids = []
    with open(sample_qid_file) as f:
        for line in f:
            sample_qids.append(line.strip())

    best_fed_data = read_jsonl("../response_generated/best-fed/gpt4.jsonl")
    naive_fed_data = read_jsonl("response_generated/naive-fed/gpt4.jsonl")
    #then remove from the best_fed_data the ones that are already parsed


    best_fed_data = [x for x in best_fed_data if x["qid"] not in parsed_ids]
    naive_fed_data = [x for x in naive_fed_data if x["qid"] not in parsed_ids]

    #then should be also in the sample_qids
    best_fed_data = [x for x in best_fed_data if x["qid"] in sample_qids]
    naive_fed_data = [x for x in naive_fed_data if x["qid"] in sample_qids]
    #the ids are different in order, need to sort them
    best_fed_data.sort(key=lambda x: x["qid"])
    naive_fed_data.sort(key=lambda x: x["qid"])
    #now print in each list



    # Initialize or update the index in Streamlit's session state
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if st.session_state.index < len(best_fed_data):
        best = best_fed_data[st.session_state.index]
        unique_id = f"{st.session_state.index}_{best['qid']}"
        naive = naive_fed_data[st.session_state.index]
        qid = best["qid"]



        print(qid)
        print(naive["qid"])

        count_label = display_pairwise_comparison(best["query"], best["response"], naive["response"],
                                                  best["search_results"], naive["search_results"], unique_id)

        count_label["No Preference"] = 'No Preference'
        col0, col1, col3 = st.columns([2, 6, 2])
        with col1:
            st.markdown("#### Choose the better response")
            choice_number_1 = st.radio("##### Coverage: Coverage measures the cumulative extent to which presented information is pertinent to the users’ information need.", list(count_label.keys()), key=f"radio_{unique_id}_1")
            choice_number_2 = st.radio("##### Consistency: Consistency measures the extent that the information provided is consistent to the citation", list(count_label.keys()), key=f"radio_{unique_id}_2")
            choice_number_3 = st.radio("##### Correctness: Correctness gauges to which degree the information provided in the response is factually correct, reliable, and addressing the user’s information needs.", list(count_label.keys()), key=f"radio_{unique_id}_3")
            choice_number_4 = st.radio("##### Clarity: Clarity measures the extent to which the information provided is clear and understandable to the user.", list(count_label.keys()), key=f"radio_{unique_id}_4")

            if st.button("Submit", key=f"submit_{unique_id}"):
                chosen_sources = []

                for choice_number in [choice_number_1, choice_number_2, choice_number_3, choice_number_4]:
                    if choice_number != "No Preference":
                        chosen_label = count_label[choice_number]
                        chosen_response = best["response"] if chosen_label == 'A' else naive["response"]
                        chosen_source = 'best-fed' if chosen_response == best["response"] else 'naive-fed'
                    else:
                        chosen_source = 'none'
                    chosen_sources.append(chosen_source)

                new_save_dict = {"qid": qid, "chosen_sources": chosen_sources}
                save_selection(new_save_dict)
                st.session_state.index += 1
                st.experimental_rerun()


if __name__ == "__main__":
    main()