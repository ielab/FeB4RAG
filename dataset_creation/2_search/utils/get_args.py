

import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset_name", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--dataset_dir", type=str, default="./", help="Directory to save and load beir datasets")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--model_dir", type=str, help="Directory to save and load models")
    parser.add_argument("--model_type", type=str, default="beir", help="beir or custom")

    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--corpus_chunk_size", default=50000, type=int, help="How many documents in one chunk,"
                                                                             "If memory is limited - make it smaller")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for indexing.")
    # parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")

    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")
    parser.add_argument("--lower_case", action="store_true", help="lowercase query and document text")
    parser.add_argument("--normalize_text", action="store_true", help="Apply function to normalize some common characters")
    parser.add_argument(
        "--little_corpus", action="store_true", help="Use only a part of corpus (for debugging locally)")

    parser.add_argument("--log_dir", type=str, default="./logs/", help="Path to the log")
    parser.add_argument("--embedding_dir", type=str, default="./embeddings/", help="Path to the log")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")
    parser.add_argument("--hostname", type=str, default='gpunode-0-11',
                        help="hostname=$(scontrol show hostnames $ SLURM_JOB_NODELIST")

    parser.add_argument("--task", type=str, default="encode", help="encode/eval or "
                        "binary_entropy/query_alteration/qgen_llm/qpp_sigma_max/qpp_nqc/qpp_smv/fusion")
    parser.add_argument("--mask_ratio", type=float, default=0.1, help="The probability of masking, "
                                                             "required for query_alteration model selection method.")
    parser.add_argument("--topk", type=int, default=10, help="How many samples to extract"
                                                             "required for query_alteration model selection method.")

    # parser.add_argument("--fake_data_dir", type=str, default="./logs/fake_data/", help="Path to the fake data")

    parser.add_argument("--fake_queries", action="store_true", help="if using real queries or fake queries")
    parser.add_argument("--fake_id_qrels", type=str, default="")
    parser.add_argument("--fake_id_queries", type=str, default="")

    parser.add_argument("--rbo_p", type=float, default=0.9, help="p parameter for rbo")

    parser.add_argument("--old_models_only",  action="store_true", )
    parser.add_argument("--special_token",  action="store_true", )

    parser.add_argument("--specific_model", default="", type=str)

    # Parameters for query generation
    parser.add_argument("--num_sample_docs", type=int, default=100)
    parser.add_argument("--num_gen_qry_per_doc", type=int, default=10)
    parser.add_argument("--openai_key", type=str, default=None)
    # add possible options
    parser.add_argument("--llm_name", type=str, help="Name of LLM for query generation. ",
                        default="flan-t5-large")

    #  Parameters for Demo
    parser.add_argument("--user_id",  type=str, default="", help="User ID. Parameter for demo")
    parser.add_argument("--job_id",  type=str, default="", help="Job ID. Parameter for demo")
    args, _ = parser.parse_known_args()

    #  CUDA_VISIBLE_DEVICE=0 python encoding_and_eval.py
    #  --dataset_name $dset
    #  --dataset_dir ../../IR_datasets/
    #  --model_dir ../../IR_DEL/IR_models/
    #  --embedding_dir ../../IR_DEL/IR_embeddings/
    #  --log_dir ../../IR_DEL/logs/
    #  --special_token
    #  --task encode
    return args