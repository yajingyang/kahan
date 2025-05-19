import os
import logging
from pathlib import Path
import json
from dotenv import load_dotenv
from utils import update_json
load_dotenv()

engine = os.getenv("OPENAI_API_ENGINE")
import re

from factscore.factscorer import FactScorer


def evaluate_generations(results_dir_str: str):
    results_dir = Path("results") / results_dir_str
    sub_results_dirs = [p for p in results_dir.iterdir() if p.is_dir()]

    generations = []
    all_numerical_results = []
    generation_ids = []

    for sub_results_dir in sub_results_dirs:
        single_result_dirs = [p for p in sub_results_dir.iterdir() if p.is_dir()]
        for cur_result_dir in single_result_dirs:
            cur_date = cur_result_dir.stem

            if not (cur_result_dir / "narrative.txt").exists():
                continue
            encodings = ['utf-8', 'latin1', 'cp1252', 'utf-16-le', 'utf-16-be']
            for encoding in encodings:
                try:            
                    with open(cur_result_dir / "narrative.txt", 'r', encoding=encoding) as f:
                        generations.append(f.read())
                        break
                except UnicodeError:
                    continue
            
            num_result_path = Path("results\metric_values") / sub_results_dir.stem / f"{cur_date}.txt"
        
            with open(num_result_path, 'r') as f:
                numerical_results = f.read()
            # numerical_result_paths = [p for p in cur_result_dir.iterdir() if 'results' in p.name and p.name.endswith('json')]
            # for num_result_path in numerical_result_paths:
            #     entity = num_result_path.stem.split('_')[-1].replace("-", " ")
            #     with open(num_result_path, 'r') as f:
            #         num_results_list = json.load(f)
            #     numerical_results += f"\nThe numerical results computed for {entity} on date {cur_date} are:\n"
            #     for x in num_results_list:
            #         metrics_str = process_metrics(x['results'])
            #         if metrics_str == "":
            #             continue
            #         numerical_results += f"{x['question']['question']}\n" if 'question' in x.keys() else ""
            #         numerical_results += metrics_str + "\n"

            all_numerical_results.append(numerical_results)
            generation_ids.append(f'{sub_results_dir.stem} {cur_date}')

    fs = FactScorer(model_name="retrieval+"+engine,
                    data_dir=".cache/factscore",
                    model_dir=".cache/factscore",
                    cache_dir=results_dir)

    out = fs.get_score(generations=generations,
                       generation_ids=generation_ids,
                        verification_context_list=all_numerical_results,
                        retrieval_method="vector",
                        vector_db_dir="finance_wiki_vectordb",
                        doc_db_path="finance_wiki.sqlite",
                        verbose=True)

    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    return out

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.CRITICAL)
    
    setup_list = [
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": False, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_gpt-4o_klg"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_touchstonegpt-7b-instruct_klg"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": False, "prompting_postfix": ""},
        # {"model_name": "touchstonegpt-7b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "touchstonegpt-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample"},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp_sample"},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot_sample"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp_sample"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": False, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_gpt-4o_klg_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_qwen2.5-7b-instruct_klg_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_group_sample"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample_energy"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample_equity"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample_energy"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_sample_equity"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_sample"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_group_sample"},
    ]


    for setup_config in setup_list:
        logging.critical(f"Model = {setup_config['model_name']}")
        if setup_config['prompting_postfix'] == "_dp":
            logging.critical(f"Setup = Direct Prompting")
        elif setup_config['prompting_postfix'] == "_cot":
            logging.critical(f"Setup = CoT")
        elif setup_config['prompting_postfix']:
            logging.critical(f"Setup = {setup_config['prompting_postfix']}")
        else:
            logging.critical(f"Setup = Entity Knowledge: {setup_config['with_entity_analysis_knowledge']}, Process Knowledge: {setup_config['with_insight_processing_knowledge']}, Narrative Knowledge: {setup_config['with_narrative_knowledge']}, Question Based Knowledge: {setup_config['question_based_insight_processing_knowledge']}")


        dir_str = f"{setup_config['model_name']}{'_entity_klg' if setup_config['with_entity_analysis_knowledge'] else  ''}{'_process_klg' if setup_config['with_insight_processing_knowledge'] else ''}{'_qst_based' if setup_config['question_based_insight_processing_knowledge']  else ''}{'_narrative_klg' if setup_config['with_narrative_knowledge'] else ''}{'_hierarchy' if setup_config['with_hierarchical_structure']  else ''}"
        dir_str += setup_config.get("prompting_postfix", "")

        cur_scores = evaluate_generations(dir_str)
        cur_scores.update(setup_config)

        update_json(cur_scores, "results/factuality_evaluation.json")