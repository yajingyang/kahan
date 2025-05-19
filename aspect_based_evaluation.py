from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass
import time
import logging
import numpy as np
from utils import get_gpt_response, update_json, store_to_json
from pathlib import Path
import pandas as pd
import random 
from functools import partial 
import json
random.seed(27)

@dataclass
class EvaluationCriteria:
    """Stores evaluation criteria and weights"""
    aspects: List[str]
    weights: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = [1.0/len(self.aspects)] * len(self.aspects)

class DnAEvaluator:     
    def __init__(self, query_llm: Callable, k: int = 3, criteria: Optional[Dict[str, List]] = None, aspect_context: Optional[str] = None, weight_context: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """Initialize DnA evaluator"""
        self.k = k  # Number of evaluation aspects to propose
        self.logger = logger or self._setup_logger()
        self.query_llm = partial(query_llm, logger=self.logger)
        if criteria:
            self.criteria = EvaluationCriteria(aspects=criteria['aspects'], weights=criteria['weights'])
        else:
            self.criteria = self.decompose_criteria(aspect_context, weight_context)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DnAEvaluator')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def decompose_criteria(self, aspect_context: str, weight_context: str) -> EvaluationCriteria:
        """Decompose evaluation criteria with context or propose new ones"""
        prompt = f"""Given the context: {aspect_context},
        Please propose three concise questions separate by new lines about whether a potential output is a good output for the given instruction. 
        Ensure these aspects orthogonal to each other. 
        Return only the questions without meta-comments."""
        
        response = self.query_llm(prompt, return_type="str")
        aspects = [aspect.strip() for aspect in response.split('\n') if aspect.strip()][:self.k]
        weights = self.get_weights(weight_context, aspects)

        store_to_json({'aspects': aspects, 'weights': weights}, "results/evaluation_criteria.json")

        return EvaluationCriteria(aspects=aspects, weights=weights)

    def get_weights(self, context: str, aspects: List[str]) -> List[float]:
        aspects_str = '\n'.join(aspects)
        """Determine weights for different aspects based on context"""
        prompt = f"""Given context: {context}
        Please propose respective importance weightage for three aspects in evaluating the summary:
        Aspects: 
        {aspects_str}
        
        Requirements:
        1) The weightages should be in decimal values form and sum up to 1; 
        2) You should directly give the weightages without any other words; 
        3) You should give weightages in the same line, separated by space.
        """
        
        response = self.query_llm(prompt, return_type="str")
        try:
            weights = list(map(float, response.split()))
            if len(weights) != len(aspects) or not np.isclose(sum(weights), 1):
                raise ValueError
            return weights
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid weights received: {response}")
            return [1/len(aspects)] * len(aspects)

    def score_candidates(self, context: str, evaluation_candidates: List[str], aspect: str, object_name: str = "responses") -> Tuple[float, float]:
        """Score candidates on a specific aspect with context"""

        indices = list(range(len(evaluation_candidates)))
        random.shuffle(indices)
        restore_map = sorted(zip(indices, range(len(evaluation_candidates))))
        shuffled = [evaluation_candidates[i] for i in indices]

        candidates_str = "\n".join([f"{object_name} {i+1}: {candidate}" for i, candidate in enumerate(shuffled)])
        prompt = f"""Context: {context}
        
        Compare the following {object_name} on the following aspects:
        Aspect: 
        {aspect}
        
        {object_name}:
        {candidates_str}
        
        Rate each on scale 0-10 considering the given context. 
        Requirements:
        1) The score should be in integer values form from 0 to 10; 
        2) You should directly give the scores without any other words; 
        3) You should give scores in the same line, separated by space.
        3) You should give scores following the order of their corresponding {object_name}."""
        
        response = self.query_llm(prompt, return_type="str")
        try:
            scores = list(map(float, response.split()))
            if len(scores) != len(evaluation_candidates) or not all([(s >=0 and s <= 10) for s in scores]):
                raise ValueError
            scores_restored_order = [scores[i] for _, i in restore_map]
            return scores_restored_order
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid score format received: {response}")
            return []

    def aggregate_scores(self, scores: List[List[float]]) -> List[float]:
        """Compute weighted sum of aspect scores"""
        agg_scores = []
        for i in range(len(scores[0])):
            candidate_score = sum(aspect_scores[i] * weight for aspect_scores, weight in zip(scores, self.criteria.weights))
            agg_scores.append(candidate_score)
        return agg_scores
    
    def setup_evaluation(self, context: str):
        self.criteria = self.decompose_criteria(context)


    def evaluate(self, context: str,  candidates: List[str]) -> Tuple[List[float], int]:
        """Complete evaluation pipeline"""
        all_scores = []
        for aspect in self.criteria.aspects:
            while True:
                scores = self.score_candidates(context, candidates, aspect, "financial market reports")
                if scores:
                    break
            all_scores.append(scores)

        final_scores = self.aggregate_scores(all_scores)
        all_scores_by_candidates = [[x[i] for x in all_scores] for i in range(len(candidates))]
        best_response_idx = np.argmax(final_scores)
        return final_scores, all_scores_by_candidates, best_response_idx
    

if __name__ == "__main__": 
    criteria_aspect_context = "\nYou need financial reports that help you understand market conditions and trends, and make investment decisions."
    criteria_weight_context = "\nYou need financial reports that help you understand market conditions and trends, and make investment decisions."
    
    with open("results\evaluation_criteria_final.json", 'r') as f:
        criterias = json.load(f)
    
    evaluator = DnAEvaluator(query_llm=get_gpt_response, criteria=criterias)

    score_context = "\nYou are evaluating financial market reports for their effectiveness in explaining market conditions and trends. The report should help you understand what happened in the markets and why it matters."

    setup_list = [
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": False, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_gpt-4o_klg"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_qwen2.5-7b-instruct_klg"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": False, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_group"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise"},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_group"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
    ]

    fname_postfix = f"{setup_list[0]['model_name']}"
    generation_dir = Path('results')
    results = []
    timestamp_postfix = f"_{time.time()}"
    data_dir = Path("data\datatales")
    market_dir_list = [x for x in data_dir.iterdir() if x.is_dir()]
    for market_dir in market_dir_list:
        market = market_dir.name
        data_path_list = [x for x in (market_dir / "test").iterdir() if x.is_file() and x.suffix == '.csv']
        knowledge_dir = f"results\knowledge\{market}"
        for data_path in data_path_list:
            report_date =  data_path.stem
            cur_candidates = []
            cur_results = []
            for setup_config in setup_list:
                dir_str = f"{setup_config['model_name']}{'_entity_klg' if setup_config['with_entity_analysis_knowledge'] else  ''}{'_process_klg' if setup_config['with_insight_processing_knowledge'] else ''}{'_qst_based' if setup_config['question_based_insight_processing_knowledge']  else ''}{'_narrative_klg' if setup_config['with_narrative_knowledge'] else ''}{'_hierarchy' if setup_config['with_hierarchical_structure']  else ''}"
                dir_str += setup_config.get("prompting_postfix", "")
                generation_path = str(generation_dir / dir_str / market / report_date / "narrative.txt")
                try:
                    with open(generation_path, 'r') as f:
                        generation = f.read()
                    cur_candidates.append(generation)

                    cur_results_i = {'market': market, 'date': report_date, 'generation': generation}
                    cur_results_i.update(setup_config)
                    cur_results.append(cur_results_i)
                except:
                    print(f"file {generation_path} not exist")
            
            if len(cur_results) != len(setup_list):
                print(f"missing files, skipping evaluation for {market} {report_date}")
                continue
            
            scores, detail_scores, best_candidate_idx = evaluator.evaluate(candidates=cur_candidates, context=score_context)
            for i, x in enumerate(cur_results):
                aspect_scores_dict = {f'aspect_{j}': x for j, x in enumerate(detail_scores[i])}
                x.update({'score': scores[i], 'is_best': best_candidate_idx==i})
                x.update(aspect_scores_dict)
            results.extend(cur_results)
            update_json(cur_results, f"results/evaluation_results_randomized_{fname_postfix}{timestamp_postfix}.json")
    df = pd.DataFrame(results)
    df.to_csv(f"results/evaluation_results_randomized_{fname_postfix}{timestamp_postfix}.tsv", sep="\t", index=False)
    agg_results = df.groupby(by=list(setup_list[0].keys()))[["score", "aspect_0", "aspect_1", "aspect_2"]].mean().reset_index()
    agg_results.to_csv(f"results/evaluation_results_agg_randomized_{fname_postfix}{timestamp_postfix}.csv", index=False)

    setup_list = [
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": False, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_gpt-4o_klg"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_qwen2.5-7b-instruct_klg"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": False, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_group"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise"},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_group"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
    ]

    fname_postfix = f"{setup_list[0]['model_name']}"
    generation_dir = Path('results')
    results = []
    timestamp_postfix = f"_{time.time()}"
    data_dir = Path("data\datatales")
    market_dir_list = [x for x in data_dir.iterdir() if x.is_dir()]
    for market_dir in market_dir_list:
        market = market_dir.name
        data_path_list = [x for x in (market_dir / "test").iterdir() if x.is_file() and x.suffix == '.csv']
        knowledge_dir = f"results\knowledge\{market}"
        for data_path in data_path_list:
            report_date =  data_path.stem
            cur_candidates = []
            cur_results = []
            for setup_config in setup_list:
                dir_str = f"{setup_config['model_name']}{'_entity_klg' if setup_config['with_entity_analysis_knowledge'] else  ''}{'_process_klg' if setup_config['with_insight_processing_knowledge'] else ''}{'_qst_based' if setup_config['question_based_insight_processing_knowledge']  else ''}{'_narrative_klg' if setup_config['with_narrative_knowledge'] else ''}{'_hierarchy' if setup_config['with_hierarchical_structure']  else ''}"
                dir_str += setup_config.get("prompting_postfix", "")
                generation_path = str(generation_dir / dir_str / market / report_date / "narrative.txt")
                try:
                    with open(generation_path, 'r') as f:
                        generation = f.read()
                    cur_candidates.append(generation)

                    cur_results_i = {'market': market, 'date': report_date, 'generation': generation}
                    cur_results_i.update(setup_config)
                    cur_results.append(cur_results_i)
                except:
                    print(f"file {generation_path} not exist")
            
            if len(cur_results) != len(setup_list):
                print(f"missing files, skipping evaluation for {market} {report_date}")
                continue
            
            scores, detail_scores, best_candidate_idx = evaluator.evaluate(candidates=cur_candidates, context=score_context)
            for i, x in enumerate(cur_results):
                aspect_scores_dict = {f'aspect_{j}': x for j, x in enumerate(detail_scores[i])}
                x.update({'score': scores[i], 'is_best': best_candidate_idx==i})
                x.update(aspect_scores_dict)
            results.extend(cur_results)
            update_json(cur_results, f"results/evaluation_results_randomized_{fname_postfix}{timestamp_postfix}.json")
    df = pd.DataFrame(results)
    df.to_csv(f"results/evaluation_results_randomized_{fname_postfix}{timestamp_postfix}.tsv", sep="\t", index=False)
    agg_results = df.groupby(by=list(setup_list[0].keys()))[["score", "aspect_0", "aspect_1", "aspect_2"]].mean().reset_index()
    agg_results.to_csv(f"results/evaluation_results_agg_randomized_{fname_postfix}{timestamp_postfix}.csv", index=False)

    setup_list = [
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": False, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_gpt-4o_klg"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_qwen2.5-7b-instruct_klg"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": False, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise"},
        # {"model_name": "llama-3.1-8b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_group"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "gpt-4o", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": False, "with_narrative_knowledge": True, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise"},
        {"model_name": "gpt-4o", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": "_pairwise_group"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": True, "with_insight_processing_knowledge": True, "with_narrative_knowledge": True, "with_hierarchical_structure": True, "question_based_insight_processing_knowledge": True, "prompting_postfix": ""},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_dp"},
        # {"model_name": "qwen2.5-7b-instruct", "with_entity_analysis_knowledge": False, "with_insight_processing_knowledge": False, "with_narrative_knowledge": False, "with_hierarchical_structure": False, "question_based_insight_processing_knowledge": False, "prompting_postfix": "_cot"},
    ]

    fname_postfix = f"{setup_list[0]['model_name']}"
    generation_dir = Path('results')
    results = []
    timestamp_postfix = f"_{time.time()}"
    data_dir = Path("data\datatales")
    market_dir_list = [x for x in data_dir.iterdir() if x.is_dir()]
    for market_dir in market_dir_list:
        market = market_dir.name
        data_path_list = [x for x in (market_dir / "test").iterdir() if x.is_file() and x.suffix == '.csv']
        knowledge_dir = f"results\knowledge\{market}"
        for data_path in data_path_list:
            report_date =  data_path.stem
            cur_candidates = []
            cur_results = []
            for setup_config in setup_list:
                dir_str = f"{setup_config['model_name']}{'_entity_klg' if setup_config['with_entity_analysis_knowledge'] else  ''}{'_process_klg' if setup_config['with_insight_processing_knowledge'] else ''}{'_qst_based' if setup_config['question_based_insight_processing_knowledge']  else ''}{'_narrative_klg' if setup_config['with_narrative_knowledge'] else ''}{'_hierarchy' if setup_config['with_hierarchical_structure']  else ''}"
                dir_str += setup_config.get("prompting_postfix", "")
                generation_path = str(generation_dir / dir_str / market / report_date / "narrative.txt")
                try:
                    with open(generation_path, 'r') as f:
                        generation = f.read()
                    cur_candidates.append(generation)

                    cur_results_i = {'market': market, 'date': report_date, 'generation': generation}
                    cur_results_i.update(setup_config)
                    cur_results.append(cur_results_i)
                except:
                    print(f"file {generation_path} not exist")
            
            if len(cur_results) != len(setup_list):
                print(f"missing files, skipping evaluation for {market} {report_date}")
                continue
            
            scores, detail_scores, best_candidate_idx = evaluator.evaluate(candidates=cur_candidates, context=score_context)
            for i, x in enumerate(cur_results):
                aspect_scores_dict = {f'aspect_{j}': x for j, x in enumerate(detail_scores[i])}
                x.update({'score': scores[i], 'is_best': best_candidate_idx==i})
                x.update(aspect_scores_dict)
            results.extend(cur_results)
            update_json(cur_results, f"results/evaluation_results_randomized_{fname_postfix}{timestamp_postfix}.json")
    df = pd.DataFrame(results)
    df.to_csv(f"results/evaluation_results_randomized_{fname_postfix}{timestamp_postfix}.tsv", sep="\t", index=False)
    agg_results = df.groupby(by=list(setup_list[0].keys()))[["score", "aspect_0", "aspect_1", "aspect_2"]].mean().reset_index()
    agg_results.to_csv(f"results/evaluation_results_agg_randomized_{fname_postfix}{timestamp_postfix}.csv", index=False)