import os
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime, date
import logging
import pandas as pd
from enum import Enum
import json
import numpy as np
from utils import get_gpt_response, store_to_json, update_json, get_dtypes_dict, remove_month_labels, count_numeric_value, \
    process_string, load_pipeline, get_llama_response, load_json_from_path, extract_and_fix_json, extract_and_validate_code, extract_text, setup_logger
from pathlib import Path
import os
import shutil
from collections import defaultdict
import argparse
import subprocess
from functools import partial 
import ast
import re


class InsightType(Enum):
    ENTITY = "entity"
    PAIRWISE = "pairwise" 
    GROUP = "group"
    OVERALL = "overall"

class NarrationGenerator:
    """Main class for generating data narratives using LLM"""
    def __init__(self,
                 data_file: str,
                 domain: str,
                 task: str,
                 market: str, 
                 report_date: Optional[datetime] = None,
                 entity_col: Optional[str] = "Product Name",
                 query_llm: Optional[Callable] = get_gpt_response,  
                 with_entity_analysis_knowledge: bool = True,
                 with_insight_processing_knowledge: bool = True,
                 with_narrative_knowledge: bool = True,
                 with_hierarchical_structure: List[str] = None,
                 question_based_insight_processing_knowledge: bool = True,
                 prompting: str = "",
                 entity_level_summary: bool = True,
                 knowledge_dir: str = None,
                 output_dir: Optional[str] = None,
                 reload_insight_dir: Optional[str] = None,
                 reuse_insight_group: Optional[List[str]] = None,
                 recreate_code: Optional[bool] = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize narration generator
        
        Args:
            data_file: Path to data file
            domain: Domain for analysis (e.g., 'finance', 'retail')
            report_date: Target date for narrative
            insight_types: Types of insights to generate
            query_llm:  Function to get LLM response
            with_question_generation: Whether to use LLM for question generation
            with_hierarchical_structure: Whether to use hierarchical processing
            with_knowledge_base: Whether to use explicit knowledge base
        """
        self.data_file = data_file
        self.domain = domain
        self.market = market
        self.task = task
        self.report_date = report_date
        self.with_entity_analysis_knowledge = with_entity_analysis_knowledge
        self.with_insight_processing_knowledge = with_insight_processing_knowledge
        self.with_narrative_knowledge = with_narrative_knowledge
        self.with_hierarchical_structure = with_hierarchical_structure
        self.question_based_insight_processing_knowledge = question_based_insight_processing_knowledge
        self.prompting = prompting
        self.entity_level_summary = entity_level_summary
        self.query_llm = query_llm
        self.reuse_insight_group = reuse_insight_group
        self.reload_insight_dir = reload_insight_dir
        self.recreate_code = recreate_code
        self.logger = logger
        self._setup_dir(knowledge_dir, output_dir, reload_insight_dir)
        self.entities, self.data = self._load_data(entity_col)
        self.knowledge_base = self._load_knowledge_base()
        self.data_schema = json.dumps(get_dtypes_dict(self.data))

    def _load_data(self, entity_col) -> pd.DataFrame:
        """Load and process input data"""
        try:
            df = pd.read_csv(self.data_file)
            entities = df[entity_col].unique().tolist()
            if self.report_date is None:
                date_col = [col for col in df.columns if 'date' in col.lower()][0]
                self.last_data_date = pd.to_datetime(df[date_col]).max()
            return entities, df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_knowledge_base(self) -> Dict:
        """Load domain knowledge base"""
        if self.with_entity_analysis_knowledge or self.with_insight_processing_knowledge or self.with_narrative_knowledge:
            knowledge_file = str(self.knowledge_dir / f"complete_knowledge_base_with{'out' if not self.question_based_insight_processing_knowledge else ''}_question.json")
            with open(knowledge_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def _setup_dir(self, knowledge_dir: str, output_dir: str, reload_insight_dir: str):
        """Setup directory for knowledge base and output"""

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True) 
        self.logger.info(f"output_dir: {self.output_dir}")

        if  knowledge_dir:
            self.knowledge_dir = Path(knowledge_dir)
            self.logger.info(f"Loading knowledge from {knowledge_dir}")
        else:
            self.knowledge_dir = None
        
        if reload_insight_dir:
            self.reload_insight_dir = Path(reload_insight_dir)
            self.logger.info(f"Loading insight from {reload_insight_dir}")
        else:
            self.reload_insight_dir = None


    def load_entity_insights(self, entity_insight_dir):
        """Load single entity insights from directory"""
        entity_insight_dir_path = Path(entity_insight_dir)
        insights = []
        for insight_path in entity_insight_dir_path.rglob("insight_entity*.json"):
            with open(insight_path, 'r') as f:
                insights.append(json.load(f))
        return insights

    def process_entity_insight_question(self, entity_insight_question: Dict[str, str], entity: str) -> Dict:
        """Process a single question to generate insight"""
        insight_type = "_".join(entity_insight_question["insight_type"].replace('/', '_').split())
        # question_text = json.dumps(entity_insight_question)

        entity_str = process_string(entity)
        
        fname = f"insight_entity_{entity_str}_{insight_type}.json"

        if self.reload_insight_dir and 'entity' in self.reuse_insight_group:
            insight_path = self.reload_insight_dir / fname
            if insight_path.exists():
                self.logger.info(f"Reload insight for {entity} {insight_type} from {str(insight_path)}...")
                insight = load_json_from_path(insight_path)
            return insight

        insight_path = self.output_dir / fname

        precomputation_code_path = self.get_precomputation_code_path(insight_type)
        # if not precomputation_code_path.exists():
        #     self.generate_precomputation_code(question_text, precomputation_code_path)

        precomputation_result_path = self.output_dir / f"insight_entity_precompute_results_{entity_str}.json"
        raw_precomputation_path = self.output_dir / f"insight_entity_precompute_results_{entity_str}_{insight_type}.text"
        precomputation_results = self.execute_computation(precomputation_code_path, self.data_file, entity, self.report_date, raw_precomputation_path)

        # retry_counts = 0
        # while retry_counts < 5:
        #     try:
        #         precomputation_results = self.execute_computation(precomputation_code_path, self.data_file, entity, self.report_date)
        #         assert len(precomputation_results) < 500, "Results saved is too long, only include the values needed in describing the insights."
        #         assert len(precomputation_results) > 10, "Results saved is too short, code invalid or not executed properly."
        #         break
        #     except Exception as e:
        #         self.logger.info(f"Error running {precomputation_code_path} {str(e)}")
        #         self.logger.info(f"Regenerating code...")
        #         self.debug_generate_precomputation_code(question_text, precomputation_code_path, error=str(e))
        #         retry_counts += 1

        update_json({'insight_type': insight_type, 'results': precomputation_results}, precomputation_result_path)

        insight =  self.interpret_entity_insight(entity, precomputation_results)
        store_to_json(insight, str(insight_path))
        return insight

    def process_entity_insight_all_question(self, entity_insight_questions: List[Dict[str, str]], entity: str) -> Dict:
        """Process a single question to generate insight"""
        entity_str = "-".join(remove_month_labels(entity).split())

        fname = f"insight_entity_{entity_str}.json"

        if self.reload_insight_dir and 'entity' in self.reuse_insight_group:
            insight_path = self.reload_insight_dir / fname
            self.logger.info(f"Reload insight for {entity} from {str(insight_path)}...")
            if insight_path.exists():
                self.logger.info(f"Reload insight for {entity} from {str(insight_path)}...")
                insight = load_json_from_path(insight_path)
                return insight
            else:
                self.logger.info(f"Insight file not exists, regenerate insight for {entity}")

        insight_path = self.output_dir / fname

        computation_results =  []
        for question in entity_insight_questions:
            insight_type = "_".join(question["insight_type"].replace('/', '_').split())
            # question_text = json.dumps(question)

            self.logger.info(f"Getting entity insight insight for {entity} {insight_type}...")
            precomputation_code_path = self.get_precomputation_code_path(insight_type)
            # if not precomputation_code_path.exists():
            #     self.generate_precomputation_code(question_text, precomputation_code_path)

            raw_insight_precomputation_result_path = self.output_dir / f"insight_entity_precompute_results_{entity_str}_{insight_type}.txt"
            try:
                precomputation_results = self.execute_computation(precomputation_code_path, self.data_file, entity, self.report_date, raw_insight_precomputation_result_path)
            except Exception as e:
                self.logger.info(f"Error getting precomputation result for {entity} {insight_type}: {e}")
                self.logger.info(f"Skipping...")
                continue

            # retry_counts = 0
            # while retry_counts < 5:
            #     try:
            #         precomputation_results = self.execute_computation(precomputation_code_path, self.data_file, entity, self.report_date, raw_precomputation_result_path)
            #         if self.with_insight_processing_knowledge:
            #             result_length_limit = 500
            #         else:
            #             result_length_limit = 2000
            #         assert len(precomputation_results) < result_length_limit, "Results saved is too long, only include the values needed in describing the insights."
            #         assert len(precomputation_results) > 10, "Results saved is too short, code invalid or not executed properly."
            #         break
            #     except Exception as e:
            #         if retry_counts < 3:
            #             self.logger.info(f"Error running {precomputation_code_path} {str(e)}")
            #             self.logger.info(f"Regenerating code...")
            #             self.debug_generate_precomputation_code(question_text, precomputation_code_path, error=str(e))
            #         else:
            #             self.generate_precomputation_code(question_text, precomputation_code_path)
            #         retry_counts += 1

            computation_results.append({'insight_type': insight_type, 'question': question, 'results': precomputation_results}) 

        computation_results_path = self.output_dir / f"insight_entity_precompute_results_{entity_str}.json"
        store_to_json(computation_results, str(computation_results_path))

        entity_insight =  self.interpret_all_entity_insight(entity, computation_results)
        store_to_json(entity_insight, str(insight_path))
        return entity_insight

    def generate_precomputation_code(self, question: str, code_path: str, insight_type: str) -> Dict:
        """Generate python code to generate precompute metrics"""
        # Generate computation code
        if self.prompting == 'cot':
            code_prompt = f"""
            Think step by step about the calculations needed, then write the Python code.

            Question: {question}

            Follow each step below to explain your thinking process, then provide the final code.
            Write all text in plain format without any markdown, formatting, or special characters.
            
            Step 1: Core Analysis Objective
            Print "OBJECTIVE_ANALYSIS_START"
            Think about and explain:
            - The specific measurements or comparisons needed
            - The relevant timeframe considerations
            - The required level of analysis
            Do not list these questions - instead explain your actual analysis for this specific problem.
            Print "OBJECTIVE_ANALYSIS_END"

            Step 2: Metric Selection
            Print "METRIC_ANALYSIS_START"
            For each metric you identify as relevant:
            - Explain why you chose this specific metric
            - Describe its relevance to the question
            - Specify the appropriate unit
            - Explain how it should be interpreted
            Do not list these points - instead provide your actual metric selection reasoning.
            Print "METRIC_ANALYSIS_END"

            Step 3: Calculation Planning
            Print "CALCULATION_ANALYSIS_START"
            For each selected metric, explain:
            - The specific data transformations you'll use
            - Your detailed calculation approach
            - Any statistical methods you'll apply
            - How you'll handle edge cases
            Provide your actual calculation planning, not these prompts.
            Print "CALCULATION_ANALYSIS_END"

            Step 4: Implementation Strategy
            Print "IMPLEMENTATION_ANALYSIS_START"
            Explain your specific plans for:
            - Data processing approach
            - Data type handling
            - Missing data strategy
            - Output structure
            Write your actual implementation strategy, not these guidelines.
            Print "IMPLEMENTATION_ANALYSIS_END"

            Requirements:
            1. Read data into pandas DataFrame from argparser argument --data_path with type str
            2. The entity name is provided as argparser argument --product_name with type str
            3. The date of interest is provided as argparser argument --date with type str
            4. Process the data to the data types specified below, be careful with the data types during operation
            5. Handle missing/invalid values in the data
            6. For time-series calculations (e.g., moving averages):
            - Maintain complete historical data until final calculation
            - Only filter by date after computing time-dependent metrics
            - Ensure proper handling of lookback periods
            7. Print results (e.g., sma-20) and values (closing price) required in a structured format:
                - For each item:
                    - Print "METRIC:" followed by the metric name
                    - Print "VALUE:" followed by the calculated value
                    - Print "UNIT:" followed by the unit (if applicable)
                    - Print "TYPE:" followed by the data type of the result
                    - Example:
                        METRIC:monthly_return\tVALUE:0.0234\tUNIT:percent\tTYPE:float
            8. Use only the following columns from data:
            {self.data_schema}

            After completing the analysis, print "FINAL CODE:" and provide the implementation code that:
            1. Computes all identified relevant metrics
            2. Handles data preprocessing appropriately
            3. Implements the calculations efficiently
            4. Returns results in the specified format

            Example of expected output format:
            OBJECTIVE_ANALYSIS_START
            For this problem, we need to calculate the 20-day moving average of closing prices. This requires maintaining the complete price history...
            OBJECTIVE_ANALYSIS_END

            [Other analysis sections...]

            FINAL CODE:
            [Your actual Python code here]

            The output should begin with your explicit reasoning process in each analysis section, followed by the code. Do not include the guiding questions in your output.
            """
            raw_result = self.query_llm(code_prompt, return_type="str")
            raw_code_path = self.get_precomputation_code_path(insight_type, prefix="raw_")
            with open(raw_code_path, 'w') as f:
                f.write(raw_result)

            computation_code, validate_code_success, validate_msg = extract_and_validate_code(raw_result, code_mark="FINAL CODE:")
        else:
            code_prompt = f"""
            Write Python code to perform the required calculations and save results.

            Question: {question}

            Requirements:
            1. Read data into pandas DataFrame from argparser argument --data_path with type str
            2. The entity name is provided as argparser argument --product_name with type str
            3. The date of interest is provided as argparser argument --date with type str
            4. Process the data to the data types specified below, be careful with the data types during operation
            5. Handle missing/invalid values in the data
            6. For time-series calculations (e.g., moving averages):
            - Maintain complete historical data until final calculation
            - Only filter by date after computing time-dependent metrics
            - Ensure proper handling of lookback periods
            7. Print results (e.g., sma-20) and values (closing price) required in a structured format:
                - For each item:
                    - Print "METRIC:" followed by the metric name
                    - Print "VALUE:" followed by the calculated value
                    - Print "UNIT:" followed by the unit (if applicable)
                    - Print "TYPE:" followed by the data type of the result
                    - Example:
                        METRIC:monthly_return\tVALUE:0.0234\tUNIT:percent\tTYPE:float
            8. Use only the following columns from data:
            {self.data_schema}
            Return the only the python code without meta-commentary, introductions, or language specification
            """
            computation_code = self.query_llm(code_prompt, return_type="str")
            computation_code, validate_code_success, validate_msg = extract_and_validate_code(computation_code, code_mark="")

        assert validate_code_success, validate_msg
        code_path = self.get_precomputation_code_path(insight_type=insight_type)
        self.logger.info(f"Saving code to {code_path}...")
        with open(code_path, 'w') as f:
            f.write(computation_code)

    def debug_generate_precomputation_code(self, question: str, code_path: str, error: Optional[str] = None) -> Dict:
        """Generate python code to generate precompute metrics"""
        with open(code_path, 'r') as f:
            cur_code = f.read()

        sample_data = self.data.to_string(max_rows=3)

        code_prompt = f"""
        Debug and fix the Python code for the required calculations.

        Question: {question}

        Current Code:
        {cur_code}

        Sample Data (First 3 rows):
        {sample_data}

        Error Message:
        {error}

        Requirements:
        1. Read data into pandas DataFrame from argparser argument --data_path with type str
        2. The entity name is provided as argparser argument --product_name with type str
        3. The date of interest is provided as argparser argument --date with type str
        4. Process the data to the data types specified below, be careful with the data types during operation
        5. Handle missing/invalid values in the data
        6. For time-series calculations (e.g., moving averages):
            - Maintain complete historical data until final calculation
            - Only filter by date after computing time-dependent metrics
            - Ensure proper handling of lookback periods
        7. Print results (e.g., sma-20) and values (closing price) required in a structured format:
            - For each item:
                - Print "METRIC:" followed by the metric name
                - Print "VALUE:" followed by the calculated value
                - Print "UNIT:" followed by the unit (if applicable)
                - Print "TYPE:" followed by the data type of the result
                - Example:
                    METRIC:monthly_return\tVALUE:0.0234\tUNIT:percent\tTYPE:float
        8. Use only the following columns from data:
        {self.data_schema}

        Debugging Guidelines:
        1. Check for data type mismatches in operations
        2. Verify column names exist in the sample data
        3. Ensure date formatting is consistent
        4. Handle potential null/missing values
        5. Validate mathematical operations (e.g., division by zero)
        6. Confirm appropriate data filtering for the specific entity

        Return only the fixed python code without explanations or comments
        """
        computation_code = self.query_llm(code_prompt, return_type="str")
        computation_code, validate_code_success, validate_msg = extract_and_validate_code(computation_code, code_mark="")
        assert validate_code_success, validate_msg

        self.logger.info(f"Saving code to {code_path}...")
        with open(code_path, 'w') as f:
            f.write(computation_code)

    def execute_computation(self, code_path: str, data_path: str, entity: str, report_date: str, raw_results_path: str) -> Dict:
        """Execute generated computation code"""

        if len(entity.split()) > 1:
            entity = f'\"{entity}\"'

        absolute_path = str(Path(data_path).absolute())

        command = f"python {code_path} --data_path {absolute_path} --product_name {entity} --date {report_date}"
        self.logger.info(f"Running command: {command}...")

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with error: {result.stderr}")
        
        with open(raw_results_path, 'w') as f:
            f.write(result.stdout)

        return result.stdout

    def interpret_entity_insight(self, entity_insight_question: Dict, entity: str, precomputation_results: str) -> Dict:
        interpretation_prompt = f"""
        As a {self.domain} expert targeting to {self.task}, interpret the following computation results and rate its significance:
        
        Question: {entity_insight_question['question']}
        Entity: {entity}
        Results: 
        {precomputation_results}
        
        Provide:
        1. A clear, professional interpretation explaining the findings and their significance
        2. A significance score (0-1) based on:
            - Impact/importance of the finding
            - How unusual/noteworthy it is
            - Actionable implications
            - Relevance to current context
        
        Structure output as JSON:
        {{
            "interpretation": "<interpretation text>",
            "significance_score": <0-1 score>,
            "reasoning": "<explanation of score>"
        }}

        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "{{") and excluding concluding remarks or follow-up suggestions (i.e., ends with "}}")
        """
        interpretation_data = self.query_llm(interpretation_prompt)

        insight = {
            'type': entity_insight_question['insight_type'],
            'entity': entity,
            'question': entity_insight_question['question'],
            'computation': precomputation_results,
            'interpretation': interpretation_data['interpretation'],
            'significance_score': interpretation_data['significance_score'],
            'score_reasoning': interpretation_data['reasoning']
        }
        return insight

    def interpret_all_entity_insight(self, entity: str, question_with_result_list: List[Dict]) -> Dict:
        if self.prompting == "cot":
            interpretation_prompt = f"""
            As a {self.domain} expert targeting to {self.task}, analyze these computation results step by step:

            Entity: {entity}
            Results: {[{"Question": question_with_result["question"], "Results": question_with_result["results"]} for question_with_result in question_with_result_list]}

            Follow each step below to explain your thinking process, then provide the final JSON output.
            Write all text in plain format without any markdown, formatting, or special characters.

            Step 1: Individual Question Analysis
            For each question-result pair:
            Print "ANALYSIS_START: <question text>"
            Analyze and explain:
            - The specific significance of this metric in {self.domain}
            - Your comparison with relevant benchmarks and standards
            - Your assessment of causes and implications
            - Your evaluation of the finding's importance
            - Your identification of related metrics
            Do not list these points - instead provide your actual analysis for this specific question-result pair.
            Print "ANALYSIS_END"

            Step 2: Cross-Question Analysis
            Print "CROSS_ANALYSIS_START"
            Analyze and explain:
            - The specific patterns you observe across metrics
            - Any contradictions or supporting relationships you find
            - The interactions between different metrics
            - The temporal relationships in the findings
            Provide your actual cross-metric analysis, not these prompts.
            Print "CROSS_ANALYSIS_END"

            Step 3: Significance Assessment
            Print "SIGNIFICANCE_ANALYSIS_START"
            For each key finding, explain:
            - Your assessment of statistical significance
            - The specific business/domain impact
            - How it compares in importance to other findings
            - Your evaluation of short-term vs long-term effects
            - Your confidence level in the interpretation
            Write your actual significance assessment, not these guidelines.
            Print "SIGNIFICANCE_ANALYSIS_END"

            Step 4: Overall Synthesis
            Print "SYNTHESIS_START"
            Synthesize and explain:
            - The critical findings and their interconnections
            - The coherent narrative that emerges
            - The broader implications you've identified
            - Any important gaps or limitations
            Provide your actual synthesis, not these prompts.
            Print "SYNTHESIS_END"

            After completing the analysis, print "FINAL JSON:" and provide the final output as JSON object with question_interpretations and overall_interpretation as shown in the example:

            Example of expected output format:
            ANALYSIS_START: What is the trend in monthly revenue?
            The 15% increase in monthly revenue represents a significant departure from the industry average of 5%. This growth rate puts the entity in the top quartile of performers...
            ANALYSIS_END

            [Other analysis sections with actual insights...]

            FINAL JSON:
            {{
                "question_interpretations": [
                    {{
                        "question": "<question text>",
                        "interpretation": "<interpretation text>",
                        "significance_score": <0-1 score>
                    }},
                    ...
                ],
                "overall_interpretation": {{
                    "summary": "<synthesis of key findings>",
                    "significance_score": <0-1 composite score>,
                    "reasoning": "<explanation of overall significance>"
                }}
            }}


            The output should begin with your explicit reasoning process in each analysis section, followed by the JSON. Do not include the guiding questions in your output.
            Ensure all text strings in the JSON are enclosed with double quotes.
            """
            raw_results = self.query_llm(interpretation_prompt, return_type='str')
            with open(self.output_dir / f"raw_interpret_{entity}.txt", 'w') as f:
                f.write(raw_results)

            interpretation_data, _, _ = extract_and_fix_json(raw_results, json_mark="FINAL JSON:")
        else:
            interpretation_prompt = f"""
                As a {self.domain} expert targeting to {self.task}, interpret these computation results and rate their significance:

                Entity: {entity}
                Results: {[{"Question": question_with_result["question"], "Results": question_with_result["results"]} for question_with_result in question_with_result_list]}

                For each question-result pair, provide interpretation points.
                
                Then synthesize an overall interpretation that:
                1. Summarizes key findings across all questions
                2. Identifies patterns and relationships
                3. Explains collective significance and implications
                4. Include necessary details and numerical values
                
                Structure output as JSON:
                {{
                    "question_interpretations": [
                        {{
                            "question": "<question text>",
                            "interpretation": "<interpretation text>",
                            "significance_score": <0-1 score>
                        }},
                        ...
                    ],
                    "overall_interpretation": {{
                        "summary": "<synthesis of key findings>",
                        "significance_score": <0-1 composite score>,
                        "reasoning": "<explanation of overall significance>"
                    }}
                }}

                Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "{{") and excluding concluding remarks or follow-up suggestions (i.e., ends with "}}")
                Ensure that each text string in the JSON object are enclosed with double quote "".
                """
            interpretation_data = self.query_llm(interpretation_prompt)

        insight = {
            'type': "all",
            'entity': entity,
            # 'question': entity_insight_question['question'],
            # 'computation': precomputation_results,
            'interpretation': interpretation_data['overall_interpretation']['summary'],
            'significance_score': interpretation_data['overall_interpretation']['significance_score'],
            'score_reasoning': interpretation_data['overall_interpretation']['reasoning'],
            'individual_insights': interpretation_data['question_interpretations'] 
        }
        return insight

    def get_entity_insights(self, entity: str) -> List[Dict]:
        """Extract basic insights for an entity"""
        insights = []
        
        if self.with_entity_analysis_knowledge:
            question_path = self.knowledge_dir / "entity_questions.json"
            questions = self._load_json_data(question_path)
            assert questions is not None
        else:
            questions = [
                {
                    "insight_type": "general",
                    "question": f"On the date of interest, for the entity of interest, analyze and find insights for {self.task} using derived values from the data"
                },
            ]
        
        if self.entity_level_summary:
            insight = self.process_entity_insight_all_question(questions, entity)
            insights.append(insight)
        else:
            for question in questions:
                insight = self.process_entity_insight_question(question, entity)
                insights.append(insight)

        return insights

    def _load_json_data(self, json_path: Path):
        json_data = None
        if json_path.exists():
            with open(json_path, 'r') as f:
                content = json.load(f)
            if content:
                json_data = content
        return json_data

    def process_pairwise_insights(self, entity_insights: List[Dict], top_k: int = 5) -> List[Dict]:
        """Process pairwise re lationships between insights"""

        if self.with_insight_processing_knowledge:
            fname = "insight_pairwise_knowledge_based.json"
        else:
            fname = "insight_pairwise_direct.json"

        if self.reload_insight_dir and 'pairwise' in self.reuse_insight_group:
            insight_path = self.reload_insight_dir / fname
            if insight_path.exists():
                self.logger.info(f"Reload pairwise insight from {str(insight_path)}...")
                pairwise_insights = load_json_from_path(insight_path)
                return pairwise_insights
            else:
                self.logger.info(f"Insight file not exists, regenerating pairwise insight...")
        else:
            self.logger.info(f"Generating pairwise insight...")

        pairwise_insights = []
        for i, focus_insight in enumerate(entity_insights[:top_k]):
            # Filter candidate insights - exclude insights from same entity
            candidate_insights = [
                insight for insight in entity_insights[i+1:] 
                if insight['entity'] != focus_insight['entity']
            ]
            if self.with_insight_processing_knowledge:
                insights = self._knowledge_based_pairwise_processing(
                    focus_insight, candidate_insights
                )
            else:
                insights = self._direct_pairwise_processing(
                    focus_insight, candidate_insights
                )

            pairwise_insights.extend(insights)

        store_to_json(pairwise_insights, self.output_dir / fname)

        return pairwise_insights

        
    def format_entity_insights_str(self, insights: List):
        insights_str  =  ""
        for i, insight in enumerate(insights):
            if self.entity_level_summary:
                insights_str += f"{i+1}. Entity: {insight['entity']}\nInterpretation: {insight['interpretation']}\n"
                detail_insight_str = "\n\t".join([x['interpretation'] for  x in insight['individual_insights']])
                insights_str += f"Details: {detail_insight_str}"
            else:
                insights_str += f"\n{i}. Type: {insight['type']}\nEntity: {insight['entity']}\nInterpretation: {insight['interpretation']}\n"
        return insights_str

    def _knowledge_based_pairwise_processing(self,
                                            focus_insight: Dict,
                                            candidate_insights: List[Dict]) -> List[Dict]:
        """Process pairwise insights using domain knowledge base"""
        # Get relevant knowledge patterns for insight type
        domain_knowledge = self.knowledge_base.get("pairwise_knowledge", {})
        domain_knowledge_str = "\n".join([f"\n{x['key_idea']}: {x['description']}"
                                            for _, knowledge_list in domain_knowledge.items() for x in knowledge_list])
        focus_insight_str = self.format_entity_insights_str([focus_insight])
        candidate_insights_str  =  self.format_entity_insights_str(candidate_insights)

        prompt = f"""
        As a {self.domain} expert targeting to {self.task}, analyze relationships between the focus insight and candidate insights using domain knowledge.

        Focus Insight:
        {focus_insight_str}

        Candidate Insights:
        {candidate_insights_str}

        Domain knowledge:
        {domain_knowledge_str}

        Task:
        Find the most significant relationships between the focus insight and any candidate insights.
        Consider all provided relationship patterns but focus on identifying the most important relationships.

        Structure output as JSON:
        [
            {{
                "type": type of relationship found,
                "focus_insight": {{
                    "type": focus insight type,
                    "interpretation": focus insight interpretation,
                    "entity": entity name
                }},
                "related_insight": {{
                    "type": related insight type,
                    "interpretation": related insight interpretation,
                    "entity": entity name
                }},
                "description": description of relationship insight,
                "evidence": ["supporting domain knowledge"],
                "significance": explanation of importance,
                "significance_score": score between 0-1
            }},
            ...
        ]

        Return only the most significant relationships that:
        1. Show clear and strong connections
        2. Have meaningful implications
        3. Are well-supported by evidence
        4. Help understand important patterns

        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "[") and excluding concluding remarks or follow-up suggestions (i.e., ends with "]")
        Ensure that each text string in the JSON object are enclosed with double quote "".
        """
        
        pairwise_insights = self.query_llm(prompt)
        
        return pairwise_insights

    def _direct_pairwise_processing(self,
                                    focus_insight: Dict, 
                                    candidate_insights: List[Dict]) -> List[Dict]:
        """Process pairwise insights directly without knowledge base"""
        candidate_insights_str = "\n".join([f"\nType: {x['type']}\tEntity: {x['entity']}\tInterpretation: {x['interpretation']}" for x in candidate_insights])
        
        prompt = f"""
        As a {self.domain} expert targeting to {self.task}, analyze relationships between the focus insight and candidate insights for {self.task} using domain knowledge.

        Focus Insight:
        Type: {focus_insight['type']}
        Entity: {focus_insight['entity']}
        Interpretation: {focus_insight['interpretation']}

        Candidate Insights:
        {candidate_insights_str}

        Consider aspects like:
        - Causal relationships
        - Correlations
        - Contrasts/comparisons
        - Supporting/conflicting evidence

        Task:
        Find the most significant relationships between the focus insight and any candidate insights.
        Consider all provided relationship patterns but focus on identifying the most important relationships.

        Structure output as JSON:
        [
            {{
                "type": type of relationship found,
                "focus_insight": {{
                    "type": focus insight type,
                    "interpretation": focus insight interpretation,
                    "entity": entity name
                }},
                "related_insight": {{
                    "type": related insight type,
                    "interpretation": related insight interpretation,
                    "entity": entity name
                }},
                "description": description of relationship insight,
                "evidence": ["supporting domain knowledge"],
                "significance": explanation of importance,
                "significance_score": score between 0-1
            }},
            ...
        ]

        Return only the most significant relationships that:
        1. Show clear and strong connections
        2. Have meaningful implications
        3. Are well-supported by evidence
        4. Help understand important patterns

        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "[") and excluding concluding remarks or follow-up suggestions (i.e., ends with "]")
        Ensure that each text string in the JSON object are enclosed with double quote "".
        """

        pairwise_insights = self.query_llm(prompt)
        
        return pairwise_insights

    def process_group_insights(self, entity_insights: List[Dict], pairwise_insights: List[Dict]) -> List[Dict]:
        """Process insights by entity clusters"""
        if self.with_insight_processing_knowledge:
            fname = "insight_group_knowledge_based.json"
        else:
            fname = "insight_group_direct.json"

        if self.reload_insight_dir and 'group' in self.reuse_insight_group:
            insight_path = self.reload_insight_dir / fname
            if insight_path.exists():
                self.logger.info(f"Reload group insight from {str(insight_path)}...")
                group_insights = load_json_from_path(insight_path)
                return group_insights
            else:
                self.logger.info(f"Insight file not exists, regenerating group insight...")
        else:
            self.logger.info(f"Generating group insight...")

        entity_clusters = self._generate_entity_cluster()
        group_insights = []

        for cluster in entity_clusters:
            if len(cluster["entities"]) <= 2:
                continue

            if self.with_insight_processing_knowledge:
                insight = self._knowledge_based_group_processing(
                    entity_insights, pairwise_insights, cluster
                )
            else:
                insight = self._direct_group_processing(
                    entity_insights, pairwise_insights, cluster
                )
            group_insights.extend(insight)

        store_to_json(group_insights, self.output_dir / fname)
        return group_insights

    def _generate_entity_cluster(self) -> Dict:
        """Generate entity clustering knowledge base directly without questions"""
        
        prompt = f"""
        As a {self.domain} expert targeting to {self.task}, identify typical entity clusters in {self.market}.

        Context:
        Entities: {self.entities}

        Task:
        Identify how these entities typically cluster and explain clustering rationale.

        Structure output as JSON:
        [
            {{
                "cluster_name": "cluster_name_1, 
                "entities": ["entity1", "entity2"],
                "reason_of_clustering": "explanation of why these entities form a cluster"
            }},
            {{
                "cluster_name": "cluster_name_2, 
                "entities": ["entity3", "entity4"],
                "reason_of_clustering": "explanation of why these entities form a cluster"
            }},
            ...
        ]

        Provide clusters that are:
        - Logically grouped
        - Well-justified
        - Relevant to {self.domain}
        Return only the JSON object. Ensure that each text string in the JSON object are enclosed with double quote "".
        """
        
        entity_clusters = self.query_llm(prompt)
        
        store_to_json(entity_clusters, self.output_dir / "entity_clusters.json") 

        return entity_clusters

    def _knowledge_based_group_processing(self, entity_insights: List[Dict], 
                                        pairwise_insights: List[Dict],
                                        cluster: Dict) -> List[Dict]:
        """Process insights by entity clusters using knowledge base"""
        # Get relevant knowledge
        domain_knowledge = self.knowledge_base.get("group_knowledge", {})

        domain_knowledge_str = "\n".join([f"\n{x['key_idea']}: {x['description']}"
                                            for _, knowledge_list in domain_knowledge.items() for x in knowledge_list])

        # Process each cluster
        cluster_entities = cluster["entities"]

        # Get insights for this cluster
        cluster_entity_insights = [i for i in entity_insights 
                                    if i["entity"] in cluster_entities]

        # Get pairwise insights involving this cluster
        cluster_pairwise_insights = [i for i in pairwise_insights 
            if (i["focus_insight"]["entity"] in cluster_entities or 
                i["related_insight"]["entity"] in cluster_entities)]

        cluster_entity_insights_str = self.format_entity_insights_str(cluster_entity_insights)
        cluster_pairwise_insights_str = "\n".join([f"\nType: {x['type']}\tDescription: {x['description']}" for x in cluster_pairwise_insights])

        prompt = f"""
        As a {self.domain} expert targeting to {self.task}, analyze insights for this entity cluster using domain knowledge.

        Cluster: {cluster['cluster_name']}
        Member Entities: {cluster_entities}

        Entity Insights: {cluster_entity_insights_str}
        Pairwise Relationship Insights: {cluster_pairwise_insights_str}

        Domain knowledge:
        {domain_knowledge_str}

        Task:
        Find the most significant insights among the entity insights and the pairwise insights.
        Consider all provided insight types but focus on identifying the most important insights.

        Structure output as JSON:
        [
            {{
                "cluster_name": "name of cluster",
                "entities": ["member entities"],
                "cluster_insights": [
                    {{
                        "type": "group insight type",
                        "description": "description of the group insight",
                        "supporting_insights": ["relevant insights"],
                        "significance": explanation of importance,
                        "significance_score": score between 0-1
                    }},
                    ...
                ],
            }},
            ...
        ]
        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "[") and excluding concluding remarks or follow-up suggestions (i.e., ends with "]")
        Ensure that each text string in the JSON object are enclosed with double quote "".
        """

        insight = self.query_llm(prompt)

        return insight

    def _direct_group_processing(self, entity_insights: List[Dict], 
                                pairwise_insights: List[Dict],
                                cluster: Dict) -> List[Dict]:
        """Process insights by entity clusters without knowledge base"""
        cluster_entities = cluster["entities"]

        # Get insights for this cluster
        cluster_entity_insights = [i for i in entity_insights 
                                    if i["entity"] in cluster_entities]
        
        cluster_pairwise_insights = [i for i in pairwise_insights 
            if (i["focus_insight"]["entity"] in cluster_entities or 
                i["related_insight"]["entity"] in cluster_entities)]
        
        cluster_entity_insights_str = self.format_entity_insights_str(cluster_entity_insights)
        cluster_pairwise_insights_str = "\n".join([f"\nType: {x['type']}\tDescription: {x['description']}" for x in cluster_pairwise_insights])

        prompt = f"""
        As a {self.domain} expert targeting to {self.task}, analyze insights for this entity cluster for {self.task} using domain knowledge.

        Cluster: {cluster['cluster_name']}
        Clustering Reason: {cluster["reason_of_clustering"]}
        Member Entities: {cluster_entities}

        
        Entity Insights: 
        {cluster_entity_insights_str}

        Pairwise Relationship Insights: 
        {cluster_pairwise_insights_str}

        Task:
        Find the most significant insights among the entity insights and the pairwise insights.
        Consider all provided insight types but focus on identifying the most important insights.

        Structure output as JSON:
        [
            {{
                "cluster_name": "name of cluster",
                "entities": ["member entities"],
                "cluster_insights": [
                    {{
                        "type": "group insight type",
                        "description": "description of the group insight",
                        "supporting_insights": ["relevant insights"],
                        "entities": ["entities involved"],
                        "significance": explanation of importance,
                        "significance_score": score between 0-1
                    }},
                    ...
                ],
            }},
            ...
        ]

        Return only the JSON object by presenting the JSON object directly without meta-commentary, introductions, or language specification (i.e., start with "[") and excluding concluding remarks or follow-up suggestions (i.e., ends with "]")
        Ensure that each text string in the JSON object are enclosed with double quote "".
        """

        insight = self.query_llm(prompt)

        return insight
    
    def process_overall_insights(self, entity_insights: List[Dict], pairwise_insights: List[Dict], group_insights: List[Dict]) -> List[Dict]:
        """Process overall system-level insights from all previous levels of analysis"""
        if self.with_insight_processing_knowledge:
            fname = "insight_overall_knowledge_based.json"
        else:
            fname = "insight_overall_direct.json"

        if self.reload_insight_dir and 'overall' in self.reuse_insight_group:
            insight_path = self.reload_insight_dir / fname
            if insight_path.exists():
                self.logger.info(f"Reload overall insight from {str(insight_path)}...")
                overall_insight = load_json_from_path(insight_path)
                return overall_insight
            else:
                self.logger.info(f"Insight file not exists, regenerating overall insight...")
        else:
            self.logger.info(f"Generating overall insight...")

        if self.with_insight_processing_knowledge:
            overall_insight = self._knowledge_based_overall_processing(
                entity_insights, pairwise_insights, group_insights
            )
        else:
            overall_insight = self._direct_overall_processing(
                entity_insights, pairwise_insights, group_insights
            )
        
        store_to_json(overall_insight, self.output_dir / fname)
        return overall_insight

    def _knowledge_based_overall_processing(self, entity_insights: List[Dict], 
                                        pairwise_insights: List[Dict],
                                        group_insights: List[Dict]) -> List[Dict]:
        """Process overall system-level insights using knowledge base"""
        # Get relevant knowledge
        overall_knowledge = self.knowledge_base.get("overall_knowledge", [])

        overall_knowledge_str = "\n".join([f"\n{x['key_idea']}: {x['description']}"
                                            for _, knowledge_list in overall_knowledge.items() for x in knowledge_list])

        entity_insights_str = self.format_entity_insights_str(entity_insights)
        pairwise_insights_str = "\n".join([f"\nType: {x['type']}\tDescription: {x['description']}" for x in pairwise_insights])
        group_insights_str = "\n".join([f"\nType: {x['type']}\tDescription: {x['description']}" for insights in group_insights for x in insights['cluster_insights']])

        prompt = f"""
        As a {self.domain} expert targeting to {self.task}, analyze system-level patterns using domain knowledge.

        Context:
        Entity Insights: 
        {entity_insights_str}

        Relationship Insights: 
        {pairwise_insights_str}

        Group Insights: 
        {group_insights_str}

        Domain Knowledge:
        {overall_knowledge_str}

        Task:
        Find the most significant insights among the entity insights and the pairwise insights.
        Consider all provided knowledge but focus on identifying the most important insights.

        Structure output as JSON:
            [
                {{
                    "type": "overall insight type",
                    "description": "description of the overall insight",
                    "supporting_insights": ["relevant insights"],
                    "entities": ["entities involved"],
                    "significance": explanation of importance,
                    "significance_score": score between 0-1
                }},
                ...
            ]
        
        Return only the JSON object by presenting the JSON object directly without meta-commentary, introductions, or language specification (i.e., start with "[") and excluding concluding remarks or follow-up suggestions (i.e., ends with "]")
        Ensure that each text string in the JSON object are enclosed with double quote "".
        """

        results = self.query_llm(prompt)

        return results

    def _direct_overall_processing(self, entity_insights: List[Dict], 
                                pairwise_insights: List[Dict],
                                group_insights: List[Dict]) -> List[Dict]:
        """Process overall system-level insights without knowledge base"""

        entity_insights_str = self.format_entity_insights_str(entity_insights)
        pairwise_insights_str = "\n".join([f"\nType: {x['type']}\tDescription: {x['description']}" for x in pairwise_insights])
        group_insights_str = "\n".join([f"\nType: {x['type']}\tDescription: {x['description']}" for insights in group_insights for x in insights['cluster_insights']])

        prompt = f"""
        As a {self.domain} expert targeting to {self.task}, analyze system-level patterns using domain knowledge.

        Context:
        Entity Insights: 
        {entity_insights_str}

        Relationship Insights: 
        {pairwise_insights_str}

        Group Insights: 
        {group_insights_str}

        Task:
        Find the most significant insights among the entity insights and the pairwise insights.
        Consider all provided knowledge but focus on identifying the most important insights.

        Structure output as JSON:
            [
                {{
                    "type": "overall insight type",
                    "description": "description of the overall insight",
                    "supporting_insights": ["relevant insights"],
                    "entities": ["entities involved"],
                    "significance": explanation of importance,
                    "significance_score": score between 0-1
                }},
                ...
            ]
        
        Return only the JSON object by presenting the JSON object directly without meta-commentary, introductions, or language specification (i.e., start with "[") and excluding concluding remarks or follow-up suggestions (i.e., ends with "]")
        Ensure that each text string in the JSON object are enclosed with double quote "".
        """

        results = self.query_llm(prompt)
                
        return results
   
    def generate_narrative(self, entity_insights: List[Dict], pairwise_insights: List[Dict], 
                           group_insights: List[Dict], overall_insights: List[Dict]) -> str:
        """Generate narrative from insights without using knowledge base"""
        entity_insights_str = self.format_entity_insights_str(entity_insights)
        pairwise_insights_str = ("\nPairwise Relationship Insights:\n" + "\n".join([f"\nType: {x['type']}\tDescription: {x['description']}" for x in pairwise_insights])) if pairwise_insights else ""
        group_insights_str = ("\nGroup Insights: \n" + "\n".join([f"\nType: {x['type']}\tDescription: {x['description']}" for insights in group_insights for x in insights['cluster_insights']]))
        overall_insights_str = ("\nOverall Insights: \n" + "\n".join([f"\nType: {x['type']}\tDescription: {x['description']}" for x in overall_insights]))
        narrative_knowledge_str = json.dumps(self.knowledge_base.get("narrative_knowledge", {})) if self.with_narrative_knowledge else ""

        if self.prompting == "cot":
            prompt = f"""
            As a {self.domain} expert targeting to {self.task} for {self.market}, let's construct the narrative step by step.

            Context:
            Entities: {self.entities}

            Insights:
            {entity_insights_str}
            {pairwise_insights_str}
            {group_insights_str}
            {overall_insights_str}

            Follow each step below to explain your thinking process, then provide the final narrative.
            Write all text in plain format without any markdown, formatting, or special characters.

            Step 1: Insight Analysis and Prioritization
            Print "INSIGHT_ANALYSIS_START"
            Analyze and explain:
            - The key findings and their significance
            - The relationships between different insight levels
            - Important patterns and trends you've identified
            - Critical contextual factors
            Provide your actual analysis of the insights, not just answers to these points.
            Print "INSIGHT_ANALYSIS_END"

            Step 2: Narrative Structure Planning
            Print "STRUCTURE_PLANNING_START"
            Outline your specific plan for:
            - The organization of key points
            - Your chosen flow between topics
            - Where and how you'll integrate evidence
            - Your approach to detail balance
            Explain your actual structural decisions, not these prompts.
            Print "STRUCTURE_PLANNING_END"

            Step 3: Context Integration
            Print "CONTEXT_ANALYSIS_START"
            Explain your decisions about:
            - Essential domain knowledge to include
            - Relevant market conditions to discuss
            - Important historical context to reference
            - Key assumptions to address
            Detail your actual context integration strategy, not these guidelines.
            Print "CONTEXT_ANALYSIS_END"

            Step 4: Language and Style Planning
            Print "STYLE_PLANNING_START"
            Describe your specific choices for:
            - Tone and style appropriate for the task
            - Professional language approach
            - Clarity and readability strategies
            - Technical term usage
            Explain your actual language and style decisions, not these prompts.
            Print "STYLE_PLANNING_END"

            Step 5: Narrative Review
            Print "NARRATIVE_REVIEW_START"
            Provide your assessment of:
            - Coverage of key insights
            - Flow and logical progression
            - Evidence integration
            - Detail level appropriateness
            Write your actual review analysis, not these checkpoints.
            Print "NARRATIVE_REVIEW_END"

            After completing the analysis, print "FINAL NARRATIVE:" and provide the final narrative that:
            1. Presents key findings logically
            2. Connects insights meaningfully
            3. Highlights important patterns
            4. Provides relevant context
            5. Draws meaningful conclusions

            Requirements:
            - Start with most significant findings
            - Flow naturally between topics
            - Support claims with evidence
            - Include relevant details while staying concise
            - Be appropriate for {self.task} format

            Example of expected output format:
            INSIGHT_ANALYSIS_START
            Based on the provided insights, the most significant finding is the 15% increase in market volatility across technology sectors. This connects directly to the group-level insights showing similar patterns in related industries...
            INSIGHT_ANALYSIS_END

            [Other analysis sections...]

            FINAL NARRATIVE:
            [Your actual narrative here]

            The output should begin with your explicit reasoning process in each analysis section, followed by the narrative. Do not include the guiding questions in your output.
            """
            raw_results = self.query_llm(prompt, return_type="str")
            with open(self.output_dir / "raw_narrative.txt", 'w') as f:
                f.write(raw_results)

            narrative, _, _ = extract_text(raw_results, text_mark="FINAL NARRATIVE:")
        else:
            prompt = f"""
            As a {self.domain} expert targeting to {self.task} for {self.market}.

            Context:
            Entities: {self.entities}
            {narrative_knowledge_str}
            Insights identified:
            {entity_insights_str}
            {pairwise_insights_str}
            {group_insights_str}
            {overall_insights_str}

            Task:
            Write a clear, professional narrative that:
            1. Presents key findings logically
            2. Connects insights meaningfully
            3. Highlights important patterns
            4. Provides relevant context
            5. Draws meaningful conclusions

            The narrative should:
            - Start with most significant findings
            - Flow naturally between topics
            - Support claims with evidence
            - Include relevant details while staying concise
            - Be appropriate for {self.task} format

            Return only the narrative text, without any markdown or special formatting.
            """
        
            narrative = self.query_llm(prompt, return_type="string")

        with open(self.output_dir / "narrative.txt", 'w') as f:
            f.write(narrative)
    
        return narrative.strip()

    def run(self) -> str:
        """
        Run full narration generation pipeline"""
        try:
            # 1. Get entity insights
            self.logger.info("Get entity insights...")
            entity_insights = []
            for entity in self.entities:
                insights = self.get_entity_insights(entity)
                for insight in insights:
                    insight['entity'] = entity
                entity_insights.extend(insights)
            entity_insights.sort(key=lambda x:x["significance_score"])

            if self.with_hierarchical_structure:
                # 2. Process pairwise relationships
                if 'pairwise' in self.with_hierarchical_structure:
                    self.logger.info("Processing pairwise relationships...")
                    pairwise_insights = self.process_pairwise_insights(entity_insights)
                else:
                    pairwise_insights = []

                # 3. Process group insights
                if 'group' in self.with_hierarchical_structure:
                    self.logger.info("Processing group insights...")
                    group_insights = self.process_group_insights(entity_insights, pairwise_insights)
                else:
                    group_insights = []
                
                # 4. Process overall insights
                if 'overall' in self.with_hierarchical_structure:
                    self.logger.info("Processing overall insights...")
                    overall_insights = self.process_overall_insights(entity_insights, pairwise_insights, group_insights)
                else:
                    overall_insights = []
            else:
                pairwise_insights = []
                group_insights = []
                overall_insights = []

            # 5. Generate narrative
            self.logger.info("Generating narrative...")
            narrative = self.generate_narrative(
                entity_insights=entity_insights,
                pairwise_insights=pairwise_insights,
                group_insights=group_insights,
                overall_insights=overall_insights
            )

            self.logger.info("Narration generation complete")
            return narrative

        except Exception as e:
            self.logger.error(f"Error in narration pipeline: {str(e)}")

    def get_precomputation_code_path(self, insight_type: str, prefix: str = ""):
        return self.knowledge_dir / f"{prefix}precomputation_code_{insight_type}_with{'' if self.with_entity_analysis_knowledge else 'out'}_knowledge{'_cot' if self.prompting =='cot' else ''}.py"

    def generate_all_computation_code(self):
        self.logger.info("Generating precomputationg code...")
        if self.with_entity_analysis_knowledge:
            question_path = self.knowledge_dir / "entity_questions.json"
            questions = self._load_json_data(question_path)
            assert questions is not None
        else:
            questions = [
                {
                    "insight_type": "general",
                    "question": f"On the date of interest, for the entity of interest, analyze and find insights for {self.task} using derived values from the data"
                },
            ]

        for question in questions:
            insight_type = "_".join(question["insight_type"].replace('/', '_').split())
            precomputation_code_path = self.get_precomputation_code_path(insight_type)

            if precomputation_code_path.exists() and not self.recreate_code:
                self.logger.info(f"Reusing code from {str(precomputation_code_path)}...")
                continue

            question_text = json.dumps(question)
            with open(self.knowledge_dir / 'code_generation_count.txt', 'w') as f:
                f.write(f"\nMarket: {self.market}\tInsight type: {insight_type}\n")

            retry_after_generate_counts = 0
            while retry_after_generate_counts < 5:
                try:
                    self.logger.info(f"Generating code for {precomputation_code_path}...")
                    self.generate_precomputation_code(question_text, precomputation_code_path, insight_type)
                    with open(self.knowledge_dir / 'code_generation_count.txt', 'w') as f:
                        f.write(f"retry_after_generate_counts: {retry_after_generate_counts}\tstatus: success\n")
                    break
                except Exception as e:
                    self.logger.info(f"Error creating {precomputation_code_path}: {str(e)}")
                    if retry_after_generate_counts < 20:
                        self.logger.info(f"Regenerating code...")
                        retry_after_generate_counts += 1
                    else:
                        self.logger.info("Terminating...")
                        precomputation_code_path.unlink(missing_ok=True)
                        with open(self.knowledge_dir / 'code_generation_count.txt', 'w') as f:
                            f.write(f"entity: {entity}\tretry_after_generate_counts: {retry_after_generate_counts}\tstatus: fail\n")
                        raise e

            pass_test = False
            for entity in self.entities:
                entity_str = "-".join(remove_month_labels(entity).split())
                raw_precomputation_result_path = self.output_dir / f"testing_precomputiong_result_{entity_str}_{insight_type}.txt"

                retry_after_test_counts = 0
                while retry_after_test_counts < 5:
                    try:
                        self.logger.info(f"Testing code for {precomputation_code_path} using {entity} data...")
                        precomputation_results = self.execute_computation(precomputation_code_path, self.data_file, entity, self.report_date, raw_precomputation_result_path)
                        if self.with_insight_processing_knowledge:
                            result_length_limit = 500
                        else:
                            result_length_limit = 2000
                        assert len(precomputation_results) < result_length_limit, f"Results print out {precomputation_results} is too long, only include the values needed in describing the insights."
                        assert len(precomputation_results) > 10, f"Results print out {precomputation_results} is too short, code invalid or not executed properly."
                        assert count_numeric_value(precomputation_results) > 0, f"Results print out should contain numerical values while the current results is {precomputation_results}, please check the calculation process..."
                        with open(self.knowledge_dir / 'code_generation_count.txt', 'w') as f:
                            f.write(f"entity: {entity}\tretry_after_generate_counts: {retry_after_generate_counts}\tstatus: success")
                        pass_test = True
                        break
                    except Exception as e:
                        self.logger.info(f"Error running {precomputation_code_path} {str(e)}")
                        self.logger.info(f"Debugging code...")
                        self.debug_generate_precomputation_code(question_text, precomputation_code_path, error=str(e))
                        retry_after_test_counts += 1
                
                if pass_test:
                    self.logger.info(f"Testing passed with {entity}")
                    break

            if not pass_test:
                self.logger.info(f"Testing failed, deleting generated code {precomputation_code_path}...")
                if precomputation_code_path.exists():
                    os.remove(precomputation_code_path)
                with open(self.knowledge_dir / 'code_generation_count.txt', 'w') as f:
                    f.write(f"entity: {entity}\tretry_after_generate_counts: {retry_after_generate_counts}\tstatus: fail\n")
                raise f"Fail to generate {precomputation_code_path}..."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="gpt-4o", type=str)
    parser.add_argument('--with_entity_analysis_knowledge', action='store_true')
    parser.add_argument('--with_insight_processing_knowledge', action='store_true')
    parser.add_argument('--with_narrative_knowledge', action='store_true')
    parser.add_argument('--with_hierarchical_structure', nargs='*', type=str)
    parser.add_argument('--question_based_insight_processing_knowledge', action='store_true')
    parser.add_argument('--prompting', default="", type=str)
    parser.add_argument('--recreate_code', action='store_true')
    parser.add_argument('--regenerate_narrative', action='store_true')
    parser.add_argument('--reload_insight_dir', default="", type=str)
    parser.add_argument('--reuse_insight_group', nargs='*', type=str)
    parser.add_argument('--knowledge_model_name', default='', type=str)
    parser.add_argument('--market_group', nargs='*', type=str)
    args = parser.parse_args()

    model_name = args.model.split('/')[-1].lower()
    knowledge_model_name = args.knowledge_model_name.split('/')[-1].lower() if  args.knowledge_model_name else model_name
    setup_str = f"{model_name}{'_entity_klg' if args.with_entity_analysis_knowledge else  ''}{'_process_klg' if args.with_insight_processing_knowledge else ''}{'_qst_based' if args.question_based_insight_processing_knowledge else ''}{'_narrative_klg' if args.with_narrative_knowledge else ''}{'_hierarchy' if args.with_hierarchical_structure else ''}{'_'+args.prompting if args.prompting else ''}{'_' + '_'.join(args.with_hierarchical_structure) if (args.with_hierarchical_structure and (len(args.with_hierarchical_structure)<3) )  else ''}"
    if model_name != knowledge_model_name:
        setup_str += f"_{knowledge_model_name}_klg"
    result_dir = Path("results") / setup_str 
    result_dir.mkdir(exist_ok=True, parents=True) 
    current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = setup_logger("NarrativeGenerator", str(result_dir / f"log_{current_time}.txt"))

    if args.model == "gpt-4o":
        query_llm_func = partial(get_gpt_response, logger=logger)
    else:
        pipeline = load_pipeline(args.model)
        query_llm_func = partial(get_llama_response, pipeline=pipeline, logger=logger)

    data_dir = Path("data") / "datatales"
    market_dir_list = [x for x in data_dir.iterdir() if x.is_dir()]
    market_dir_list.sort()

    print(args)
    for market_dir in market_dir_list:
        market = market_dir.name
        if args.market_group and market not in args.market_group:
            continue
        data_path_list = [x for x in (market_dir / "test").iterdir() if x.is_file() and x.suffix == '.csv']
        knowledge_dir = str(Path("results") / "knowledge" / knowledge_model_name / market)
        data_path_list.sort()
        computation_code_created = False
        for data_path in data_path_list:
            report_date =  data_path.stem
            output_dir = str(Path("results")/ setup_str / market / report_date)
            if args.reload_insight_dir:
                cur_reload_insight_dir = str(Path(args.reload_insight_dir) / market / report_date)
            else:
                cur_reload_insight_dir = ""

            if (Path(output_dir) / "narrative.txt").exists() and not args.regenerate_narrative:
                continue

            if args.regenerate_narrative:
                (Path(output_dir) / "narrative.txt").unlink()

            narration_generator = NarrationGenerator(data_file=data_path, 
                                                    domain="finance market analysis", 
                                                    market="market",
                                                    task="generate daily market report that help reader understand market conditions and trends, and make investment decisions",
                                                    report_date=report_date,
                                                    entity_col="Product Name",
                                                    query_llm=query_llm_func,
                                                    with_entity_analysis_knowledge = args.with_entity_analysis_knowledge,
                                                    with_insight_processing_knowledge=args.with_insight_processing_knowledge,
                                                    with_narrative_knowledge = args.with_narrative_knowledge,
                                                    with_hierarchical_structure = args.with_hierarchical_structure,
                                                    question_based_insight_processing_knowledge=args.question_based_insight_processing_knowledge,
                                                    prompting=args.prompting,
                                                    knowledge_dir=knowledge_dir,
                                                    output_dir=output_dir,
                                                    recreate_code=args.recreate_code,
                                                    reload_insight_dir=cur_reload_insight_dir,
                                                    reuse_insight_group=args.reuse_insight_group,
                                                    logger=logger)
            if not computation_code_created:
                try:
                    narration_generator.generate_all_computation_code()
                    computation_code_created = True
                except:
                    continue

            narration_generator.run()
