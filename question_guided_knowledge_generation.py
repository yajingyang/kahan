import os
import json
from typing import Optional, List, Dict, Callable
from pathlib import Path 
from utils import json_serialize, get_gpt_response, get_dtypes_dict, get_llama_response, load_pipeline, setup_logger
import pandas as pd
from functools import partial
import argparse
import logging
from datetime import datetime, date

class KnowledgeBaseGenerator:
    def __init__(self, domain: str, task: str, market: str, entities: List[str], query_llm: Callable, output_dir: str, with_question_guided: Optional[bool] = True):
        self.domain = domain
        self.task = task
        self.market = market
        self.entities = entities
        self.query_llm = query_llm
        self.output_dir = Path(output_dir) / market
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.with_question_guided = with_question_guided
        self.insight_str = ""


    def generate_entity_insight_questions(self, schema: Dict) -> List[str]:
        """Generate analysis questions using LLM"""
        schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])

        prompt = f"""
        You are a {self.domain} expert focused on {self.market}. 
        Your eventual goal is to {self.task}.
        For now, generate 5 comprehensive analysis questions to extract insights for the task.

        Available data fields:
        {schema_str}

        Task:
        Generate analysis questions that:
        1. Cover different aspects of entity analysis
        2. Can be answered using quantifiable metrics
        3. Provide meaningful insights for {self.task}

        For each question:
        1. Specify a clear insight type
        2. List ALL metrics that need to be pre-computed
        3. Consider relative comparisons if relevant

        Structure output as JSON list:
        [
            {{
                "insight_type": "descriptive name of insight type",
                "question": "analysis question with ALL required metrics and calculations specified",
                "required_metrics": [
                    {{
                        "metric": "metric name",
                        "calculation": "how to compute",
                        "purpose": "how used in analysis"
                    }}
                ],
                "comparisons": ["relevant comparisons"]
            }}
        ]

        Example output:
        [
            {{
                "insight_type": "trend",
                "question": "On the date of interest, analyze the trend direction and strength by comparing price against multiple technical indicators, including moving averages, momentum indicators, and support/resistance levels",
                "required_metrics": [
                    {{
                        "metric": "SMA_20",
                        "calculation": "20-day Simple Moving Average of closing_price",
                        "purpose": "Identify short-term trend direction"
                    }},
                    {{
                        "metric": "SMA_50",
                        "calculation": "50-day Simple Moving Average of closing_price",
                        "purpose": "Identify medium-term trend direction"
                    }},
                    {{
                        "metric": "SMA_200",
                        "calculation": "200-day Simple Moving Average of closing_price",
                        "purpose": "Identify longer-term trend direction"
                    }},
                    {{
                        "metric": "MACD_line",
                        "calculation": "12-day EMA minus 26-day EMA of closing_price",
                        "purpose": "Measure trend momentum and possible reversals"
                    }},
                    {{
                        "metric": "MACD_signal",
                        "calculation": "9-day EMA of MACD_line",
                        "purpose": "Generate trend signals when crossed with MACD line"
                    }},
                    {{
                        "metric": "support_level",
                        "calculation": "Minimum price levels with multiple bounces in last 30 days",
                        "purpose": "Identify price support areas"
                    }},
                    {{
                        "metric": "resistance_level",
                        "calculation": "Maximum price levels with multiple rejections in last 30 days",
                        "purpose": "Identify price resistance areas"
                    }}
                ],
                "comparisons": [
                    "Current price vs moving averages",
                    "MACD line vs signal line",
                    "Price position relative to support/resistance",
                    "Current vs historical trend strength"
                ]
            }},
            ...
        ]

        Ensure the questions:
        1. Are specific and quantifiable
        2. Use available data fields
        3. Cover different analytical aspects
        4. Support {self.task} objectives
        5. Are appropriate for {self.market} analysis
        6. Assigned with a different insight type

        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "[") and excluding concluding remarks or follow-up suggestions (i.e., ends with "]")
        """

        entity_insights = self.query_llm(prompt)
        self._save_json(entity_insights, "entity_questions.json")
            
        return entity_insights

    def generate_pairwise_questions(self) -> Dict:
        """Generate questions for pairwise insight analysis"""
        prompt = f"""
        You are a {self.domain} expert focused on {self.market}. 
        Your eventual goal is to {self.task}.
        For now, generate questions that will help build a knowledge base about relationships and patterns between entities.

        Context:
        Entities being analyzed: {self.entities}
        
        Types of insights being analyzed for each entity:
        {self.insight_str}

        Task:
        Generate questions in these categories:
        1. Questions about how different metrics and measurements typically relate to each other
        2. Questions about how different entities typically interact or influence each other
        3. Questions about common factors that affect multiple entities or metrics
        4. Questions about time-based patterns and sequences
        5. Questions about system-level patterns and dependencies

        Structure output as JSON:
        {{
            "metric_patterns": [
                "How do changes in [metric A] typically relate to changes in [metric B]?",
                "What conditions affect the relationship between [metric A] and [metric B]?"
            ],
            "entity_interactions": [
                "What are common ways that [entity type A] affects [entity type B]?",
                "Under what conditions do [entities] influence each other most strongly?"
            ],
            "common_factors": [
                "What external factors commonly affect multiple [entities/metrics]?",
                "How do [domain-specific factors] typically impact different [entities]?"
            ],
            "temporal_patterns": [
                "What typical sequences of changes occur in [metrics/entities]?",
                "How do time-based factors affect relationships between [entities]?"
            ],
            "system_patterns": [
                "What system-level conditions influence relationships between [entities]?",
                "How do structural factors in [domain] affect interactions?"
            ]
        }}

        Generate questions that are:
        - Specific to the provided entities and insight types
        - Relevant to understanding relationships in {self.domain}
        - Focused on common and important patterns
        - Applicable across different scenarios
        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "{{") and excluding concluding remarks or follow-up suggestions (i.e., ends with "}}") 
        """
        
        questions = self.query_llm(prompt)
        
        self._save_json(questions, "pairwise_questions.json")
            
        return questions
            
    def generate_pairwise_knowledge(self, pairwise_questions: Optional[Dict] = None) -> Dict:
        """Generate knowledge base for pairwise analysis"""
        pairwise_questions_str = f"\nQuestions to address:\n {json.dumps(pairwise_questions, indent=2)}\n"  if pairwise_questions else ""

        prompt = f"""
        You are a {self.domain} expert focused on {self.market}. 
        Your eventual goal is to {self.task}.
        For now, generate domain knowledge about relationships and patterns based on these analysis questions.

        Context:
        Entities in scope: {self.entities}
        
        Insight types being analyzed:
        {self.insight_str}
        {pairwise_questions_str}
        Task:
        {"For each category of questions, " if  pairwise_questions else ""}provide domain knowledge about:
        1. Typical patterns and relationships
        2. Common influencing factors
        3. Important conditions and contexts
        4. Notable exceptions or special cases

        Structure output as JSON:
        {{
            "<knowledge_group_name>": [
                {{
                    "key_idea": <{"analysis question" if self.with_question_guided else "keywords about the knowledge"}>,
                    "description":  <{"answer to analysis question" if self.with_question_guided else "detail about the knowledge"}>
                }},
                ...
            ],
            ...
        }}


        Provide knowledge that is:
        - Specific to {self.domain}
        - Relevant for analyzing the given entities and insight types
        - Based on established patterns and relationships
        - Applicable across different scenarios
        - Clear and actionable for analysis
        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "{{") and excluding concluding remarks or follow-up suggestions (i.e., ends with "}}") 
        """

        knowledge = self.query_llm(prompt)
        
        self._save_json(knowledge, f"pairwise_knowledge{'_without_question' if pairwise_questions  is None else '_with_question'}.json")
            
        return knowledge

    def generate_group_questions(self) -> Dict:
        """Generate questions for group insight analysis"""

        prompt = f"""
        You are a {self.domain} expert focused on {self.market}. 
        Your eventual goal is to {self.task}.
        For now, generate questions that will help build a knowledge base about group-level patterns and causal chains.

        Context:
        Entities being analyzed: {self.entities}
        
        Types of insights being analyzed:
        {self.insight_str}

        Task:
        Generate questions that will help identify:
        1. Patterns and behaviors common to groups of similar entities/insights
        2. How groups interact and influence each other
        3. How effects propagate through groups
        4. Causal chains at the group level

        Structure output as JSON:
        {{
            "group_characteristics": [
                "What common patterns appear across [similar entities/insights]?",
                "How do group characteristics affect behavior patterns?"
            ],
            "inter_group_dynamics": [
                "How do different groups typically interact?",
                "What factors influence between-group relationships?"
            ],
            "propagation_patterns": [
                "How do effects typically spread across groups?",
                "What conditions affect propagation between groups?"
            ],
            "causal_chains": [
                "What typical cause-effect sequences occur at group level?",
                "How do group characteristics influence causal chains?"
            ]
        }}

        Generate questions that are:
        - Specific to the provided entities and insight types
        - Build upon identified pairwise patterns
        - Focus on group-level dynamics
        - Help identify causal relationships

        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "[") and excluding concluding remarks or follow-up suggestions (i.e., ends with "]") 
        """
        
        questions = self.query_llm(prompt)
        
        self._save_json(questions, "group_questions.json")
            
        return questions

    def generate_group_knowledge(self, pairwise_patterns: Dict, group_questions: Optional[Dict] = None) -> Dict:
        """Generate knowledge base for group analysis
        Previously identified pairwise patterns: {json.dumps(pairwise_patterns, indent=2)}"""
        group_questions_str = f"\nQuestions to address:\n {json.dumps(group_questions, indent=2)}\n"  if self.with_question_guided else ""

        prompt = f"""
        You are a {self.domain} expert focused on {self.market}. 
        Your eventual goal is to {self.task}.
        For now, generate domain knowledge {" by answering these analysis questions" if self.with_question_guided else ""}. 

        Context:
        Entities in scope: {self.entities}
        Insight types being analyzed: {self.insight_str}
        {group_questions_str}
        Task:
        {" For each category of questions" if self.with_question_guided else ""}, provide knowledge about:
        1. Common patterns across groups
        2. How effects propagate through groups 
        3. Causal relationships at group level

        Structure output as JSON:
        {{
            "<knowledge_group_name>": [
                {{
                    "key_idea": <{"analysis question" if self.with_question_guided else "keywords about the knowledge"}>,
                    "description":  <{"answer to analysis question" if self.with_question_guided else "detail about the knowledge"}>
                }},
                ...
            ],
            ...
        }}

        
        Provide knowledge that is:
        - Specific to {self.domain}
        - Builds on identified pairwise patterns
        - Relevant to the given entities and insight types
        - Focused on group-level dynamics
        - Clear and actionable for analysis
        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "{{") and excluding concluding remarks or follow-up suggestions (i.e., ends with "}}") 
        """
        
        knowledge = self.query_llm(prompt)
        
        self._save_json(knowledge, f"group_knowledge{'_without_question' if group_questions  is None else '_with_question'}.json")
            
        return knowledge

    def generate_overall_questions(self) -> Dict:
        """Generate questions for overall insight analysis"""
        prompt = f"""
        You are a {self.domain} expert focused on {self.market}. 
        Your eventual goal is to {self.task}.
        For now, generate questions that will help build a knowledge base about system-level patterns and causal networks.

        Context:
        Entities being analyzed: {self.entities}
        Types of insights: {self.insight_str}

        Task:
        Generate questions that will help identify:
        1. System-wide patterns and behaviors
        2. Networks of causal relationships
        3. Strategic implications and impacts
        4. Emergent phenomena and feedback systems

        Structure output as JSON:
        {{
            "system_patterns": [
                "What patterns emerge at the system level?",
                "How do system components collectively behave?"
            ],
            "causal_networks": [
                "What major causal networks exist in the system?",
                "How do different causal chains interact?"
            ],
            "strategic_implications": [
                "What are the system-wide implications of [patterns]?",
                "How do system behaviors affect outcomes?"
            ],
            "feedback_systems": [
                "What feedback loops exist in the system?",
                "How do system components regulate each other?"
            ]
        }}

        Generate questions that are:
        - Build upon pairwise and group patterns
        - Focus on system-level dynamics
        - Help identify complex causal networks
        - Address strategic implications
        
        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "[]") and excluding concluding remarks or follow-up suggestions (i.e., ends with "]") 
        """
        
        questions = self.query_llm(prompt)
        
        self._save_json(questions, "overall_questions.json")
            
        return questions

    def generate_overall_knowledge(self, pairwise_patterns: Dict, group_patterns: Dict, overall_questions: Optional[Dict] = None) -> Dict:
        """Generate knowledge base for overall analysis
        Previously identified patterns:
        - Pairwise: {json.dumps(pairwise_patterns, indent=2)}
        - Group: {json.dumps(group_patterns, indent=2)}"""
        overall_questions_str = f"\nQuestions to address:\n {json.dumps(overall_questions, indent=2)}\n"  if self.with_question_guided else ""

        prompt = f"""
        You are a {self.domain} expert focused on {self.market}. 
        Your eventual goal is to {self.task}.
        For now, generate domain knowledge about system-level patterns{" by answering these analysis questions" if self.with_question_guided else ""}.

        Context:
        Entities in scope: {self.entities}
        Insight types: {self.insight_str}
        {overall_questions_str}
        Task:
        {"For each category of questions, " if self.with_question_guided else ""}provide knowledge about:
        1. System-wide patterns and behaviors
        2. Strategic implications
        3. Complex causal networks
        4. Feedback systems

        Structure output as JSON:
        {{
            "<knowledge_group_name>": [
                {{
                    "key_idea": <{"analysis question" if self.with_question_guided else "keywords about the knowledge"}>,
                    "description":  <{"answer to analysis question" if self.with_question_guided else "detail about the knowledge"}>
                }},
                ...
            ],
            ...
        }}

        Provide knowledge that is:
        - Specific to {self.domain}
        - Builds on pairwise and group patterns
        - Relevant to the given entities and insight types
        - Focused on system-level dynamics
        - Strategic in nature
        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "{{") and excluding concluding remarks or follow-up suggestions (i.e., ends with "}}") 
        """
        
        knowledge = self.query_llm(prompt)
        
        self._save_json(knowledge, f"overall_knowledge{'_with_question' if self.with_question_guided else '_without_question'}.json")
            
        return knowledge

    def generate_narrative_knowledge(self) -> Dict:
        """Generate knowledge base for narrative generation"""
        prompt = f"""
        You are a {self.domain} expert focused on {self.market}. 
        Your eventual goal is to {self.task}.
        For now, generate Generate knowledge about effective narrative structures for {self.task}.

        Context:
        Domain: {self.domain}
        Task: {self.task}
        Entities being analyzed: {self.entities}

        Task:
        Provide knowledge about:
        1. Narrative structures and patterns
        2. Selected important entities to focus on
        3. Selectged significant types of insights to focus on
        4. Domain specific language such as how the value should be expressed for future prices

        Structure output as JSON:
        {{
            "narrative_structures": str
            "focus_entities": [
                {{
                    "entities_name": "name of the entity",
                    "reasoning": "reason why the entity is significant"
                }}
            ],
            "focus_insights": [
                {{
                    "insight_types": "name of the entity",
                    "reasoning": "reason why the entity is significant"
                }}
            ],
            "domain_language": [
                "domain specific rules of language"
            ]
        }}

        Provide knowledge that:
        1. Is specific to {self.domain} {self.task}
        2. Considers typical audience needs and expectations
        3. Incorporates domain-specific communication practices
        4. Addresses common challenges in data storytelling
        5. Balances detail and clarity

        Return only the JSON object by presenting the list directly without meta-commentary, introductions, or language specification (i.e., start with "{{") and excluding concluding remarks or follow-up suggestions (i.e., ends with "}}")
        """
        
        knowledge = self.query_llm(prompt)
        
        if self.output_dir:
            self._save_json(knowledge, f"narrative_knowledge.json")
            
        return knowledge


    def generate_complete_knowledge_base(self, data_schema: Dict) -> Dict:
        """Generate complete hierarchical knowledge base"""
        # 0. Entity Analysis
        # entity_questions = self.generate_entity_insight_questions(data_schema)
        # for insight_question in entity_questions:
        #     self.insight_str += f"\nType: {insight_question['insight_type']}\nDescription: {insight_question['question']}\n"

        with open(self.output_dir / f"complete_knowledge_base_with_question.json", 'r') as f:
            saved_complete_knowledge = json.load(f)

        # 1. Pairwise Analysis
        if self.with_question_guided:
            if saved_complete_knowledge["pairwise_knowledge"]:
                pairwise_knowledge = saved_complete_knowledge["pairwise_knowledge"]
            else:
                pairwise_questions = self.generate_pairwise_questions()
                pairwise_knowledge = self.generate_pairwise_knowledge(pairwise_questions)
        else:
            pairwise_knowledge = self.generate_pairwise_knowledge()

        # 2. Group Analysis
        if self.with_question_guided:
            if saved_complete_knowledge["group_knowledge"]:
                group_knowledge = saved_complete_knowledge["group_knowledge"]
            else:
                group_questions = self.generate_group_questions()
                group_knowledge = self.generate_group_knowledge(pairwise_knowledge, group_questions)
        else:
            group_knowledge = self.generate_group_knowledge(pairwise_knowledge)

        # 3. Overall Analysis
        if self.with_question_guided:
            if saved_complete_knowledge["overall_knowledge"]:
                overall_knowledge = saved_complete_knowledge["overall_knowledge"]
            else:
                overall_questions = self.generate_overall_questions()
                overall_knowledge = self.generate_overall_knowledge(pairwise_knowledge, group_knowledge, overall_questions)
        else:
            overall_knowledge = self.generate_overall_knowledge(pairwise_knowledge, group_knowledge)

        # 4. Narrative Structure
        if saved_complete_knowledge["narrative_knowledge"]:
            narrative_knowledge = saved_complete_knowledge["narrative_knowledge"]
        else:
            narrative_knowledge = self.generate_narrative_knowledge()

        # Combine all knowledge
        complete_knowledge = {
            "pairwise_knowledge": pairwise_knowledge,
            "group_knowledge": group_knowledge,
            "overall_knowledge": overall_knowledge,
            "narrative_knowledge": narrative_knowledge
        }

        if self.output_dir:
            self._save_json(complete_knowledge, f"complete_knowledge_base_with{'out' if not self.with_question_guided else ''}_question.json")

        return complete_knowledge
    
    def _save_json(self, data: Dict, filename: str) -> None:
        """Save JSON data to file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=json_serialize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="gpt-4o", type=str)
    parser.add_argument('--with_question_guided', action='store_true')
    args = parser.parse_args()
    model_name = args.model.split('\\')[-1].split('/')[-1].lower()

    output_dir = Path("results")/ "knowledge" /  model_name
    domain = "finance"
    task = "generate daily market reports that help reader understand market conditions and trends, and make investment decisions"
    data_dir = Path("data") / "datatales"
    market_dir_list = [x for x in data_dir.iterdir() if x.is_dir()]

    output_dir.mkdir(exist_ok=True, parents=True) 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger = setup_logger("KnowledgeGenerator", str(output_dir / f"log_{current_time}.txt"))

    if args.model == "gpt-4o":
        query_llm_func = partial(get_gpt_response, logger=logger)
    else:
        pipeline = load_pipeline(args.model)
        query_llm_func = partial(get_llama_response, pipeline=pipeline, logger=logger)


    for market_dir in market_dir_list:
        market = market_dir.name
        print(f"generating knowledge for {market}")
        sample_data_path = [x for x in (market_dir / "test").iterdir() if x.is_file()][0] 
        sample_df = pd.read_csv(sample_data_path)
        entities = sample_df["Product Name"].unique().tolist()
        knowledge_base_generator = KnowledgeBaseGenerator(domain=domain, task=task, market=market, entities=entities, query_llm=query_llm_func, with_question_guided=args.with_question_guided, output_dir=str(output_dir))
        data_schema = sample_df.dtypes.to_dict()
        knowledge_base_generator.generate_complete_knowledge_base(data_schema)