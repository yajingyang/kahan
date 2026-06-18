import os
import json
import re
import datetime
import openai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
import ast

load_dotenv()

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = "2024-02-15-preview"
engine = os.getenv("OPENAI_API_ENGINE")

ALLOWED_PACKAGES = {
    'data_block': ['pandas', 'numpy'],
    'data_processing': ['pandas', 'numpy', 'sklearn'],
    'scoring': ['pandas', 'numpy', 'sklearn']
}


def replace_outer_quotes(text):
    result = []
    for line in text.split('\n'):
        if not line.strip():
            result.append(line)
            continue
            
        # Find the position after the colon and any whitespace
        colon_pos = line.find(':')
        if colon_pos == -1:
            result.append(line)
            continue
            
        # Split the line into key and value parts
        key_part = line[:colon_pos + 1]
        value_part = line[colon_pos + 1:].strip()
        
        # Check if the line ends with a comma
        has_comma = value_part.endswith(',')
        if has_comma:
            value_part = value_part[:-1]
        
        # Only process if the value starts and ends with single quotes
        if value_part.startswith("'") and value_part.endswith("'"):
            # Replace only the first and last single quotes
            value_part = '"' + value_part[1:-1] + '"'
        
        # Add the comma back if it was present
        if has_comma:
            value_part += ','
            
        result.append(key_part + ' ' + value_part)
    
    return '\n'.join(result)

def extract_and_parse_json(text):
    """
    Extract JSON content from text and parse it, handling malformed quotes.
    Supports both object ({}) and array ([]) JSON structures.
    """
    # First, try to extract the JSON content
    def find_json_boundaries(s):
        # Find the first { or [ and the last } or ]
        start_brace = s.find('{')
        start_bracket = s.find('[')
        
        # Determine which comes first (if both exist)
        if start_brace == -1 and start_bracket == -1:
            raise ValueError("No JSON object or array found in the string")
        elif start_brace == -1:
            start = start_bracket
            end = s.rfind(']')
        elif start_bracket == -1:
            start = start_brace
            end = s.rfind('}')
        else:
            start = min(i for i in (start_brace, start_bracket) if i != -1)
            end = s.rfind('}') if start == start_brace else s.rfind(']')
        
        if end == -1:
            raise ValueError("No matching closing brace/bracket found")
        
        return start, end + 1

    # Extract the JSON string
    json_start, json_end = find_json_boundaries(text)
    s = text[json_start:json_end]

    s = replace_outer_quotes(s)
    
    # Now try to parse the extracted JSON
    result = json.loads(s)
    return result


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches

def setup_logger(name, log_output_path: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_output_path
    )
    logger = logging.getLogger(name)
    
    return logger

def get_gpt_response(messages, engine=engine, logger=None, return_type="json"):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]        

    num_tries = 1
    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=engine,
                messages=messages,
                max_tokens=int(os.getenv("MAX_TOKENS")),
                top_p=float(os.getenv("TOP_P")),
                frequency_penalty=float(os.getenv("FREQUENCY_PENALTY")),
                presence_penalty=float(os.getenv("PRESENCE_PENALTY")),
                stop=None
            )
            result = response.choices[0].message.content
            if return_type == "json":
                try:
                    result, _, _ = extract_and_fix_json(result, json_mark="")
                except Exception as e:
                    raise ValueError(f"Failed to extract or parse JSON: {result}")
            else:
                result = result
            break
        except openai.error.RateLimitError:
            if logger:
                logger.info('RateLimitError')
            time.sleep(20)
            continue
        except Exception as e:
            if logger:
                logger.info(f"Encounter error {e}...")
            if num_tries >= 3:
                break
            num_tries += 1

    return result


def get_llama_response(prompt, pipeline, logger, return_type="json", max_length=4000):
    num_tries = 1
    while True:
        try:
            messages = [
                {"role": "user", "content": prompt},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=max_length,
            )

            result = outputs[0]["generated_text"][-1]['content']
            if "</think>" in result: # for deepseek model
                result, _, _ = extract_text(result, text_mark="</think>")
            if return_type == "json":
                try:
                    result, _, _ = extract_and_fix_json(result, json_mark="")
                except Exception as e:
                    raise ValueError(f"Failed to extract or parse JSON: {result}")
            else:
                result = result
            break
        except Exception as e:
            logger.info(f"Error: {e}")
            logger.info(f"Current generation result:  {result}")
            logger.info(f"Retring {num_tries} times...")
            if num_tries >= 5:
                break
            num_tries += 1
 
    return result


def get_gpt_embeddings(input_str):
    response = openai.Embedding.create(
        input=input_str,
        engine="text-embedding-ada-002")
    return response['data'][0]['embedding']

class LLM(ABC):
    @abstractmethod
    def generate_heuristics(self, data_structure: Dict, user_requirements: Dict) -> List[Dict]:
        pass

    @abstractmethod
    def generate_function(self, computation_need: str) -> str:
        pass

def remove_month_labels(text):
    text = text.replace(" (front month)", "")
    text = text.replace(" (second month)", "")
    text = text.replace(" (third month)", "")
    return text

def json_serialize(obj):
    """
    JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def store_to_json(json_object, filepath):
    if isinstance(filepath, str): filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w+') as f:
        json.dump(json_object, f, indent=2, default=json_serialize)
    print(f"JSON object stored in {filepath}")

def update_json(update_object, filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            cur_list = json.load(f)
            if isinstance(update_object, list):
                cur_list.extend(update_object)
            else:
                cur_list.append(update_object)
    else:
        if isinstance(update_object, list):
            cur_list = update_object
        else:
            cur_list = [update_object]
            
    with open(filepath, 'w+') as f:
        json.dump(cur_list, f, indent=2, default=json_serialize)
    print(f"JSON object updated in {filepath}")

def get_dtypes_dict(df):
    dtype_dict = df.dtypes.to_dict()
    type_mapping = {
    'object': 'str',
    'float64': 'float', 
    'int64': 'int'
    }

    converted_types = {col: type_mapping[str(dtype)] for col, dtype in dtype_dict.items()}
    
    return converted_types

def process_string(s):
    s = s.lower()
    cleaned = re.sub(r'[^\w\-]', '-', s)
    return cleaned


def load_json_from_path(fpath):
    with fpath.open() as f:
        data = json.load(f)
    return data


def load_pipeline(model_name):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline


def process_cot(text, extract_cue="FINAL ANSWER:"):
    pattern = f'(?i){extract_cue}\s*(.*?)(?=\s*$)'
    match = re.search(pattern, text, re.DOTALL)
    result = match.group(1).strip()
    result = re.sub(r'^\n+|\n+$', '', result)
    result = re.sub(r'\n\s*\n', '\n', result)
    return result


def count_numeric_value(text):
    pattern = r'\d+(?:\.\d+)?'
    matches = re.findall(pattern, text)
    return len(matches)

def extract_and_validate_code(llm_output: str, code_mark: str="") -> tuple[str, bool, str]:
    """Extract and validate Python code from LLM output that includes chain of thought reasoning."""
    if code_mark:
        code_start = llm_output.find(code_mark)
        if code_start == -1:
            return "", False, "No code sections found in output"
    else:
        code_start = 0
    
    code = llm_output[code_start + len(code_mark):].strip()
    
    # Remove any remaining markdown code blocks
    code = re.sub(r'```python\s*', '', code)
    code = re.sub(r'```\s*', '', code)
    
    
    try:
        ast.parse(code)
    except SyntaxError as e:
        return code, False, f"Syntax error in code: {str(e)}"
    
    required_components = [
        ('argparse', "Missing argument parser implementation"),
        ('pandas', "Missing pandas import or usage")
    ]
    
    for component, error_msg in required_components:
        if component not in code:
            return code, False, error_msg
    
    return code, True, "Code successfully validated"


def extract_and_fix_json(llm_output: str, json_mark: str="FINAL ANSWER:") -> tuple[dict, bool, str]:
    """
    Extract and validate JSON from LLM output that includes chain of thought reasoning.
    Attempts to fix common JSON formatting issues.
    """
    try:
        # Find the JSON part after "FINAL ANSWER:"
        if json_mark:
            json_start = llm_output.find(json_mark)
            if json_start == -1:
                return {}, False, f"No '{json_mark}' marker found"
        else:
            json_start = 0
        
        json_str = llm_output[json_start + len(json_mark):].strip()
        
        # Common fixes for LLM-generated JSON
        def fix_json(json_str: str) -> str:
            # Replace single quotes with double quotes
            json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
            json_str = re.sub(r":'([^']*)'", r':"\1"', json_str)
            
            # Fix missing quotes around property names
            json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            
            # Fix floating point numbers
            json_str = re.sub(r':\s*(\d+\.\d+)', r':\1', json_str)
            
            # Remove trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            return json_str
        
        # Try to parse with fixes
        json_str = fix_json(json_str)
        parsed_json = json.loads(json_str)
        
        # Validate structure
        required_fields = {
            "question_interpretations": list,
            "overall_interpretation": dict
        }
        
        for field, expected_type in required_fields.items():
            if field not in parsed_json:
                return parsed_json, False, f"Missing required field: {field}"
            if not isinstance(parsed_json[field], expected_type):
                return parsed_json, False, f"Invalid type for {field}: expected {expected_type}"
        
        # Validate question interpretations
        for idx, interp in enumerate(parsed_json["question_interpretations"]):
            required_interp_fields = {
                "question": str,
                "interpretation": str,
                "significance_score": (int, float),
                "reasoning": str
            }
            
            for field, expected_type in required_interp_fields.items():
                if field not in interp:
                    return parsed_json, False, f"Missing field '{field}' in interpretation {idx}"
                if not isinstance(interp[field], expected_type):
                    return parsed_json, False, f"Invalid type for {field} in interpretation {idx}"
                
                # Validate score range
                if field == "significance_score" and not (0 <= interp[field] <= 1):
                    return parsed_json, False, f"Significance score out of range in interpretation {idx}"
        
        # Validate overall interpretation
        required_overall_fields = {
            "summary": str,
            "significance_score": (int, float),
            "reasoning": str
        }
        
        for field, expected_type in required_overall_fields.items():
            if field not in parsed_json["overall_interpretation"]:
                return parsed_json, False, f"Missing field '{field}' in overall interpretation"
            if not isinstance(parsed_json["overall_interpretation"][field], expected_type):
                return parsed_json, False, f"Invalid type for {field} in overall interpretation"
            
            # Validate score range
            if field == "significance_score" and not (0 <= parsed_json["overall_interpretation"][field] <= 1):
                return parsed_json, False, "Overall significance score out of range"
        
        return parsed_json, True, "JSON successfully validated"
    
    except json.JSONDecodeError as e:
        return {}, False, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return {}, False, f"Unexpected error: {str(e)}"
    

def extract_text(llm_output: str, text_mark: str="") -> tuple[str, bool, str]:
    """
    Extract text object from LLM output that includes chain of thought reasoning.
    """
    try:
        # Find the text part after "FINAL ANSWER:"
        if text_mark:
            text_start = llm_output.find(text_mark)
            if text_start == -1:
                return {}, False, f"No '{text_mark}' marker found"
        else:
            text_start = 0
        
        text_str = llm_output[text_start + len(text_mark):].strip()
        return text_str, True, "JSON successfully validated"
    
    except Exception as e:
        return "", False, f"Unexpected error: {str(e)}"
    