# KAHAN: Knowledge-Augmented Hierarchical Analysis and Narration

This repository provides the official implementation for the EMNLP 2025 Findings paper:
**[KAHAN: Knowledge-Augmented Hierarchical Analysis and Narration for Financial Data Narration](https://aclanthology.org/2025.findings-emnlp.1405/)**

KAHAN is a knowledge-augmented, hierarchical framework that systematically extracts insights from raw tabular financial data. It operates by analyzing data at four levels: **entity**, **pairwise**, **group**, and **system**.

The framework uniquely leverages Large Language Models (LLMs) as domain experts to first build a structured knowledge base, and then to generate coherent, multi-level narratives grounded in that knowledge.

-----

## Project Structure

  * `run_kahan_agent.py`: A top-level agent script to automate the entire pipeline.
  * `kahan/`: The core package containing all logic.
      * `knowledge_generation.py`: Script for building the hierarchical knowledge base (Stage 1).
      * `narrative_generation.py`: Script for generating the final data-grounded narratives (Stage 2).
      * `factuality_evaluation.py`: Script to run factual consistency evaluation (using FActScore).
      * `aspect_based_evaluation.py`: Script to run qualitative, aspect-based evaluation (LLM-as-judge).
      * `factscore/`: Module for factual consistency checking, adapted from FActScore.
      * `technical_indicators.py`: Utility script for pre-calculating financial metrics.
      * `utils.py`: Shared helper functions, LLM API calls, and JSON processing.
  * `data/datatales/`: Directory containing the benchmark data from **DataTales** ([aclanthology.org/2024.emnlp-main.601](https://aclanthology.org/2024.emnlp-main.601/)).

-----

## Setup and Installation

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/kahan.git
cd kahan
```

### 2\. Install Dependencies

Install the required Python packages using the provided `requirements.txt` file (which will be added to the repo).

```bash
pip install -r requirements.txt
```

### 3\. Set Up Environment Variables

This project uses API keys managed via a `.env` file.

1.  Create a file named `.env` in the root of the project directory.
2.  Copy the contents of the [`.env.template`](#envtemplate) section below into your new `.env` file and fill in your API details.

### 4\. Prepare Data

The project requires two sets of data to be in place before running.

1.  **FActScore Knowledge Base:** The factual evaluation relies on a Wikipedia database. Run the following script to download the necessary demo files and the `enwiki-20230401.db` database.

    ```bash
    python kahan/factscore/download_data.py
    ```

2.  **Ground-Truth Metrics:** You must pre-calculate the technical indicators (e.g., SMA, RSI) for all financial CSVs. This script processes the raw data from `data/datatales` and saves the numerical results to `results/metric_values/`. These results are used as the ground truth for both narrative generation and fact-checking.

    ```bash
    python kahan/technical_indicators.py
    ```

-----

## How to Run the KAHAN Pipeline

### Step 1: Knowledge Generation

First, run the agent to build the knowledge base for all markets. This script will scan the `data/datatales/` directory, identify the markets, and generate a corresponding knowledge base in `results/knowledge/`.

```bash
# Run knowledge generation using the default model (gpt-4o)
# This step can be skipped in future runs.
python run_kahan_agent.py --skip_narrative_generation
```

### Step 2: Narrative Generation

Once the knowledge base exists, run the agent again to generate the narratives. This will use the knowledge from Step 1 to create narratives for each data file in `data/datatales/*/test/`.

```bash
# Run narrative generation, skipping the knowledge-gen step
python run_kahan_agent.py --skip_knowledge_generation
```

### Example: Run Full Pipeline with a Local Model

This example uses `llama-3.1-8b-instruct` for both steps and forces the re-generation of any existing narratives.

```bash
python run_kahan_agent.py \
    --model "llama-3.1-8b-instruct" \
    --regenerate_narrative
```

-----

## How to Run Evaluation

After generating narratives, you can evaluate them for factuality and quality.

### 1\. Factual Consistency Evaluation

This script uses the **FActScore** method ([github.com/shmsw25/FActScore](https://github.com/shmsw25/FActScore)) to verify narrative claims.

**To Run:**

1.  Edit the `setup_list` at the bottom of `kahan/factuality_evaluation.py` to match the model(s) you generated.
2.  Execute the script:
    ```bash
    python kahan/factuality_evaluation.py
    ```
3.  Results will be saved in `results/factuality_evaluation.json`.

### 2\. Aspect-Based Quality Evaluation

This script uses an LLM-as-judge based on the **Divide-and-Conquer (DnA)** framework ([aclanthology.org/2025.coling-main.156/](https://aclanthology.org/2025.coling-main.156/)).

**To Run:**

1.  Ensure you have an evaluation criteria file at `results/evaluation_criteria_final.json` (the script can also generate this).
2.  Edit the `setup_list` at the bottom of `kahan/aspect_based_evaluation.py` to match the model(s) you want to compare.
3.  Execute the script:
    ```bash
    python kahan/aspect_based_evaluation.py
    ```
4.  Aggregated results will be saved to a `.csv` file in the `results/` directory.

-----

## .env.template

Create a `.env` file in the root directory and add the following, filling in your API credentials.

```ini
# --- OpenAI/Azure API Configuration ---
# Set to 'azure' if using Azure, or 'openai' if using standard OpenAI
OPENAI_API_TYPE="azure"

# Your Azure endpoint (e.g., https://your-endpoint.openai.azure.com/)
OPENAI_API_BASE="your-api-base-url"

# Your API Key
OPENAI_API_KEY="your-api-key"

# API version for generation (e.g., 2024-02-15-preview)
OPENAI_API_VERSION="2024-02-15-preview"

# API version for factuality (can be the same or different)
OPENAI_API_FACTUALITY_API_VERSION="2024-02-15-preview"

# Deployment/Engine name for the main generation model (e.g., "gpt-4o")
OPENAI_API_ENGINE="gpt-4o"

# --- LLM Generation Parameters ---
MAX_TOKENS=4000
TOP_P=0.9
FREQUENCY_PENALTY=0.0
PRESENCE_PENALTY=0.0
```

-----

## Citations

If you use this work, please cite the KAHAN paper. The data is from the DataTales benchmark.

```bibtex
@inproceedings{yang-etal-2025-kahan,
    title = "{KAHAN}: Knowledge-Augmented Hierarchical Analysis and Narration for Financial Data Narration",
    author = "Yang, Yajing  and
      Deng, Tony  and
      Kan, Min-Yen",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = dec,
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1405"
}

@inproceedings{yang-etal-2024-datatales,
    title = "{D}ata{T}ales: A Benchmark for Real-World Intelligent Data Narration",
    author = "Yang, Yajing  and
      Deng, Tony  and
      Kan, Min-Yen",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2024",
    address = "Miami, Florida",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.601",
    pages = "8723--8738"
}
```