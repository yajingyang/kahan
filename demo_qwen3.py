"""
Demo: run the KAHAN pipeline end-to-end with Qwen3-8B on a single market.

This is a smoke test (not a full experiment). It:
  1. Loads the Qwen3-8B text-generation pipeline once.
  2. Builds the hierarchical knowledge base for ONE market.
  3. Generates a narrative for the first few test dates of that market.
  4. Prints the generated narrative(s).

Run from the repository root:

    python demo_qwen3.py                              # market=corn, 1 date
    python demo_qwen3.py --market gold --num_dates 2
    python demo_qwen3.py --skip_knowledge_generation  # reuse an existing KB

Requirements:
  * The packages in requirements.txt (transformers, torch, pandas, ...).
  * A CUDA GPU with enough memory to hold an 8B model in bfloat16.
  * No OpenAI/Azure credentials are needed for local models.

Outputs:
  * Knowledge base -> results/knowledge/<model_name>/<market>/
  * Narratives     -> results/<setup_str>/<market>/<date>/narrative.txt
"""
import argparse
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

from utils import setup_logger, get_llama_response, load_pipeline, get_dtypes_dict
from knowledge_generation import KnowledgeBaseGenerator
from narrative_generation import NarrationGenerator

DOMAIN = "finance"
KNOWLEDGE_TASK = ("generate daily market reports that help reader understand market "
                  "conditions and trends, and make investment decisions")
NARRATIVE_TASK = ("generate daily market report that help reader understand market "
                  "conditions and trends, and make investment decisions")
ENTITY_COL = "Product Name"


def main():
    parser = argparse.ArgumentParser(description="KAHAN demo with Qwen3-8B on one market.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="HuggingFace model id (default: Qwen/Qwen3-8B).")
    parser.add_argument("--market", default="corn",
                        help="Market under --data_dir_base to run (default: corn).")
    parser.add_argument("--num_dates", type=int, default=1,
                        help="Number of test dates to narrate (default: 1).")
    parser.add_argument("--data_dir_base", default="data/datatales")
    parser.add_argument("--knowledge_dir_base", default="results/knowledge")
    parser.add_argument("--results_dir_base", default="results")
    parser.add_argument("--skip_knowledge_generation", action="store_true",
                        help="Reuse an existing knowledge base instead of building it.")
    parser.add_argument("--regenerate_narrative", action="store_true",
                        help="Regenerate narratives even if they already exist.")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1].lower()  # e.g. "qwen3-8b"

    log_dir = Path(args.results_dir_base) / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = setup_logger("KahanDemo", str(log_dir / f"demo_{model_name}_{ts}.txt"))

    test_dir = Path(args.data_dir_base) / args.market / "test"
    if not test_dir.is_dir():
        raise SystemExit(f"No test data at {test_dir}. Check --data_dir_base / --market.")
    test_csvs = sorted(p for p in test_dir.iterdir() if p.suffix == ".csv")
    if not test_csvs:
        raise SystemExit(f"No CSV files under {test_dir}.")

    print(f"Loading {args.model} (this can take a while on first run)...")
    pipeline = load_pipeline(args.model)
    query_llm = partial(get_llama_response, pipeline=pipeline, logger=logger)

    # ---- Stage 1: knowledge base -> results/knowledge/<model_name>/<market> ----
    knowledge_out_base = Path(args.knowledge_dir_base) / model_name
    knowledge_dir = knowledge_out_base / args.market
    if args.skip_knowledge_generation:
        print(f"Skipping knowledge generation; expecting it at {knowledge_dir}")
    else:
        print(f"[1/2] Generating knowledge base for '{args.market}'...")
        sample_df = pd.read_csv(test_csvs[0])
        entities = sample_df[ENTITY_COL].unique().tolist()
        kb = KnowledgeBaseGenerator(
            domain=DOMAIN,
            task=KNOWLEDGE_TASK,
            market=args.market,
            entities=entities,
            query_llm=query_llm,
            with_question_guided=True,
            output_dir=str(knowledge_out_base),  # generator appends /<market>
        )
        kb.generate_complete_knowledge_base(get_dtypes_dict(sample_df))
        print(f"      Knowledge base written to {knowledge_dir}")

    if not knowledge_dir.exists():
        raise SystemExit(f"Knowledge directory missing: {knowledge_dir}. "
                         f"Run once without --skip_knowledge_generation first.")

    # ---- Stage 2: narrative generation (full KAHAN configuration) ----
    # Mirrors run_kahan_agent's all-knowledge + full-hierarchy setup string.
    setup_str = f"{model_name}_entity_klg_process_klg_qst_based_narrative_klg_hierarchy"
    result_dir = Path(args.results_dir_base) / setup_str

    narratives = []
    computation_code_created = False
    for data_path in test_csvs[: args.num_dates]:
        report_date = data_path.stem
        output_dir = str(result_dir / args.market / report_date)
        narrative_path = Path(output_dir) / "narrative.txt"

        if narrative_path.exists() and not args.regenerate_narrative:
            print(f"[2/2] Narrative already exists for {report_date} "
                  f"(use --regenerate_narrative to redo).")
            narratives.append((report_date, narrative_path.read_text()))
            continue

        print(f"[2/2] Generating narrative for {args.market}/{report_date}...")
        gen = NarrationGenerator(
            data_file=data_path,
            domain="finance market analysis",
            market=args.market,
            task=NARRATIVE_TASK,
            report_date=report_date,
            entity_col=ENTITY_COL,
            query_llm=query_llm,
            with_entity_analysis_knowledge=True,
            with_insight_processing_knowledge=True,
            with_narrative_knowledge=True,
            with_hierarchical_structure=["pairwise", "group", "overall"],
            question_based_insight_processing_knowledge=True,
            prompting="",
            knowledge_dir=str(knowledge_dir),
            output_dir=output_dir,
            recreate_code=False,
            reload_insight_dir="",
            reuse_insight_group=None,
            logger=logger,
        )
        if not computation_code_created:
            print("      Generating precomputation code (once per market)...")
            gen.generate_all_computation_code()
            computation_code_created = True
        gen.run()
        text = narrative_path.read_text() if narrative_path.exists() else "<no narrative produced>"
        narratives.append((report_date, text))

    print("\n" + "=" * 80)
    for report_date, text in narratives:
        print(f"\n### {args.market} {report_date}\n")
        print(text)
    print("\n" + "=" * 80)
    print(f"Done. Narratives under {result_dir / args.market}/")


if __name__ == "__main__":
    main()
