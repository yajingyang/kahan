import argparse
from pathlib import Path
from datetime import datetime
from functools import partial
import pandas as pd
import os
import sys

# --- Setup Python Path ---
# Add the script's parent directory to the path to find the 'kahan' package
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent
sys.path.insert(0, str(project_root))

try:
    # Import main components
    from kahan.utils import (
        setup_logger, 
        get_gpt_response, 
        get_llama_response, 
        load_pipeline, 
        get_dtypes_dict
    )
    from kahan.knowledge_generation import KnowledgeBaseGenerator
    from kahan.narrative_generation import NarrationGenerator
    
    # Import dependencies to ensure they are found
    import kahan.factscore.openai_lm
    import kahan.factscore.lm
    import kahan.technical_indicators
    import kahan.factscore.retrieval
    import kahan.factscore.document_store
    
except ImportError as e:
    print(f"Error: Failed to import modules: {e}")
    print(f"Please make sure you save this script ('run_kahan_agent.py') in the directory")
    print(f"that *contains* the 'kahan' folder, not inside it.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Refactored function from knowledge_generation.py ---
def run_knowledge_generation(model, with_question_guided, data_dir_base, knowledge_output_dir_base, logger):
    """
    Executes the knowledge base generation process for all markets.
    """
    print("--- Starting Knowledge Generation ---")
    logger.info("--- Starting Knowledge Generation ---")
    model_name = model.split('\\')[-1].split('/')[-1].lower()
    # The KnowledgeBaseGenerator constructor expects the base output dir
    output_dir_base = Path(knowledge_output_dir_base)
    output_dir_base.mkdir(exist_ok=True, parents=True)
    
    if model == "gpt-4o":
        query_llm_func = partial(get_gpt_response, logger=logger)
    else:
        # Assuming other models are local and need pipeline
        try:
            pipeline = load_pipeline(model)
            query_llm_func = partial(get_llama_response, pipeline=pipeline, logger=logger)
        except Exception as e:
            logger.error(f"Failed to load pipeline for model {model}: {e}")
            print(f"Error: Could not load model pipeline for {model}. Aborting knowledge generation.")
            return False

    data_dir = Path(data_dir_base)
    market_dir_list = [x for x in data_dir.iterdir() if x.is_dir()]

    for market_dir in market_dir_list:
        market = market_dir.name
        print(f"Generating knowledge for {market}...")
        logger.info(f"Generating knowledge for {market}...")
        try:
            # Find the first CSV in the /test directory to use as a sample
            sample_data_path = next((market_dir / "test").glob('*.csv'))
            sample_df = pd.read_csv(sample_data_path)
            entities = sample_df["Product Name"].unique().tolist()
            data_schema = get_dtypes_dict(sample_df)
            
            knowledge_base_generator = KnowledgeBaseGenerator(
                domain="finance", 
                task="generate daily market reports that help reader understand market conditions and trends, and make investment decisions", 
                market=market, 
                entities=entities, 
                query_llm=query_llm_func, 
                with_question_guided=with_question_guided, 
                output_dir=str(output_dir_base) # Generator will append /model_name/market
            )
            
            knowledge_base_generator.generate_complete_knowledge_base(data_schema)
            logger.info(f"Successfully generated knowledge for {market}")
        except StopIteration:
            logger.warning(f"No CSV data found in {market_dir / 'test'}, skipping knowledge gen for {market}.")
            print(f"Warning: No data found for market {market}, skipping.")
        except Exception as e:
            logger.error(f"Failed to generate knowledge for {market}: {e}", exc_info=True)
            print(f"Error generating knowledge for {market}: {e}")
            
    print("--- Knowledge Generation Complete ---")
    logger.info("--- Knowledge Generation Complete ---")
    return True


# --- Refactored function from narrative_generation.py ---
def run_narrative_generation(
    model,
    with_entity_analysis_knowledge,
    with_insight_processing_knowledge,
    with_narrative_knowledge,
    with_hierarchical_structure,
    question_based_insight_processing_knowledge,
    prompting,
    recreate_code,
    regenerate_narrative,
    reload_insight_dir,
    reuse_insight_group,
    knowledge_model_name,
    market_group,
    data_dir_base,
    knowledge_dir_base,
    results_dir_base,
    logger
):
    """
    Executes the narrative generation process for all markets and data files.
    """
    print("--- Starting Narrative Generation ---")
    logger.info("--- Starting Narrative Generation ---")
    
    model_name = model.split('/')[-1].lower()
    knowledge_model_name = knowledge_model_name.split('/')[-1].lower() if knowledge_model_name else model_name
    
    # Build the setup string for result directory
    setup_str = f"{model_name}{'_entity_klg' if with_entity_analysis_knowledge else ''}" \
                f"{'_process_klg' if with_insight_processing_knowledge else ''}" \
                f"{'_qst_based' if question_based_insight_processing_knowledge else ''}" \
                f"{'_narrative_klg' if with_narrative_knowledge else ''}" \
                f"{'_hierarchy' if with_hierarchical_structure else ''}" \
                f"{'_'+prompting if prompting else ''}" \
                f"{'_' + '_'.join(with_hierarchical_structure) if (with_hierarchical_structure and (len(with_hierarchical_structure)<3) ) else ''}"
    
    if model_name != knowledge_model_name:
        setup_str += f"_{knowledge_model_name}_klg"
        
    result_dir = Path(results_dir_base) / setup_str 
    result_dir.mkdir(exist_ok=True, parents=True) 

    if model == "gpt-4o":
        query_llm_func = partial(get_gpt_response, logger=logger)
    else:
        try:
            pipeline = load_pipeline(model)
            query_llm_func = partial(get_llama_response, pipeline=pipeline, logger=logger)
        except Exception as e:
            logger.error(f"Failed to load pipeline for model {model}: {e}")
            print(f"Error: Could not load model pipeline for {model}. Aborting narrative generation.")
            return False

    data_dir = Path(data_dir_base)
    market_dir_list = [x for x in data_dir.iterdir() if x.is_dir()]
    market_dir_list.sort()

    for market_dir in market_dir_list:
        market = market_dir.name
        if market_group and market not in market_group:
            continue
            
        print(f"Processing narratives for {market}...")
        logger.info(f"Processing narratives for {market}...")
            
        data_path_list = sorted([x for x in (market_dir / "test").iterdir() if x.is_file() and x.suffix == '.csv'])
        knowledge_dir = str(Path(knowledge_dir_base) / knowledge_model_name / market)
        
        # Check if knowledge dir exists
        if not Path(knowledge_dir).exists():
            logger.warning(f"Knowledge directory not found for {market}: {knowledge_dir}")
            print(f"Warning: Knowledge directory not found for {market}. Skipping narrative generation.")
            continue
            
        computation_code_created = False
        
        for data_path in data_path_list:
            report_date = data_path.stem
            output_dir = str(result_dir / market / report_date)
            
            cur_reload_insight_dir = ""
            if reload_insight_dir:
                cur_reload_insight_dir = str(Path(reload_insight_dir) / market / report_date)

            if (Path(output_dir) / "narrative.txt").exists() and not regenerate_narrative:
                print(f"  Skipping {market}/{report_date}, narrative already exists.")
                logger.info(f"Skipping {market}/{report_date}, narrative already exists.")
                continue

            if regenerate_narrative and (Path(output_dir) / "narrative.txt").exists():
                try:
                    (Path(output_dir) / "narrative.txt").unlink()
                except OSError as e:
                    logger.error(f"Could not delete existing narrative: {e}")

            try:
                narration_generator = NarrationGenerator(
                    data_file=data_path, 
                    domain="finance market analysis", 
                    market=market,
                    task="generate daily market report that help reader understand market conditions and trends, and make investment decisions",
                    report_date=report_date,
                    entity_col="Product Name",
                    query_llm=query_llm_func,
                    with_entity_analysis_knowledge=with_entity_analysis_knowledge,
                    with_insight_processing_knowledge=with_insight_processing_knowledge,
                    with_narrative_knowledge=with_narrative_knowledge,
                    with_hierarchical_structure=with_hierarchical_structure,
                    question_based_insight_processing_knowledge=question_based_insight_processing_knowledge,
                    prompting=prompting,
                    knowledge_dir=knowledge_dir,
                    output_dir=output_dir,
                    recreate_code=recreate_code,
                    reload_insight_dir=cur_reload_insight_dir,
                    reuse_insight_group=reuse_insight_group,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"Failed to initialize NarrationGenerator for {market}/{report_date}: {e}", exc_info=True)
                print(f"Error initializing generator for {market}/{report_date}: {e}")
                continue
            
            # Generate computation code once per market
            if not computation_code_created:
                try:
                    print(f"  Generating computation code for {market}...")
                    logger.info(f"Generating computation code for {market}...")
                    narration_generator.generate_all_computation_code()
                    computation_code_created = True
                    # If recreating code, force it for next market
                    if recreate_code:
                         computation_code_created = False
                    print(f"  Computation code generation complete for {market}.")
                    logger.info(f"Computation code generation complete for {market}.")
                except Exception as e:
                    logger.error(f"Failed to create computation code for {market}: {e}", exc_info=True)
                    print(f"  Failed to create computation code for {market}, skipping market. Error: {e}")
                    break # Skip this market if code gen fails

            # Run narrative generation for the data file
            if computation_code_created:
                try:
                    print(f"  Running narrative generation for {market}/{report_date}...")
                    logger.info(f"Running narrative generation for {market}/{report_date}...")
                    narration_generator.run()
                    print(f"  Finished narrative generation for {market}/{report_date}.")
                    logger.info(f"Finished narrative generation for {market}/{report_date}.")
                except Exception as e:
                    logger.error(f"Failed to run narrative generation for {market}/{report_date}: {e}", exc_info=True)
                    print(f"  Error running narrative generation for {market}/{report_date}: {e}")
            else:
                print(f"  Skipping narrative run for {market}/{report_date} due to code generation failure.")
                logger.warning(f"Skipping narrative run for {market}/{report_date} due to code generation failure.")
    
    print("--- Narrative Generation Complete ---")
    logger.info("--- Narrative Generation Complete ---")
    return True


def main():
    parser = argparse.ArgumentParser(description="Kahan Model Agent: Auto-run knowledge and narrative generation.")
    
    # --- General arguments ---
    parser.add_argument('--model', default="gpt-4o", type=str, help="Model name (e.g., 'gpt-4o', 'llama-3.1-8b-instruct')")
    parser.add_argument('--data_dir_base', default="data/datatales", type=str, help="Base directory for input data (e.g., 'data/datatales')")
    parser.add_argument('--knowledge_dir_base', default="results/knowledge", type=str, help="Base directory to save/load knowledge")
    parser.add_argument('--results_dir_base', default="results", type=str, help="Base directory to save narrative results")
    parser.add_argument('--skip_knowledge_generation', action='store_true', help="Skip knowledge generation step (e.g., if it already exists)")
    parser.add_argument('--skip_narrative_generation', action='store_true', help="Skip narrative generation step")

    # --- Knowledge generation arguments ---
    parser.add_argument('--with_question_guided', action='store_true', help="Use question-guided knowledge generation")

    # --- Narrative generation arguments ---
    parser.add_argument('--no_entity_analysis_knowledge', dest='with_entity_analysis_knowledge', action='store_false', help="Disable entity analysis knowledge")
    parser.add_argument('--no_insight_processing_knowledge', dest='with_insight_processing_knowledge', action='store_false', help="Disable insight processing knowledge")
    parser.add_argument('--no_narrative_knowledge', dest='with_narrative_knowledge', action='store_false', help="Disable narrative knowledge")
    parser.add_argument('--with_hierarchical_structure', nargs='*', type=str, help="Hierarchical structure levels (default: 'pairwise' 'group' 'overall')")
    parser.add_argument('--no_question_based_insight_processing', dest='question_based_insight_processing_knowledge', action='store_false', help="Disable question-based insight processing")
    parser.add_argument('--prompting', default="", type=str, help="Prompting type (e.g., 'cot', 'dp')")
    parser.add_argument('--recreate_code', action='store_true', help="Force recreation of precomputation code for narratives")
    parser.add_argument('--regenerate_narrative', action='store_true', help="Force regeneration of narratives even if they exist")
    parser.add_argument('--reload_insight_dir', default="", type=str, help="Directory to reload insights from (skips insight generation)")
    parser.add_argument('--reuse_insight_group', nargs='*', type=str, help="Insight groups to reuse (e.g., 'entity', 'pairwise')")
    parser.add_argument('--knowledge_model_name', default='', type=str, help="Model name for knowledge (if different from main model)")
    parser.add_argument('--market_group', nargs='*', type=str, help="Specific markets to run for (default: all)")
    
    # Set defaults based on the original scripts
    parser.set_defaults(
        with_entity_analysis_knowledge=True,
        with_insight_processing_knowledge=True,
        with_narrative_knowledge=True,
        question_based_insight_processing_knowledge=True,
        with_question_guided=True,
        with_hierarchical_structure=['pairwise', 'group', 'overall']
    )

    args = parser.parse_args()

    # --- Setup main agent logger ---
    agent_log_dir = Path(args.results_dir_base) / "agent_logs"
    agent_log_dir.mkdir(exist_ok=True, parents=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = setup_logger("KahanAgent", str(agent_log_dir / f"agent_run_{current_time}.txt"))
    
    logger.info(f"Kahan Agent started with args: {args}")
    print(f"Kahan Agent started. Log file at: {agent_log_dir / f'agent_run_{current_time}.txt'}")

    # --- Step 1: Knowledge Generation ---
    if not args.skip_knowledge_generation:
        run_knowledge_generation(
            model=args.model,
            with_question_guided=args.with_question_guided,
            data_dir_base=args.data_dir_base,
            knowledge_output_dir_base=args.knowledge_dir_base,
            logger=logger
        )
    else:
        print("Skipping knowledge generation as requested.")
        logger.info("Skipping knowledge generation as requested.")

    # --- Step 2: Narrative Generation ---
    if not args.skip_narrative_generation:
        run_narrative_generation(
            model=args.model,
            with_entity_analysis_knowledge=args.with_entity_analysis_knowledge,
            with_insight_processing_knowledge=args.with_insight_processing_knowledge,
            with_narrative_knowledge=args.with_narrative_knowledge,
            with_hierarchical_structure=args.with_hierarchical_structure,
            question_based_insight_processing_knowledge=args.question_based_insight_processing_knowledge,
            prompting=args.prompting,
            recreate_code=args.recreate_code,
            regenerate_narrative=args.regenerate_narrative,
            reload_insight_dir=args.reload_insight_dir,
            reuse_insight_group=args.reuse_insight_group,
            knowledge_model_name=args.knowledge_model_name,
            market_group=args.market_group,
            data_dir_base=args.data_dir_base,
            knowledge_dir_base=args.knowledge_dir_base,
            results_dir_base=args.results_dir_base,
            logger=logger
        )
    else:
        print("Skipping narrative generation as requested.")
        logger.info("Skipping narrative generation as requested.")
        
    print("Kahan Agent run finished.")
    logger.info("Kahan Agent run finished.")

if __name__ == "__main__":
    main()