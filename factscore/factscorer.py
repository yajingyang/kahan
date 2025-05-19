import argparse
import string
import json
import numpy as np
import os
import logging

from tqdm import tqdm
from pathlib import Path
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import Retrieval

class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+ChatGPT",
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "retrieval+gpt-4o", "retrieval+gpt-4o-mini", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"))
        elif any(["gpt-4o" in model_name, "o1-mini" in model_name, "o3-mini" in model_name]):
            llm_model = model_name.split("+")[1]
            self.lm = OpenAIModel(llm_model,
                        cache_file=os.path.join(cache_dir, f"{llm_model}.pkl"))
        else:
            self.lm = None

        self.atomic_facts_path = Path(self.cache_dir) / f"atomic_facts-{model_name}.json"
        self.result_cache_path = Path(self.cache_dir) / f"decision_cache-{model_name}.json"
        self.atomic_fact_cache, self.result_cache = self.load_cache()

    def save_cache(self):
        # if self.lm:
        #     self.lm.save_cache()
        # if "npm" in self.model_name:
        #     for k, v in self.npm.items():
        #         v.save_cache()
        # for k, v in self.retrieval.items():
        #     v.save_cache()
        with open(self.result_cache_path, 'w+') as f:
            json.dump(self.result_cache, f, indent=2)

    def save_af_cache(self):
        with open(self.atomic_facts_path, 'w+') as f:
            json.dump(self.atomic_fact_cache, f, indent=2)

    def load_cache(self):
        if self.atomic_facts_path.exists():
            with open(self.atomic_facts_path, 'r', encoding="utf-8") as f:
                atomic_fact_cache = json.load(f)
        else:
            atomic_fact_cache = {}

        if self.result_cache_path.exists():
            with open(self.result_cache_path, 'r', encoding="utf-8") as f:
                result_cache = json.load(f)
        else:
            result_cache = {}
        return atomic_fact_cache, result_cache

    def register_knowledge_source(self, name="enwiki-financial_markets-20250319", wiki_content_dir=None, doc_db_path=None, vector_db_dir=None):
        assert name not in self.retrieval, f"{name} already registered"

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.retrieval[name] = Retrieval(
            wiki_content_dir=wiki_content_dir,
            doc_db_path=doc_db_path,
            vector_db_dir=vector_db_dir,
            # retrieval_method=retrieval_method,  # Can be "bm25", "embedding", or "vector"
            embedding_model_name_or_path=".cache\\factscore\\model\\sentence-transformers\\all-MiniLM-L6-v2",
            cache_path=cache_path,
            embed_cache_path=embed_cache_path,
            force_rebuild=False  # Set to True to rebuild databases
        )

        # if "npm" in self.model_name:
        #     cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
        #     embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
        #     self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
        #                          "npm-single",
        #                          cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))

    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.002
        elif model == "gpt-4o":
            rate = 0.002
        elif model == "gpt-4o-mini":
            rate = 0.00016
        elif model == "o3-mini":
            rate = 0.0015

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(self,
                  generations,
                  generation_ids,
                  topics=None,
                  gamma=10,
                  atomic_facts=None,
                  verification_context_list=None,
                  knowledge_source="enwiki-financial_markets-20250319",
                  wiki_content_dir="finance_wiki_content",
                  doc_db_path=None,
                  vector_db_dir=None,
                  retrieval_method=None,
                  verbose=False):


        if retrieval_method == "vector":
            assert vector_db_dir, "Vector DB directory needed for vector based retrieval!"
            knowledge_key = f"{knowledge_source}_vectordb"
            self.register_knowledge_source(knowledge_key, wiki_content_dir=wiki_content_dir, vector_db_dir=vector_db_dir)
        # elif retrieval_method == "hybrid":
        #     self.register_knowledge_source(knowledge_source, wiki_content_dir=wiki_content_dir, doc_db_path=doc_db_path, vector_db_dir=vector_db_dir)
        else:
            assert doc_db_path, "Document DB path needed for embedding or BM25 based retrieval!"
            knowledge_key = f"{knowledge_source}_docdb"
            self.register_knowledge_source(knowledge_key, wiki_content_dir=wiki_content_dir, doc_db_path=doc_db_path)

        if topics is None and type(generations)==list:
            topics = [None] * len(generations)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(demon_dir=os.path.join(self.data_dir, "demos"),
                                                        cache_file=os.path.join(self.cache_dir, "atomic_facts.pkl"))

            # # estimate the total cost of atomic fact generation
            # total_words = 0
            # for gen in generations:
            #     total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            # self.print_cost_estimates(total_words, task="atomic fact generation", model="gpt-4o-mini")

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                # response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                # if response_abstained:
                #     atomic_facts.append(None)
                #     continue
                # continue only when the response is not abstained
                cache_af = self.atomic_fact_cache.get(gen, None)
                if cache_af:
                    curr_afs = cache_af
                else:
                    curr_afs, _ = self.af_generator.run(gen)
                    curr_afs = [fact for _, facts in curr_afs for fact in facts]
                    self.atomic_fact_cache[gen] = curr_afs
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()
                    self.save_af_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()
            self.save_af_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        # if any(["ChatGPT" in self.model_name, "o3-mini" in self.model_name, "gpt-4o-mini" in self.model_name]):
        #     # estimate the total cost of response generation
        #     total_words = 0
        #     for topic, verification_context, facts in zip(topics, verification_context_list, atomic_facts):
        #         if facts is not None:
        #             total_words += self._get_score(topic, facts, verification_context, date=date, knowledge_source=knowledge_key, cost_estimate=self.cost_estimate)

        #     if "ChatGPT" in self.model_name:
        #         self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")
        #     elif "o3-mini" in self.model_name:
        #         self.print_cost_estimates(total_words, task="factscore evaluation", model="o3-mini")
        #     elif "gpt-4o-mini" in self.model_name:
        #         self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-4o-mini")

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        for topic, gen, verification_context, facts, gen_id in zip(topics, generations, verification_context_list, atomic_facts, generation_ids):
            if facts is None:
                decisions.append(None)
            else:
                if gen in self.result_cache:
                    decision = self.result_cache[gen]['decision']
                    score = self.result_cache[gen]['score']
                else:
                    decision = self._get_score(topic, facts, verification_context, knowledge_source=knowledge_key)
                    score = np.mean([d["is_supported"] for d in decision])
                
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                self.result_cache[gen] = {'decision': decision, 'score': score, 'id': gen_id}
                self.save_cache()

        out = {"score": np.mean(scores),
               "n_examples": len(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _get_score(self, topic, atomic_facts, verification_context, knowledge_source, cost_estimate=None):
        decisions = []
        total_words = 0
        for atom in atomic_facts:
            atom = atom.strip()
            if topic:
                query = (topic, atom)
            else:
                query = atom

            if self.lm:
                cache_key = f"ATOMIC_FACT:{atom}NUM_CONTEXT:{verification_context}"            
                cache_result = self.lm.get_cache_result(cache_key)
                if cache_result:
                    output = cache_result
                else:
                    passages = self.retrieval[knowledge_source].get_passages(query, k=5)
                    definition = f"Answer the question{' about ' + topic if topic else ''} based on the given context.\n\n"
                    numerical_verification_intruction = \
                    """\n When verifying numerical statements, consider reasonable rounding. A statement should be considered true if:
                        1. The stated value is a rounded version of the exact value
                        2. The difference between the stated value and the exact value is within ±0.01 for percentages
                        3. The stated value matches the exact value to the number of significant digits presented\n"""
                    ambiguous_statements_verification_intruction = \
                    """\nWhen verifying statements that don't specify an entity, consider the statement TRUE if it applies to ANY of the relevant entities in the data.
                    Example:
                    Statement: "The closing price is above key moving averages."
                    Data: 
                    - Brent: closing=$100.54, SMA20=$104.01, SMA50=$111.58
                    - Natural Gas: closing=$7.71, SMA20=$7.42, SMA50=$7.59
                    - WTI: closing=$94.42, SMA20=$98.4, SMA50=$107.29
                    Correct verification: TRUE (because it applies to Natural Gas, even though not to Brent or WTI)\n"""
                    context = "\nPrimary context for factuality check:\n" + verification_context + numerical_verification_intruction + ambiguous_statements_verification_intruction + "\n\nAdditional domain knowledge:\n" if verification_context else ""
                    for psg_idx, psg in enumerate(reversed(passages)):
                        context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                    definition += context.strip()
                    if not definition[-1] in string.punctuation:
                        definition += "."
                    prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())

                    if cost_estimate:
                        if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                            total_words += len(prompt.split())
                        elif cost_estimate == "ignore_cache":
                            total_words += len(prompt.split())
                        continue

                    output = self.lm.generate(prompt, cache_key=cache_key)

                if type(output[1])==np.ndarray:
                    # when logits are available
                    logits = np.array(output[1])
                    assert logits.shape[0] in [32000, 32001]
                    true_score = logits[5852]
                    false_score = logits[7700]
                    is_supported = true_score > false_score
                else:
                    # when logits are unavailable
                    generated_answer = output[0].lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            else:
                is_supported = True

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3

            decisions.append({"atom": atom, "is_supported": is_supported})

        if cost_estimate:
            return total_words
        else:
            return decisions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+gpt-4o-mini")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)

    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")    
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.n_samples is not None and tot==args.n_samples:
                break
    out = fs.get_score(topics=topics,
                       generations=generations,
                       gen_ids=['']*len(generations),
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    # Save out as a json file
    with open(args.input_path.replace(".jsonl", f"_factscore_output.json"), 'w') as f:
        f.write(json.dumps(out) + "\n")

