import json
import numpy as np
import re
import functools
import string
import spacy
import sys
import nltk
import openai
from rank_bm25 import BM25Okapi
import os
import time
from nltk.tokenize import sent_tokenize

from factscore.openai_lm import OpenAIModel

nltk.download("punkt")


class AtomicFactGenerator(object):
    def __init__(self, demon_dir, model_name="gpt-4o-mini", cache_file=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = False
        self.demon_path = os.path.join(demon_dir, "demons.json" if self.is_bio else "demons_complex.json")

        self.openai_lm = OpenAIModel(model_name, cache_file=cache_file)

        # get the demos
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_cache(self):
        self.openai_lm.save_cache()

    def run(self, generation, cost_estimate=None):
        """Convert the generation into a set of atomic facts. Return a total words cost if cost_estimate != None."""
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        return self.get_atomic_facts_from_paragraph(paragraphs, cost_estimate=cost_estimate)

    def get_atomic_facts_from_paragraph(self, paragraphs, cost_estimate=None):
        sentences = []
        para_breaks = []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0 :
                para_breaks.append(len(sentences))

            initials = detect_initials(paragraph)

            curr_sentences = sent_tokenize(paragraph)
            curr_sentences_2 = sent_tokenize(paragraph)

            curr_sentences = fix_sentence_splitter(curr_sentences, initials)
            curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

            # checking this, just to ensure the crediability of the sentence splitter fixing algorithm
            assert curr_sentences == curr_sentences_2, (paragraph, curr_sentences, curr_sentences_2)

            sentences += curr_sentences

        atoms_or_estimate = self.get_init_atomic_facts_from_sentence([sent for i, sent in enumerate(sentences) if not (not self.is_bio and ( \
                            (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                            (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))))], cost_estimate=cost_estimate)

        if cost_estimate:
            return atoms_or_estimate
        else:
            atoms = atoms_or_estimate

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            if not self.is_bio and ( \
                (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))):
                atomic_facts_pairs.append((sent, []))
            elif self.is_bio and sent.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((sent, []))
            elif sent.startswith("Sure") or sent.startswith("Please") or (i==0 and sent.startswith("Here are")):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        # postprocess_atomic_facts will fix minor issues from InstructGPT
        # it is supposed to handle sentence splitter issue too, but since here
        # we fixed sentence splitter issue already,
        # the new para_breaks should be identical to the original para_breaks
        if self.is_bio:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)

        return atomic_facts_pairs, para_breaks


    def get_init_atomic_facts_from_sentence(self, sentences, cost_estimate=None):
        """Get the initial atomic facts from the sentences. Return a total words cost if cost_estimate != None."""

        is_bio = self.is_bio
        demons = self.demons

        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        prompts = []
        prompt_to_sent = {}
        atoms = {}
        for sentence in sentences:
            if sentence in atoms:
                continue
            top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            prompt = ""

            for i in range(n):
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(list(demons.keys())[i])
                for fact in demons[list(demons.keys())[i]]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"

            for match in top_machings:
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(match)
                for fact in demons[match]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"
            prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(sentence)
            prompts.append(prompt)
            prompt_to_sent[prompt] = sentence

        if cost_estimate:
            total_words_estimate = 0
            for prompt in prompts:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
                    continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            for prompt in prompts:
                output, _, _ = self.openai_lm.generate(prompt, cache_key=prompt)
                atoms[prompt_to_sent[prompt]] = text_to_sentences(output)

            for key, value in demons.items():
                if key not in atoms:
                    atoms[key] = value

            return atoms


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_machings


# transform InstructGPT output into sentences
def text_to_sentences(text):
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if len(sentences) > 0: 
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.' 
    else:
        sentences = []
    return sentences


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False

def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)


    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)
        
    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities

def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):

    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split())==1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        # print (entities)
        new_facts = []
        for i, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any([fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for j, other_fact in enumerate(facts) if j != i]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities==covered_entities
        except Exception:
            new_facts = facts # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences


def main():
    generator = AtomicFactGenerator("llm-guided_insight_search\\FActScore\\factscore\\demos")

    sample_text = """Today's market analysis reveals an overall bearish sentiment across major indices, characterized by lower-than-average trading volumes and subdued volatility. The S&P 500 closed at 3839.5, which is slightly above its 10-day Simple Moving Average (SMA) of 3833.85 but below the 50-day SMA of 3898.26. This indicates mixed short-term momentum but weaker medium-term momentum. The trading volume for the S&P 500 was 2,979,870,000, lower than the 30-day average of 4,032,987,000, suggesting reduced investor activity. The day's price increase of 10.44 points translates to a modest 0.27% gain, yet the overall market trend remains negative, with the aggregate market direction showing a significant decline of -2631.72 USD.

The NASDAQ Composite displayed similar bearish characteristics, with a closing price of 10466.48, which is below both the 10-day and 50-day SMAs, indicating a downward momentum. Despite a 0.95% daily gain, trading volume was lower than average, further reinforcing the market's bearish trend. The NASDAQ 100, heavily weighted towards technology stocks, also experienced reduced volatility and below-average trading volumes, closing at 10939.76. While it saw a minor daily increase of 0.96%, the broader market sentiment remains pessimistic.

The Dow Jones Industrial Average (DJIA) showed low volatility and slightly decreased trading volume. The current closing price of 33147.25 is slightly above its 10-day SMA but below the 50-day SMA, highlighting weakening short-term momentum. The DJIA's modest 0.08% daily gain contrasts with the overall bearish market trend.

Sector-specific analysis reveals that the S&P 500 Information Technology sector also experienced a bearish trend, with lower-than-average trading volumes and a closing price of 2172.17, below its short- and medium-term moving averages. The S&P 500 Health Care sector displayed moderate volatility with recent downward momentum, closing lower than its opening price by 5.16 points. Similarly, the S&P 500 Consumer Staples and S&P 500 Utilities sectors exhibited consistent bearish trends, supported by negative price changes and low trading volumes.

Volatility measures, such as the CBOE Volatility Index (VIX), indicate mixed signals. The VIX showed lower-than-usual volatility, suggesting reduced market uncertainty, yet the overall momentum remains weak. The VIX�s lower levels often correlate with a stronger Dollar Index, although today's data suggests a slight decline in the Dollar Index, which closed at 103.52, reflecting lower-than-usual volatility and negative momentum.

Economic indicators like bond yields showed positive momentum, with the U.S. 10-Year Bond Yield closing at 3.88, above both its 10-day and 50-day SMAs, indicating a bullish trend. The U.S. 2-Year Bond Yield also demonstrated consistent volatility with positive momentum, suggesting short-term bullish sentiment.

Gold prices exhibited relatively low volatility, with weak short-term but strong medium-term momentum. The price change of -0.76 indicates minor negative performance for the day. The inverse relationship between Gold and the Dollar Index remains evident, as fluctuations in the Dollar Index often influence gold prices.

In summary, the overall market trend today is predominantly bearish, with major indices displaying weak momentum, lower-than-average trading volumes, and subdued volatility. Sector-specific analysis corroborates the bearish sentiment, with most sectors experiencing negative price changes and reduced investor activity. Economic indicators like bond yields show positive momentum, yet the broader market trend remains downward. Investors should exercise caution given the current market conditions and consider diversifying their portfolios to manage risk effectively."""
    atomic_facts, para_breaks = generator.run(sample_text)

    print(atomic_facts)
    print(para_breaks)

if __name__ == "__main__":
    main()