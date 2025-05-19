import os
import json
import time
import sqlite3
import numpy as np
import pickle as pkl
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Union, Optional, Any, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from factscore.document_store import VectorDB, DocDB

MAX_LENGTH = 512
SPECIAL_SEPARATOR = "<SPECIAL_SEPARATOR>"


class RetrievalMethod(ABC):
    """Abstract base class for retrieval methods."""
    
    @abstractmethod
    def retrieve(self, query: str, passages: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Retrieve the most relevant passages for a query."""
        pass


class BM25Retrieval(RetrievalMethod):
    """BM25 based retrieval method."""
    
    def __init__(self, cache: Dict = None):
        """Initialize with optional cache."""
        self.cache = cache or {}
    
    def retrieve(self, query: str, passages: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Retrieve passages using BM25 ranking."""
        topic = passages[0]["title"] if passages else ""
        
        if topic in self.cache:
            bm25 = self.cache[topic]
        else:
            bm25 = BM25Okapi([psg["text"].replace("<s>", "").replace("</s>", "").split() for psg in passages])
            self.cache[topic] = bm25
        
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        
        result = [passages[i] for i in indices]
        # Add scores to the results
        for i, idx in enumerate(indices):
            result[i]["score"] = scores[idx]
        
        return result


class EmbeddingRetrieval(RetrievalMethod):
    """Embedding-based retrieval using sentence transformers."""
    
    def __init__(self, model_name: str, batch_size: int, cache: Dict = None, device: str = None):
        """Initialize with model parameters and optional cache."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache = cache or {}
        self.encoder = None
        self.device = device
    
    def _ensure_encoder_loaded(self):
        """Load the encoder if not already loaded."""
        if self.encoder is None:
            encoder = SentenceTransformer("sentence-transformers/" + self.model_name)
            if self.device:
                encoder = encoder.to(self.device)
            encoder = encoder.eval()
            self.encoder = encoder
    
    def retrieve(self, query: str, passages: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Retrieve passages using embedding similarity."""
        self._ensure_encoder_loaded()
        
        topic = passages[0]["title"] if passages else ""
        
        if topic in self.cache:
            passage_vectors = self.cache[topic]
        else:
            inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
            passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, 
                                                 device=self.encoder.device if hasattr(self.encoder, 'device') else None)
            self.cache[topic] = passage_vectors
        
        query_vectors = self.encoder.encode([query], batch_size=self.batch_size, 
                                           device=self.encoder.device if hasattr(self.encoder, 'device') else None)[0]
        
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]
        
        result = [passages[i] for i in indices]
        # Add scores to the results
        for i, idx in enumerate(indices):
            result[i]["score"] = scores[idx]
        
        return result


class VectorDBRetrieval(RetrievalMethod):
    """Retrieval using a vector database."""
    
    def __init__(self, vector_db: VectorDB):
        """Initialize with a vector database."""
        self.vector_db = vector_db
    
    def retrieve(self, query: str, passages: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Retrieve passages using the vector database."""
        # For VectorDB, we ignore the provided passages and use the DB directly
        docs = self.vector_db.similarity_search(query, k=k)
        
        # Convert to the standard format
        result = []
        for i, doc in enumerate(docs):
            result.append({
                "title": doc.metadata.get("title", "Unknown"),
                "text": doc.page_content,
                "score": 1.0 - (i / k)  # Approximate score based on rank
            })
        
        return result


class Retrieval:
    """
    Unified retrieval system for Wikipedia content with:
    1. Database access (VectorDB and DocDB)
    2. Different retrieval methods
    3. Multi-step retrieval process
    """

    def __init__(self, 
                 wiki_content_dir: str,
                 doc_db_path: str = None,
                 vector_db_dir: str = None,
                 embedding_model_name_or_path: str = None,
                 batch_size: int = 8,
                 cache_path: Optional[str] = None,
                 embed_cache_path: Optional[str] = None,
                 device: Optional[str] = "cuda",
                 force_rebuild: bool = False):
        """
        Initialize the Wikipedia retrieval system.
        
        Parameters:
        - wiki_content_dir: Directory containing the extracted Wikipedia content
        - doc_db_path: Path to the SQLite database file
        - vector_db_dir: Directory for the vector database
        - embedding_model_name_or_path: Name or path of the embedding model (for embedding method)
        - batch_size: Batch size for embedding generation
        - cache_path: Path to cache file for query results
        - embed_cache_path: Path to cache file for embeddings
        - device: Device to use for computations ("cuda" or "cpu")
        - force_rebuild: Whether to rebuild databases even if they exist
        """
        self.wiki_content_dir = wiki_content_dir

        # Step 1: Load caches if provided
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.cache = {}
        self.embed_cache = {}
        self.add_n = 0
        self.add_n_embed = 0

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self.cache = json.load(f)

        if embed_cache_path and os.path.exists(embed_cache_path):
            with open(embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        
        # Step 2: Initialize document stores
        if doc_db_path:
            self.doc_store = DocDB(
                db_path=doc_db_path, 
                wiki_content_dir=wiki_content_dir,
                embedding_model_name_or_path=embedding_model_name_or_path,
                force_rebuild=force_rebuild,
                cache=self.cache,
                embed_cache=self.embed_cache
            )
            print(f"Initialized DocDB at {doc_db_path}")
        else:
            self.doc_store = None
        
        if vector_db_dir:
            self.vector_db = VectorDB(
                db_dir=vector_db_dir,
                wiki_content_dir=wiki_content_dir,
                embedding_model=embedding_model_name_or_path,
                force_rebuild=force_rebuild,
                cache=self.cache,
                embed_cache=self.embed_cache
            )
            print(f"Initialized VectorDB at {vector_db_dir}")
        else:
            self.vector_db = None

        assert any([self.doc_store, self.vector_db]), "Provide either document storage path or vector db directory!"

        # Step 3: Initialize the retrieval method
        self.batch_size = batch_size
        self.device = device

    def save_cache(self):
        """Save caches to disk."""
        if self.add_n > 0 and self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
        
        if self.add_n_embed > 0 and self.embed_cache_path:
            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def get_passages(self, query_info: Union[str, Tuple[str, str]], k: int = 4, use_title=True) -> List[Dict[str, Any]]:
        """
        Main method to retrieve passages.
        
        Parameters:
        - query_info: Either a statement (str) or a tuple of (topic, question)
        - k: Number of passages to retrieve
        
        Returns:
        - List of relevant passages
        """
        # Step 1: Determine query type and create retrieval key
        if isinstance(query_info, tuple) and len(query_info) == 2:
            # Topic + question format (original approach)
            topic, question = query_info
            retrieval_query = f"{topic} {question}".strip()
            cache_key = f"{topic}#{retrieval_query}"
        else:
            # Single statement format (new approach)
            topic = None
            statement = query_info
            retrieval_query = statement
            cache_key = f"statement#{statement}"
        
        # Step 2: Check cache first
        # if cache_key in self.cache:
        #     result_timestamp = self.cache[cache_key]
        #     cur_cache_path = Path(self.cache_path).parent / 'retrieval_cache_files' / f"{result_timestamp}.json"
        #     with open(cur_cache_path, 'r') as f:
        #         results = json.load(f)
        #     return results
        
        # Step 3: Get passages based on approach
        if self.doc_store:
            if topic:
                results = self.doc_store.retrieve(retrieval_query, titles=[topic], k=k)
            else:
                results = self.doc_store.retrieve(retrieval_query,  titles=None , k=k)
        else:
            results = self.vector_db.retrieve(retrieval_query, k=k, use_title=use_title)

        # Cache and return results
        # result_timestamp = {int(time.time())}
        # self.cache[cache_key] = result_timestamp
        # retrieval_cache_dir = Path(self.cache_path).parent / 'retrieval_cache_files'
        # retrieval_cache_dir.mkdir(exist_ok=True)
        # cur_cache_path = retrieval_cache_dir / f"{result_timestamp}.json"
        # with open(cur_cache_path, 'w') as f:
        #     json.dump(results, f, indent=2)
        self.add_n += 1
        return results

    
    def get_passages_for_factuality(self, statement: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Specialized method for factuality evaluation, focusing on finding relevant passages.
        
        Parameters:
        - statement: Statement to evaluate factuality of
        - k: Number of passages to retrieve
        
        Returns:
        - List of relevant passages
        """
        print(f"\n=== Retrieving passages for factual evaluation: {statement} ===")
        
        # Step 1: Find relevant passages
        passages = self.get_passages(statement, k=k)
        
        # Step 2: Format the results for the factuality evaluator
        results = []
        for i, passage in enumerate(passages):
            print(f"\nPassage {i+1}:")
            print(f"Title: {passage['title']}")
            print(f"Text: {passage['text'][:200]}..." if len(passage['text']) > 200 else f"Text: {passage['text']}")
            
            results.append({
                "title": passage["title"],
                "text": passage["text"],
                "score": passage.get("score", 1.0 - (i / k))  # Use provided score or estimate from rank
            })
        
        print(f"\nRetrieved {len(results)} relevant passages for factuality evaluation")
        return results
    
    def hybrid_retrieval(self, statement: str, topic: Optional[str] = None, k: int = 4, 
                         alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining topic-based and statement-based approaches.
        
        Parameters:
        - statement: Statement to find passages for
        - topic: Optional topic to focus on (if None, will use vector DB to find topics)
        - k: Number of passages to retrieve
        - alpha: Weight for statement-based results (1-alpha for topic-based)
        
        Returns:
        - List of relevant passages
        """
        # Get statement-based passages
        statement_passages = self.get_passages(statement, k=k)
        
        # Get topic-based passages if a topic is provided or can be found
        if topic is None and self.vector_db and self.vector_db.is_initialized():
            # Find the most relevant topic
            docs = self.vector_db.similarity_search(statement, k=1)
            if docs and "title" in docs[0].metadata:
                topic = docs[0].metadata["title"]
        
        if topic:
            topic_passages = self.get_passages((topic, statement), k=k)
        else:
            # No topic found, just use statement passages
            return statement_passages
        
        # Combine and weight the results
        combined = []
        
        # Add statement-based passages with their weight
        for i, passage in enumerate(statement_passages):
            score = passage.get("score", 1.0 - (i / k))
            combined.append({
                "title": passage["title"],
                "text": passage["text"],
                "score": alpha * score
            })
        
        # Add topic-based passages with their weight
        for i, passage in enumerate(topic_passages):
            score = passage.get("score", 1.0 - (i / k))
            combined.append({
                "title": passage["title"],
                "text": passage["text"],
                "score": (1 - alpha) * score
            })
        
        # Remove duplicates (prefer higher scores)
        seen_texts = {}
        for passage in combined:
            text = passage["text"]
            if text in seen_texts:
                if passage["score"] > seen_texts[text]["score"]:
                    seen_texts[text] = passage
            else:
                seen_texts[text] = passage
        
        # Sort by score and take top k
        results = list(seen_texts.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:k]


def main():    
    wiki_content_dir = "finance_wiki_content"
    doc_db_path = "finance_wiki.sqlite"
    vector_db_dir = "finance_wiki_vectordb"
    
    retrieval = Retrieval(
        wiki_content_dir=wiki_content_dir,
        doc_db_path=doc_db_path,
        vector_db_dir=vector_db_dir,
        retrieval_method="vector",  # Can be "bm25", "embedding", or "vector"
        model_name=".cache\\factscore\model\\sentence-transformers\\all-MiniLM-L6-v2",
        cache_path="wiki_cache.json",
        embed_cache_path="wiki_embed_cache.pkl",
        force_rebuild=False  # Set to True to rebuild databases
    )
    
    # Example 1: Statement-based retrieval (for factuality evaluation)
    statement = "Hedge funds typically charge a 2% management fee and 20% performance fee."
    passages = retrieval.get_passages_for_factuality(statement, k=4)
    
    # Example 2: Topic + question retrieval
    topic_question = ("Hedge fund", "What fees do hedge funds typically charge?")
    topic_passages = retrieval.get_passages(topic_question, k=4)
    
    print("\n=== Topic + Question Results ===")
    for i, passage in enumerate(topic_passages):
        print(f"\nPassage {i+1}:")
        print(f"Title: {passage['title']}")
        print(f"Text: {passage['text'][:200]}..." if len(passage['text']) > 200 else f"Text: {passage['text']}")
    
    # Example 3: Hybrid retrieval
    hybrid_passages = retrieval.hybrid_retrieval(statement, topic="Hedge fund", k=4, alpha=0.7)
    
    print("\n=== Hybrid Retrieval Results ===")
    for i, passage in enumerate(hybrid_passages):
        print(f"\nPassage {i+1}:")
        print(f"Title: {passage['title']}")
        print(f"Text: {passage['text'][:200]}..." if len(passage['text']) > 200 else f"Text: {passage['text']}")
        print(f"Score: {passage.get('score', 'N/A')}")
    
    # Save caches
    retrieval.save_cache()

if __name__ == "__main__":
    main()