import os
import json
import time
import sqlite3
import numpy as np
import pickle as pkl
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm.auto import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Constants
MAX_LENGTH = 512
SPECIAL_SEPARATOR = "<SPECIAL_SEPARATOR>"


class DocDB:
    """SQLite backed document storage for Wikipedia articles with retrieval capabilities."""

    def __init__(self, db_path=None, wiki_content_dir=None, force_rebuild=False,
                 embedding_model_name_or_path=None, cache=None, embed_cache=None, retrieval_method="bm25"):
        """
        Initialize the document database.
        
        Parameters:
        - db_path: Path to the SQLite database file
        - wiki_content_dir: Directory containing Wikipedia content
        - force_rebuild: Whether to rebuild the database even if it exists
        - embedding_model: Model name to use for embedding-based search
        """
        self.db_path = db_path
        self.embedding_model = embedding_model_name_or_path
        self._encoder = None
        self.cache = cache if cache else {}
        self.embed_cache = embed_cache if embed_cache else {}

        # Check if we need to build the database
        db_exists = os.path.exists(db_path) if db_path else False
        
        if not db_exists or force_rebuild:
            assert wiki_content_dir is not None, f"wiki_content_dir must be provided to build the database."
            print(f"Building database at {db_path} from {wiki_content_dir}...")
            self.build_db(db_path, wiki_content_dir)
        
        # Connect to the database
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        print(f"Connected to database at {db_path}")
        if retrieval_method == "embedding":
            self.retrieval_func = self._embedding_retrieve
            self._ensure_encoder_loaded()
        else:
            self.retrieval_func = self._bm25_retrieve

        self.all_titles = self.get_all_titles()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, wiki_content_dir):
        """
        Build the database from the Wikipedia content.
        
        Parameters:
        - db_path: Path to create the SQLite database
        - wiki_content_dir: Directory containing the extracted Wikipedia content
        """
        # Read the index file
        index_path = os.path.join(wiki_content_dir, 'finance_pages_index.csv')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")
        
        index_df = pd.read_csv(index_path)
        print(f"Loaded index with {len(index_df)} Wikipedia pages")
        
        # Filter to only successfully downloaded pages
        success_pages = index_df[index_df['status'] == 'success']
        print(f"Found {len(success_pages)} successfully downloaded pages")
        
        # Create the database and table
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS documents (title TEXT PRIMARY KEY, text TEXT);")
        
        # Process each page and add to the database
        total_processed = 0
        batch_size = 1000
        current_batch = []
        
        for _, row in tqdm(success_pages.iterrows(), total=len(success_pages), desc="Processing pages"):
            try:
                if pd.isna(row['filepath']) or not os.path.exists(row['filepath']):
                    continue
                
                with open(row['filepath'], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip empty content
                if not content.strip():
                    continue
                
                # Split content into passages and join with a separator
                paragraphs = [p for p in content.split('\n\n') if p.strip()]
                text = SPECIAL_SEPARATOR.join(paragraphs)
                
                current_batch.append((row['title'], text))
                total_processed += 1
                
                # Process in batches
                if len(current_batch) >= batch_size:
                    cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", current_batch)
                    connection.commit()
                    current_batch = []
            
            except Exception as e:
                print(f"Error processing {row['title']}: {str(e)}")
        
        # Process any remaining pages
        if current_batch:
            cursor.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", current_batch)
            connection.commit()
        
        print(f"Database built with {total_processed} pages")
        connection.close()

    def get_text_from_title(self, title: str) -> List[Dict[str, str]]:
        """
        Fetch text content for a document by title.
        
        Parameters:
        - title: The document title to retrieve
        
        Returns:
        - List of passages with title and text keys
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        cursor.close()
        
        if not results or len(results) != 1:
            raise ValueError(f"Title '{title}' not found in the database.")
            
        # Split the content by the separator and create passage objects
        passages = results[0][0].split(SPECIAL_SEPARATOR)
        results = [{"title": title, "text": para} for para in passages]
        
        if not results:
            raise ValueError(f"No content found for title '{title}'.")
            
        return results

    def get_all_titles(self) -> List[str]:
        """
        Fetch all document titles from the database.
        
        Returns:
        - List of all titles in the database
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT title FROM documents")
        results = cursor.fetchall()
        cursor.close()
        return [r[0] for r in results]
    
    def _ensure_encoder_loaded(self):
        """Load the encoder if not already loaded."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(self.embedding_model)
            encoder = encoder.eval()
            self._encoder = encoder
    
    def retrieve(self, query: str, titles=None, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve passages using specified method (embedding or BM25).
        
        Parameters:
        - query: The search query
        - titles: Optional list of titles to restrict search to
        - k: Number of results to return
        - return_raw: Whether to return raw passage objects
        
        Returns:
        - List of retrieved passages with scores
        """
        
        if query in self.cache:
            return self.cache[query]

        # Get passages from the specified titles or pageids
        passages = []
        
        if titles:
            for title in titles:
                try:
                    passages.extend(self.get_text_from_title(title))
                except ValueError:
                    continue
        else:
            title_query = f"TITLE#{query}"
            if title_query in self.cache:
                similar_titles = self.cache[title_query]
            else:
                similar_titles = self.retrieval_func(query, passages=None, k=10, retrieve_mode = 'title')
            for title in similar_titles:
                try:
                    passages.extend(self.get_text_from_title(title))
                except ValueError:
                    continue

        if not passages:
            return []
        
        # Perform retrieval using the specified method
        results = self. retrieval_func(query, passages, k)
        self.cache[query] = results

        return results


    def _embedding_retrieve(self, query: str, passages: List[Dict[str, Any]], k: int, batch_size: int=8, retrieve_mode='passage') -> List[Dict[str, Any]]:
        """Retrieve passages using embedding similarity."""
        # self._ensure_encoder_loaded()
        if retrieve_mode == 'title':
            if "ALLTITLE" in self.embed_cache:
                content_vectors = self.embed_cache["ALLTITLE"]
            else:
                content_vectors = self._encoder.encode(self.all_titles, batch_size=batch_size, 
                                    device=self._encoder.device if hasattr(self._encoder, 'device') else None)
            self.cache[f"ALLTITLE"] = content_vectors
        else:
            topic = passages[0]["title"] if passages else ""

            if topic in self.embed_cache:
                content_vectors = self.embed_cache[f"EMBED#{topic}"]
            else:
                inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
                content_vectors = self._encoder.encode(inputs, batch_size=batch_size, 
                                                       device=self._encoder.device if hasattr(self._encoder, 'device') else None)
                self.embed_cache[f"EMBED#{topic}"] = content_vectors

        query_vector = self._encoder.encode(query, batch_size=1)
        
        # Calculate scores (cosine similarity)
        scores = np.inner(query_vector, content_vectors)
        indices = np.argsort(-scores)[:k]
        
        if retrieve_mode == 'title':
            result = [self.all_titles[i] for i in indices]
        else:
            result = [passages[i].copy() for i in indices]
            # Add scores to the results
            for i, idx in enumerate(indices):
                result[i]["score"] = float(scores[idx])
        
        return result
    
    def _bm25_retrieve(self, query: str, passages: List[Dict[str, Any]], k: int, retrieve_mode = "passage") -> List[Dict[str, Any]]:
        """Retrieve passages using BM25 ranking."""
        from rank_bm25 import BM25Okapi
        
        topic = passages[0]["title"] if passages else ""
            
        if topic in self.cache:
            bm25 = self.cache[f"BM25#{topic}"]
        else:
            # Create corpus from passages
            if retrieve_mode == 'title':
                corpus = [title.split() for title in self.all_titles]
            else:
                corpus = [psg["text"].split() for psg in passages]
            bm25 = BM25Okapi(corpus)
            self.cache[f"BM25#{topic}"] = bm25
        
        # Get scores
        query_tokens = query.split()
        scores = bm25.get_scores(query_tokens)
        indices = np.argsort(-scores)[:k]
        
        if retrieve_mode == 'title':
            return [self.all_titles[i] for i in indices]
        result = [passages[i].copy() for i in indices]
        # Add scores to the results
        for i, idx in enumerate(indices):
            result[i]["score"] = float(scores[idx])
        
        return result


class VectorDB:
    """Vector database for Wikipedia content semantic search."""
    
    def __init__(self, db_dir: str, wiki_content_dir: str = None, 
                 embedding_model: str = None,
                 chunk_size: int = 512, chunk_overlap: int = 50,
                 force_rebuild: bool = False, cache: Dict = None, embed_cache: Dict = None):
        """
        Initialize the vector database.
        
        Parameters:
        - db_dir: Directory to store/load the vector database
        - wiki_content_dir: Directory containing Wikipedia content
        - embedding_model: Model to use for embeddings
        - chunk_size: Size of text chunks for processing
        - chunk_overlap: Overlap between text chunks
        - force_rebuild: Whether to rebuild the database even if it exists
        """
        self.db_dir = db_dir
        self.wiki_content_dir = wiki_content_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache = cache if cache else {}
        self.embed_cache = embed_cache if embed_cache else {}
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        print(f"Initialized embeddings with model: {embedding_model}")
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize dictionaries for title embeddings
        self.title_embeddings = {}
        self.title_to_id = {}

        # Check if we need to build the database
        db_exists = os.path.exists(db_dir) and os.path.isdir(db_dir) and len(os.listdir(db_dir)) > 0
        
        if not db_exists or force_rebuild:
            if wiki_content_dir is None:
                raise ValueError("wiki_content_dir must be provided to build the vector database")
            
            print(f"Building vector database at {db_dir} from {wiki_content_dir}...")
            self.build_db(wiki_content_dir)
        else:
            # Load existing database
            self.db = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
            print(f"Loaded existing vector database from {db_dir}")
            
            # Try to load title embeddings from disk
            if not self._load_title_embeddings():
                # If not found, initialize them from the database
                print("Title embeddings not found, initializing from database...")
                self._initialize_title_embeddings()

    def _get_title_embeddings_path(self):
        """Get the path for title embeddings file."""
        return os.path.join(self.db_dir, "title_embeddings.pkl")

    def _save_title_embeddings(self):
        """Save title embeddings to disk."""
        embeddings_path = self._get_title_embeddings_path()
        
        data_to_save = {
            "title_embeddings": self.title_embeddings,
            "title_to_id": self.title_to_id,
            "embedding_model": self.embeddings.model_name
        }
        
        try:
            with open(embeddings_path, 'wb') as f:
                pkl.dump(data_to_save, f)
            print(f"Saved {len(self.title_embeddings)} title embeddings to {embeddings_path}")
            return True
        except Exception as e:
            print(f"Error saving title embeddings: {str(e)}")
            return False

    def _load_title_embeddings(self):
        """Load title embeddings from disk."""
        embeddings_path = self._get_title_embeddings_path()
        
        if not os.path.exists(embeddings_path):
            return False
        
        try:
            with open(embeddings_path, 'rb') as f:
                data = pkl.load(f)
            
            # Check if the embeddings were generated with the same model
            if data.get("embedding_model") != self.embeddings.model_name:
                print(f"Warning: Embeddings were generated with a different model ({data.get('embedding_model')})")
                return False
            
            self.title_embeddings = data.get("title_embeddings", {})
            self.title_to_id = data.get("title_to_id", {})
            
            print(f"Loaded {len(self.title_embeddings)} title embeddings from {embeddings_path}")
            return len(self.title_embeddings) > 0
        except Exception as e:
            print(f"Error loading title embeddings: {str(e)}")
            return False

    def _initialize_title_embeddings(self):
        """Initialize title embeddings from the existing database."""
        if not hasattr(self, 'db') or self.db is None:
            raise ValueError("Vector database not initialized")
        
        collection = self.db._collection
        all_metadatas = collection.get()["metadatas"]
        
        # Extract unique titles and their IDs
        unique_titles = []
        seen_titles = set()
        
        for i, meta in enumerate(all_metadatas):
            title = meta.get("title")
            if title and title not in seen_titles:
                unique_titles.append(title)
                seen_titles.add(title)
        
        # Generate embeddings for all titles in batch
        if unique_titles:
            title_embedding_list = self.embeddings.embed_documents(unique_titles)
            for i, title in enumerate(unique_titles):
                self.title_embeddings[title] = title_embedding_list[i]
        
        print(f"Initialized embeddings for {len(self.title_embeddings)} unique titles")
        
        # Save the title embeddings to disk
        self._save_title_embeddings()

    def build_db(self, wiki_content_dir: str):
        """
        Build the vector database from Wikipedia content.
        
        Parameters:
        - wiki_content_dir: Directory containing the extracted Wikipedia content
        """
        # Read the index file
        index_path = os.path.join(wiki_content_dir, 'finance_pages_index.csv')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")
        
        index_df = pd.read_csv(index_path)
        print(f"Loaded index with {len(index_df)} Wikipedia pages")
        
        # Filter to only successfully downloaded pages
        success_pages = index_df[index_df['status'] == 'success']
        print(f"Found {len(success_pages)} successfully downloaded pages")
        
        # Process documents and create chunks
        all_chunks = []
        all_metadatas = []
            
        # Collect unique titles for title embeddings
        unique_titles = []
        
        for _, row in tqdm(success_pages.iterrows(), total=len(success_pages), desc="Processing documents"):
            try:
                if pd.isna(row['filepath']) or not os.path.exists(row['filepath']):
                    continue
                    
                with open(row['filepath'], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip empty content
                if not content.strip():
                    continue
                
                # Split the content into chunks
                chunks = self.text_splitter.create_documents(
                    texts=[content],
                    metadatas=[{
                        'title': row['title'],
                        'source': row['filepath']
                    }]
                )
                
                # Extract texts and metadatas
                texts = [chunk.page_content for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]
                
                all_chunks.extend(texts)
                all_metadatas.extend(metadatas)
                
            except Exception as e:
                print(f"Error processing {row['title']}: {str(e)}")
        
        print(f"Created {len(all_chunks)} chunks from {len(success_pages)} documents")
        
        # Create directories if they don't exist
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Create and persist the database
        self.db = Chroma.from_texts(
            texts=all_chunks,
            metadatas=all_metadatas,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        # Generate title embeddings
        if unique_titles:
            title_embedding_list = self.embeddings.embed_documents(unique_titles)
            for i, title in enumerate(unique_titles):
                self.title_embeddings[title] = title_embedding_list[i]
            
            print(f"Generated embeddings for {len(unique_titles)} unique titles")
            
            # Save title embeddings to disk
            self._save_title_embeddings()

        print(f"Vector database built with {len(all_chunks)} chunks at {self.db_dir}")
        

    def retrieve(self, query: str, use_title: bool = True, k: int = 5, k_title: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve passages using specified method (embedding or BM25).
        
        Parameters:
        - query: The search query
        - titles: Optional list of titles to restrict search to
        - k: Number of results to return
        - return_raw: Whether to return raw passage objects
        
        Returns:
        - List of retrieved passages with scores
        """
        
        # Get passages from the specified titles or pageids
        if query in self.cache:
            return self.cache[query]
        
        query_embedding = self.embeddings.embed_query(query)        

        if use_title:
            title_query = f"TITLE#{query}"
            if title_query in self.cache:
                similar_titles = self.cache[title_query]
            else:
                similar_titles = self.title_similarity_search(query_embedding, k_title)
                self.cache[query] = similar_titles

            if not similar_titles:
                return []
        else:
            similar_titles = None
            
        # Perform retrieval using the specified method
        results = self.content_similarity_search(query_embedding, filter_titles=similar_titles, k=k)
        self.cache[query] = results

        return results

    def title_similarity_search(self, query_embedding, k: int = 4):
        """
        Perform similarity search on the vector database.
        
        Parameters:
        - query: Query text
        - k: Number of results to return
        - search_type: Type of search ('content' or 'title')
        
        Returns:
        - List of document objects with text and metadata for content search
        - List of titles for title search
        """
        if not hasattr(self, 'db') or self.db is None:
            raise ValueError("Vector database not initialized")

        # Search using title embeddings
        if not self.title_embeddings:
            raise ValueError("No title embeddings available")
        
        # # Generate query embedding
        # query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarities
        similarities = []
        for title, embedding in self.title_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((title, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k similar titles
        top_titles = [title for title, _ in similarities[:k]]
        
        # # Fetch documents for these titles
        # results = []
        # # for title in top_titles:
        # #     # Get page ID from title
        # #     page_id = self.title_to_id.get(title)
        # #     if not page_id:
        # #         continue
                
        # #     # Find documents with this title
        # #     title_docs = self.db.similarity_search(
        # #         f"title:{title}",  # Using title as a query
        # #         k=1,              # Get just one result per title
        # #         filter={"pageid": page_id}  # Filter by page ID
        # #     )
            
        # #     # Add to results
        # #     results.extend(title_docs)
            
        # #     # Break if we have enough results
        # #     if len(results) >= k:
        # #         break
        
        # # # If we didn't get enough results, try a broader search
        # # if len(results) < k:
        #     # Get all documents for these titles without filtering
        # collection = self.db._collection
        # ids = [id for id in collection.get()["ids"] 
        #     if collection.get(ids=[id])["metadatas"][0].get("title") in top_titles]
        
        # # Get documents for these IDs
        # if ids:
        #     docs = self.db.get_documents(ids=ids[:k-len(results)])
        #     results.extend(docs)
        
        return top_titles

    def content_similarity_search(self, query_embedding, filter_titles=None, k: int = 4):
        """
        Perform similarity search on a filtered subset of the vector database.
        
        Parameters:
        - query: Query text
        - filter_titles: List of titles to restrict the search to
        - k: Number of results to return
        - return_raw: If True, returns a list of dictionaries with title, text, and score
        
        Returns:
        - If return_raw=False: List of document objects with text and metadata
        - If return_raw=True: List of dicts with {'title': title, 'text': text, 'score': score}
        """
        if not hasattr(self, 'db') or self.db is None:
            raise ValueError("Vector database not initialized")
            
        # Build the filter condition
        filter_condition = {}
        
        if filter_titles:
            # Filter by title
            filter_condition = {"title": {"$in": filter_titles}}
        # results = self.db.similarity_search_with_score(
        #     query=query,
        #     k=k,
        #     filter=filter_condition
        # )
        
            # Measure execution time
        
        try:
            # If you have direct access to Chroma's underlying implementation
            # you could use query_embedding directly to avoid recomputation
            # query_embedding = self.embeddings.embed_query(query)
            
            # if filter_titles:
            #     where_clause = {"title": {"$in": filter_titles}}
            results = self.db.similarity_search_by_vector(
                embedding=query_embedding,
                k=k,
                filter=filter_condition
            )
            # else:
            #     results = self.db.similarity_search_with_score(query=query, k=k)
                
        except RuntimeError as e:
            if "Cannot return the results in a contigious 2D array" in str(e):
                # Fall back options with better error handling
                try:
                    # Option 1: Reduce k
                    results = self.db.similarity_search_by_vector(embedding=query_embedding, k=max(1, k//2), filter=filter_condition)
                except:
                    # Option 2: Try without filter
                    results = self.db.similarity_search_by_vector(embedding=query_embedding, k=k)
            else:
                raise  # Re-raise if it's a different error
        

        return [
            {
                'title': doc.metadata.get('title', ''),
                'text': doc.page_content,
                # 'score': score
            } for doc in results
        ]

    def is_initialized(self) -> bool:
        """Check if the vector database is initialized."""
        return hasattr(self, 'db') and self.db is not None
