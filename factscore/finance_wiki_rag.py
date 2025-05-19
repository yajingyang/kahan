import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, ChatOpenAI
import openai
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

class FinanceWikiRAG:
    def __init__(self, content_dir='finance_wiki_content', 
                 db_dir='finance_wiki_vectordb',
                 embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 chunk_size=512,
                 chunk_overlap=50):
        """
        Initialize the Finance Wikipedia RAG system
        
        Parameters:
        - content_dir: Directory containing the extracted Wikipedia content
        - db_dir: Directory to store the vector database
        - embedding_model: Model to use for embeddings
        - chunk_size: Size of text chunks for processing
        - chunk_overlap: Overlap between text chunks
        """
        self.content_dir = content_dir
        self.db_dir = db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load the index file
        self.index_path = os.path.join(content_dir, 'finance_pages_index.csv')
        if os.path.exists(self.index_path):
            self.index_df = pd.read_csv(self.index_path)
            print(f"Loaded index with {len(self.index_df)} Wikipedia finance pages")
        else:
            raise FileNotFoundError(f"Index file not found at {self.index_path}. Run the extractor first.")
            
        # Setup the embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        print(f"Initialized embeddings with model: {embedding_model}")
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
    def process_documents(self):
        """
        Process all documents, split them into chunks, and prepare for indexing
        
        Returns:
        - Dictionary of documents with their chunks and metadata
        """
        all_chunks = []
        all_metadatas = []
        
        # Filter to only successfully downloaded pages
        success_pages = self.index_df[self.index_df['status'] == 'success']
        
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
                        'pageid': str(row['pageid']),
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
        
        return {
            'texts': all_chunks,
            'metadatas': all_metadatas
        }
    
    def create_vector_db(self, force_recreate=False):
        """
        Create a vector database from the processed documents
        
        Parameters:
        - force_recreate: Whether to recreate the database if it already exists
        
        Returns:
        - The vector database
        """
        # Check if the database already exists
        if os.path.exists(self.db_dir) and not force_recreate:
            print(f"Loading existing vector database from {self.db_dir}")
            return Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
        
        # Create the directory if it doesn't exist
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Process documents
        processed_docs = self.process_documents()
        
        # Create and persist the database
        db = Chroma.from_texts(
            texts=processed_docs['texts'],
            metadatas=processed_docs['metadatas'],
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        
        print(f"Created vector database with {len(processed_docs['texts'])} chunks at {self.db_dir}")
        return db
    
    def setup_retriever(self, top_k=4):
        """
        Set up the retriever for RAG
        
        Parameters:
        - top_k: Number of documents to retrieve
        
        Returns:
        - The retriever
        """
        db = self.create_vector_db()
        retriever = db.as_retriever(search_kwargs={"k": top_k})
        return retriever
    
    def setup_rag_chain(self, model_name="gpt-3.5-turbo", temperature=0, top_k=4):
        """
        Set up the RAG chain
        
        Parameters:
        - model_name: Name of the LLM to use
        - temperature: Temperature for the LLM
        - top_k: Number of documents to retrieve
        
        Returns:
        - The RAG chain
        """
        retriever = self.setup_retriever(top_k=top_k)
        
        # Initialize the LLM
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Create the RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        
        return rag_chain
    
    def query(self, question, model_name="gpt-3.5-turbo", temperature=0, top_k=4):
        """
        Query the RAG system
        
        Parameters:
        - question: Question to ask
        - model_name: Name of the LLM to use
        - temperature: Temperature for the LLM
        - top_k: Number of documents to retrieve
        
        Returns:
        - Answer and source documents
        """
        rag_chain = self.setup_rag_chain(model_name=model_name, temperature=temperature, top_k=top_k)
        result = rag_chain({"query": question})
        
        return {
            'question': question,
            'answer': result['result'],
            'source_documents': result['source_documents']
        }
    
    def direct_retrieval(self, query, top_k=4):
        """
        Direct retrieval without using the LLM
        
        Parameters:
        - query: Query to search for
        - top_k: Number of documents to retrieve
        
        Returns:
        - Retrieved documents
        """
        db = self.create_vector_db()
        docs = db.similarity_search(query, k=top_k)
        
        print(f"Top {len(docs)} documents for query: '{query}'")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Source: {doc.metadata['title']} (ID: {doc.metadata['pageid']})")
            print(f"Content: {textwrap.fill(doc.page_content[:300])}...")
        
        return docs
    
    def evaluate_rag(self, questions):
        """
        Evaluate the RAG system on a set of questions
        
        Parameters:
        - questions: List of questions to evaluate
        
        Returns:
        - Evaluation results
        """
        results = []
        
        for question in tqdm(questions, desc="Evaluating questions"):
            result = self.query(question)
            results.append(result)
            
            print(f"\nQuestion: {question}")
            print(f"Answer: {result['answer']}")
            print("Sources:")
            for i, doc in enumerate(result['source_documents']):
                print(f"  {i+1}. {doc.metadata['title']}")
            
        return results

def main():
    questions = [
        "What is the difference between stocks and bonds?",
        "Explain how mutual funds work",
        "What are the main financial ratios used to evaluate companies?",
        "How does inflation affect the economy?",
        "What strategies can be used for retirement planning?"
    ]
    
    # Initialize the RAG system
    finance_rag = FinanceWikiRAG()
    
    # Create the vector database (only needs to be done once)
    finance_rag.create_vector_db()
    
    # Simple retrieval
    finance_rag.direct_retrieval("financial derivatives")
    
    # Query the RAG system
    for question in questions:
        result = finance_rag.query(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print("Sources:")
        for i, doc in enumerate(result['source_documents']):
            print(f"  {i+1}. {doc.metadata['title']}")

if __name__ == "__main__":
    main()