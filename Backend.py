import glob
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer

CHROMA_PATH = str(Path(__file__).parent/ "chroma")
DATA_PATH = str(Path(__file__).parent/ "dataset")

class LLM:
    def __init__(self) -> None:
        model_id = 'sentence-transformers/multi-qa-mpnet-base-dot-v1' #'sentence-transformers/multi-qa-distilbert-cos-v1'
        model_kwargs = {'device': 'cpu'}
        self.hf_embedding = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs=model_kwargs
        )
        self.vector_store = None
        self.checkDataBase()

    def getSplitData(self) -> list:
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()

        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            strip_whitespace=True,
        )

        r_splitter_result = r_splitter.split_documents(documents)
        return r_splitter_result
    
    def checkDataBase(self) -> None:
        if not glob.glob(os.path.join(CHROMA_PATH, "*.sqlite3")):
            r_splitter_result = self.getSplitData()
            self.vector_store = Chroma.from_documents(
                r_splitter_result, self.hf_embedding, persist_directory=CHROMA_PATH
            )
        else:
            self.vector_store = Chroma(
                persist_directory=CHROMA_PATH, embedding_function=self.hf_embedding
            )
    
    def getSimilarity(self, question : str) -> str:
        docs = self.vector_store.similarity_search(question,k=1)
        return docs[0].page_content
