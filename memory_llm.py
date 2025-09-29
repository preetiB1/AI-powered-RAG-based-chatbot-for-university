import os
import requests
from bs4 import BeautifulSoup
import PyPDF2

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdf_folder = "data/"  # Path to your PDFs folder
urls_file = "urls.txt"  # Text file with one URL per line

def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents=load_pdf_files(data=pdf_folder)
#print("Length of PDF pages: ",len(documents))



def load_webpages_from_urls_file(urls_file):
    documents = []
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            page_text = "\n".join(p.get_text() for p in paragraphs)
            metadata = {"source": url}
            doc = Document(page_content=page_text, metadata=metadata)
            documents.append(doc)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    
    return documents

# Usage
web_documents = load_webpages_from_urls_file(urls_file)
#print("Number of webpages scraped:", len(web_documents))


# Combine your PDF and webpage documents
all_documents = documents + web_documents

# Initialize RecursiveCharacterTextSplitter
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # max characters per chunk
    chunk_overlap=200,      # overlap between chunks to maintain context
    )
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=all_documents)
#print("Length of Text chunks: ",len(text_chunks))

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model=get_embedding_model()

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)