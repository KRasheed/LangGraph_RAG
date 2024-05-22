from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from typing import List
from dotenv import load_dotenv
import os
import shutil
import base64
import traceback
# import fitz
# from docx import Document

load_dotenv()
openai_api_key = os.environ["Openai_API_key"]


def read_text_files(directory_path):
    texts = {}
    # List all files in the given directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Check if the file is a text file
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                texts[filename] = file.read()
    return texts


def get_doc_splits(file_path: str) -> List[Document]:
    """Reads text files and returns a list of document splits."""
    text = read_text_files(file_path)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=50
    )

    doc_splits = []
    for filename, content in text.items():
        tokenized_texts = text_splitter.split_text(
            content
        )  # Tokenize the content of each file
        for chunk in tokenized_texts:
            doc_splits.append(
                Document(page_content=chunk, metadata={"source_document": filename})
            )

    return doc_splits


def setup_retriever(documents, question):
    """Sets up and returns a vector-based retriever from a list of documents."""
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="chroma",
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )
    document = retriever.get_relevant_documents(question)
    return document


def read_docx(file_path):
    try:
        doc = Document(file_path)
        content = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(content)
    except Exception as e:
        return f"Error reading DOCX file: {e}"


def read_pdf(file_path):
    try:
        text = ""
        # Open the PDF file
        with fitz.open(file_path) as pdf:
            total_pages = pdf.page_count
            for page_number in range(total_pages):
                page = pdf.load_page(page_number)
                # Extract text from the page
                page_text = page.get_text("text")
                text += page_text
                text += "\n\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF file: {e}"


def read_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"Error reading TXT file: {e}"
