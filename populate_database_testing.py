import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores.chroma import Chroma
import re


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents_paragraphs(documents):
    # Funktion, die Text basierend auf Absätzen und dynamischer Chunk-Größe splittet.
    def split_text_by_paragraph(text, max_chunk_size=400, overlap=60):
        paragraphs = re.split(r"\n \n+", text)  # Split nach Absätzen
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()  # Entferne überflüssige Leerzeichen

            # Wenn der aktuelle Chunk mit dem neuen Absatz die Grenze überschreiten würde:
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:  # +2 für "\n\n"
                # Chunk abschließen und hinzufügen
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
            else:
                # Absatz zum aktuellen Chunk hinzufügen
                current_chunk += paragraph + "\n\n"

        # Letzten Chunk hinzufügen, falls nicht leer
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Optionale Overlap-Logik
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i > 0:
                    overlap_text = chunks[i - 1][-overlap:]
                    overlapped_chunks.append(overlap_text + chunks[i])
                else:
                    overlapped_chunks.append(chunks[i])
            return overlapped_chunks

        return chunks

    # Wende den Absatz-Splitter auf alle Dokumente an.
    all_chunks = []
    for doc in documents:
        text_chunks = split_text_by_paragraph(doc.page_content)
        for i, chunk in enumerate(text_chunks):
            chunk_doc = Document(
                page_content=chunk,
                metadata={**doc.metadata, "chunk_index": i},
            )
            all_chunks.append(chunk_doc)

    return all_chunks

def split_documents_dynamic(documents, min_words=10, max_words=200, overlap=20):
    chunks = []

    for doc in documents:
        # Get the text content
        text = doc.page_content

        # Tokenize the text into words
        words = text.split()

        # Initialize variables for dynamic chunking
        current_chunk = []
        current_chunk_size = 0

        for i, word in enumerate(words):
            # Add the current word to the chunk
            current_chunk.append(word)
            current_chunk_size += 1

            # Check if the chunk has reached the maximum size
            if current_chunk_size >= max_words:
                # Join the words in the current chunk and add to chunks list
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))

                # Create overlap for the next chunk
                current_chunk = current_chunk[-overlap:]  # keep the last `overlap` words
                current_chunk_size = len(current_chunk)

            elif i == len(words) - 1 and current_chunk_size >= min_words:
                # If at the end of text and the current chunk meets minimum size
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))

    return chunks

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        length_function=len,
        is_separator_regex=False,
        separators=[
            ". \n \n \n",
            "\n\n\n",
            ". \n \n",
            "\n\n",
            ". ",
            ". \n",
            ".\n"
        ],
        chunk_size=800,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")

        # Batch size for processing documents to avoid exceeding token limits
        batch_size = 50  # Adjust this value as needed, e.g., 20, 10, etc.

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            new_chunk_ids = [chunk.metadata["id"] for chunk in batch]

            # Add documents to the database in batches
            db.add_documents(batch, ids=new_chunk_ids)
            db.persist()

        print("✅ All new documents added to the database")
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
