from extras.constants import CONFIG_PATH
from preprocessor.extract import Processor
from extras.utils import read_yaml, extract_images
from unstructured.chunking.title import chunk_by_title
from embedder.multimodal_embedding import get_multimodal_embedding
from storage.db import VectorStore
import numpy as np

if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)

    # Initialize VectorStore (ApertureDB)
    vector_store = VectorStore(
        collection_name=config.get("collection_name"),
    )
    vector_store.set_collection(dimensions=512)

    # Extract document elements
    pdf_path = config.get("clinical_doc")
    images_info = extract_images(pdf_path=pdf_path)
    processor = Processor()
    clinical_doc_elements = processor.extract(document=pdf_path)

    # Chunking    document by title
    chunks = chunk_by_title(clinical_doc_elements)

    # Extract tables
    tables = [el for el in clinical_doc_elements if el.category == "Table"]

    # Organize and ingest data
    page_data = {}

    # Group text chunks by page
    for chunk in chunks:
        page_number = chunk.metadata.page_number
        if page_number not in page_data:
            page_data[page_number] = {"text": [], "table": None, "image": None}
        page_data[page_number]["text"].append(chunk.text)

    # Associate tables with the respective page
    for table in tables:
        page_number = table.metadata.page_number
        if page_number in page_data:
            page_data[page_number]["table"] = table

    # Associate images with the respective page
    for idx, image_info in enumerate(images_info, start=1):
        page_number = image_info["page"]
        if page_number in page_data:
            page_data[page_number]["image"] = {
                "filename": image_info["image"],
                "id": f"img_{idx}",  # Generate a unique ID for each image
                "page": page_number
            }

    # Process and ingest embeddings and images
    embeddings, ids, metadatas = [], [], []

    for page_number, data in page_data.items():
    # Combine all textual info
        combined_text = "\n".join(data["text"])
        if combined_text.strip():
            emb = get_multimodal_embedding(combined_text, is_image=False)
            embeddings.append(emb)
            ids.append(f"text_page_{page_number}")
            metadatas.append({
                "type": "text",
                "page_number": page_number,
                "text": combined_text
            })

        if data["table"]:
            table_text = data["table"].text
            emb = get_multimodal_embedding(table_text, is_image=False)
            embeddings.append(emb)
            ids.append(f"table_page_{page_number}")
            metadatas.append({
                "type": "table",
                "page_number": page_number,
                "table": table_text
            })

        if data["image"]:
            emb = get_multimodal_embedding(data["image"]["filename"], is_image=True)
            embeddings.append(emb)
            ids.append(data["image"]["id"])
            metadatas.append({
                "type": "image",
                "page_number": page_number,
                "image": data["image"]["filename"]
            })

    vector_store.ingest_embeddings(embeddings, ids, metadatas)
