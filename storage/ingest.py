from extras.constants import CONFIG_PATH
from preprocessor.extract import Processor
from extras.utils import read_yaml, extract_images
from unstructured.chunking.title import chunk_by_title
from embedder.embeddings import get_embeddings
from storage.db import VectorStore
import numpy as np

if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)

    # Initialize VectorStore (ApertureDB)
    vector_store = VectorStore(
        host=config.get("db_host"),
        user=config.get("db_user"),
        password=config.get("db_password")
    )
    vector_store.set_collection(config.get("collection_name"))

    # Extract document elements
    pdf_path = config.get("clinical_doc")
    images_info = extract_images(pdf_path=pdf_path)
    processor = Processor()
    clinical_doc_elements = processor.extract(document=pdf_path)

    # Chunking document by title
    chunks = chunk_by_title(clinical_doc_elements)

    # Extract tables
    tables = [el for el in clinical_doc_elements if el.category == "Table"]
    
    print(f"Total Tables Found: {len(tables)}")
    if tables:
        print(f"Table Metadata: {tables[0].metadata.page_number}")
    if chunks:
        print(f"Chunk Metadata: {chunks[0].metadata.page_number}")

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
        combined_text = "\n".join(data["text"])
        embedding = get_embeddings(combined_text)
        embeddings.append(embedding)
        ids.append(str(page_number))
        metadatas.append({"text": combined_text, "page_number": page_number})

        if data["table"]:
            metadatas[-1]["table"] = data["table"].text

        if data["image"]:
            image_metadata = {
                "id": data["image"]["id"],
                "page": data["image"]["page"]
            }
            vector_store.add_image(data["image"]["filename"], image_metadata)

    vector_store.ingest_embeddings(np.array(embeddings), ids, metadatas)
