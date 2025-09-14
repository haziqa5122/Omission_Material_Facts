from aperturedb import Connector
import numpy as np
from nomic import embed

class VectorStore:
    def __init__(self, host: str, user: str, password: str):
        """
        Initializes the ApertureDB client.

        :param host: The database instance name or IP (without http://).
        :param user: Username for authentication.
        :param password: Password for authentication.
        """
        self.client = Connector.Connector(host=host, user=user, password=password)
        self.client.query([{"GetStatus": {}}])  # Verify connection
        self.descriptorset_name = None

    def set_collection(self, collection_name: str, dimensions: int = 1024):
        """
        Sets the descriptor set (collection) to be used. If it doesn't exist, it creates one.

        :param collection_name: Name of the descriptor set.
        :param dimensions: Dimensionality of the embeddings.
        """
        self.descriptorset_name = collection_name
        q = [{
            "AddDescriptorSet": {
                "name": collection_name,
                "dimensions": dimensions,
                "engine": "Flat",
                "metric": "L2",
                "properties": {
                    "year_created": 2025,
                    "source": "ApertureDB dataset",
                }
            }
        }]
        self.client.query(q)

    def ingest_embeddings(self, embeddings: np.ndarray, ids: list, metadatas: list = None):
        """
        Ingests embeddings along with metadata into ApertureDB.

        :param embeddings: The embeddings (as a NumPy array) to be stored.
        :param ids: A list of unique IDs for each embedding.
        :param metadatas: A list of metadata dictionaries for each embedding.
        """
        if self.descriptorset_name is None:
            raise ValueError("Descriptor set is not set. Use 'set_collection' first.")
        
        queries = []
        blobs = []
        
        for idx, embedding in enumerate(embeddings):
            embedding_bytes = embedding.astype('float32').tobytes()
            metadata = metadatas[idx] if metadatas else {}
            
            q = {
                "AddDescriptor": {
                    "set": self.descriptorset_name,
                    "label": metadata.get("label", "unknown"),
                    "properties": {"id": ids[idx], **metadata},
                    "if_not_found": {"id": ["==", ids[idx]]}
                }
            }
            
            queries.append(q)
            blobs.append(embedding_bytes)
        
        self.client.query(queries, blobs)

    def query_embeddings(self, query_embedding: np.ndarray, top_k: int = 5, return_images: bool = True):
        """
        Queries the ApertureDB for similar embeddings.
        If results are images and `return_images=True`, also fetch the image blobs.

        :param query_embedding: The query embedding to search for similar items.
        :param top_k: Number of similar embeddings to return.
        :param return_images: Whether to fetch actual images for image descriptors.
        :return: List of results with id, label, metadata, score, and optional image blob.
        """
        if self.descriptorset_name is None:
            raise ValueError("Descriptor set is not set. Use 'set_collection' first.")

        embedding_bytes = query_embedding.astype("float32").tobytes()

        q = [{
            "FindDescriptor": {
                "set": self.descriptorset_name,
                "k": top_k,
                "return": ["id", "label", "properties", "score"]
            }
        }]

        responses, _ = self.client.query(q, [embedding_bytes])
        descriptors = responses[0]["FindDescriptor"]["descriptors"]

        results = []
        for d in descriptors:
            result = {
                "id": d["properties"]["id"],
                "label": d["label"],
                "metadata": d["properties"],
                "score": d.get("score")
            }

            # If this is an image descriptor and we want blobs, fetch the image
            if d["label"] == "image" and return_images:
                q_img = [{
                    "FindImage": {
                        "constraints": {"id": ["==", d["properties"]["id"]]},
                        "blobs": True,
                        "results": {"limit": 1}
                    }
                }]
                resp, blobs = self.client.query(q_img)
                if blobs:
                    result["image_blob"] = blobs[0]

            results.append(result)

        return results

    
    def add_image(self, image_path: str, metadata: dict):
        """
        Adds an image along with metadata to ApertureDB.

        :param image_path: Path to the image file.
        :param metadata: Dictionary containing metadata for the image.
        """
        with open(image_path, 'rb') as fd:
            image_blob = [fd.read()]
        
        q = [{
            "AddImage": {
                "properties": metadata,
                "if_not_found": {"id": ["==", metadata["id"]]}
            }
        }]
        
        response, _ = self.client.query(q, image_blob)
        return response

    def add_image_with_embedding(self, image_path: str, metadata: dict):
        """
        Adds an image to ApertureDB, generates its embedding with nomic,
        and stores both image + embedding into the database.

        :param image_path: Path to the image file.
        :param metadata: Dictionary containing metadata for the image.
                        Must include a unique "id".
        """
        if self.descriptorset_name is None:
            raise ValueError("Descriptor set is not set. Use 'set_collection' first.")

        with open(image_path, "rb") as fd:
            image_blob = [fd.read()]

        q_img = [{
            "AddImage": {
                "properties": metadata,
                "if_not_found": {"id": ["==", metadata["id"]]}
            }
        }]
        self.client.query(q_img, image_blob)

        output = embed.image(
            images=[image_path],
            model="nomic-embed-vision-v1.5",
        )
        embedding = np.array(output["embeddings"][0], dtype="float32")
        embedding_bytes = embedding.tobytes()

        q_desc = [{
            "AddDescriptor": {
                "set": self.descriptorset_name,
                "label": metadata.get("label", "image"),
                "properties": {"id": metadata["id"], **metadata},
                "if_not_found": {"id": ["==", metadata["id"]]}
            }
        }]
        self.client.query(q_desc, [embedding_bytes])

        return {"image_added": True, "embedding_shape": embedding.shape}
