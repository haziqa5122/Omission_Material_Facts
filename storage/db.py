from aperturedb import Connector
import numpy as np
from nomic import embed
from aperturedb.CommonLibrary import create_connector
from dotenv import load_dotenv

load_dotenv()
class VectorStore:
    def __init__(self, collection_name: str):
        """
        Initializes the ApertureDB client.

        :param host: The database instance name or IP (without http://).
        :param user: Username for authentication.
        :param password: Password for authentication.
        """
        self.client = create_connector(key = os.getenv("APERTUREDB_API_KEY"))
        self.client.query([{"GetStatus": {}}])  # Verify connection
        self.descriptorset_name = collection_name

    def set_collection(self, dimensions: int = 512):
        """
        Sets the descriptor set (collection) to be used. If it doesn't exist, it creates one.

        :param collection_name: Name of the descriptor set.
        :param dimensions: Dimensionality of the embeddings.
        """
        q = [{
            "AddDescriptorSet": {
                "name": self.descriptorset_name,
                "dimensions": dimensions,
                "engine": "Flat",
                "metric": "L2",
                "properties": {
                    "year_created": 2025,
                    "source": "ApertureDB dataset",
                }
            }
        }]
        return self.client.query(q)

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
            embedding = np.array(embedding, dtype=np.float32).tobytes()
            metadata = metadatas[idx] if metadatas else {}
            q = {
                "AddDescriptor": {
                    "set": self.descriptorset_name,
                    "label": metadata.get("_label", "unknown"),
                    "properties": {"id": ids[idx],
                                    **metadata},
                    "if_not_found": {"id": ["==", ids[idx]]}
                }
            }
            
            queries.append(q)
            blobs.append(embedding)
        
        print(len(blobs))
        return self.client.query(queries, blobs)
    
    def query_embeddings(self, query_embedding: np.ndarray, top_k: int = 5, return_images: bool = True):
        if self.descriptorset_name is None:
            raise ValueError("Descriptor set is not set. Use 'set_collection' first.")

        if not isinstance(query_embedding, np.ndarray):
            print("Changed")
            query_embedding = np.array(query_embedding, dtype=np.float32)

        embedding_blob = query_embedding.tobytes()

        q = [{
            "FindDescriptor": {
                "set": self.descriptorset_name,
                "k_neighbors": top_k,
                "results": { 
                    "list": ["id", "text","table_text","image","type"]
                }
            }
        }]

        responses, blobs = self.client.query(q, [embedding_blob])
        print("responses", responses, "blobs", blobs)
        
        # Check if the query was successful
        if not responses or "FindDescriptor" not in responses[0]:
            return []
        
        # Handle case where no descriptors are found
        descriptors = responses[0]["FindDescriptor"].get("entities", [])
        if not descriptors:
            return []

        results = []
        for d in descriptors:
            # Safely access nested properties
            props = d.get("properties", {})
            result = {
                "id": props.get("id"),
                "label": d.get("_label"),
                "metadata": props,
                "score": d.get("score")
            }

            # Fetch image if label is "image" and return_images is True
            if d.get("label") == "image" and return_images and props.get("id"):
                q_img = [{
                    "FindImage": {
                        "constraints": {"id": ["==", props["id"]]},
                        "blobs": True,
                        "results": {"limit": 1}
                    }
                }]
                try:
                    resp, img_blobs = self.client.query(q_img)
                    if img_blobs and len(img_blobs) > 0:
                        result["image_blob"] = img_blobs[0]
                except Exception as e:
                    print(f"Error fetching image for id {props['id']}: {e}")

            results.append(result)

        return results

    def delete_descriptor_set(self, set_name: str = None, confirm: bool = False):
        """
        Delete a descriptor set and all its descriptors.
        
        :param set_name: Name of the descriptor set to delete. If None, uses self.descriptorset_name
        :param confirm: Safety flag - must be True to actually delete
        :return: Response from ApertureDB
        """
        name_to_delete = set_name if set_name else self.descriptorset_name
        
        if name_to_delete is None:
            raise ValueError("No descriptor set name provided. Either pass set_name or use 'set_collection' first.")
        
        if not confirm:
            raise ValueError(
                f"Are you sure you want to delete descriptor set '{name_to_delete}'? "
                "This will delete ALL descriptors in it! "
                "Set confirm=True to proceed."
            )
        
        q = [{
            "DeleteDescriptorSet": {
                "with_name": name_to_delete
            }
        }]
        
        print(f"🗑️  Deleting descriptor set: '{name_to_delete}'...")
        response, _ = self.client.query(q)
        
        if response[0]["DeleteDescriptorSet"]["status"] == 0:
            print(f"✓ Successfully deleted descriptor set '{name_to_delete}'")
        else:
            print(f"❌ Failed to delete: {response}")
        
        return response
        
    def delete_descriptors(self, ids: list = None, delete_all: bool = False):
        """
        Deletes descriptors from ApertureDB.
        
        :param ids: A list of unique IDs of descriptors to delete. If None and delete_all is False, nothing is deleted.
        :param delete_all: If True, deletes all descriptors in the descriptor set. Use with caution!
        :return: Response from ApertureDB
        """
        if self.descriptorset_name is None:
            raise ValueError("Descriptor set is not set. Use 'set_collection' first.")
        
        if not delete_all and not ids:
            raise ValueError("Either provide 'ids' to delete specific descriptors or set 'delete_all=True'")
        
        queries = []
        
        if delete_all:
            # Delete all descriptors in the set
            q = {
                "DeleteDescriptor": {
                    "set": self.descriptorset_name
                }
            }
            queries.append(q)
        else:
            # Delete specific descriptors by ID
            for id_val in ids:
                q = {
                    "DeleteDescriptor": {
                        "set": self.descriptorset_name,
                        "constraints": {
                            "id": ["==", id_val]
                        }
                    }
                }
                queries.append(q)
        
        print(f"Deleting {len(queries)} descriptor(s)")
        return self.client.query(queries)

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
