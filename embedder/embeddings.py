from sentence_transformers import SentenceTransformer
from extras.utils import read_yaml
from extras.constants import CONFIG_PATH
import numpy as np


config = read_yaml(CONFIG_PATH)
model = SentenceTransformer(config.get("embedding_model"))


def get_embeddings(sentences:list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of sentences using a pre-trained model.

    Args:
        sentences (list[str]): List of sentences to encode.
        batch_size (int, optional): Number of sentences to process at once (default is 32).

    Returns:
        np.ndarray: Array of sentence embeddings.
    """
    if not sentences:
        raise ValueError("The list of sentences is empty.")
    
    embeddings = model.encode(sentences)
    return embeddings