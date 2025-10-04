from sentence_transformers import SentenceTransformer
from extras.utils import read_yaml
from extras.constants import CONFIG_PATH
import numpy as np
from PIL import Image

config = read_yaml(CONFIG_PATH)
model = SentenceTransformer(config.get("multimodal_embedding_model"))


def get_multimodal_embedding(input_data, is_image=False):
    if is_image:
        img = Image.open(input_data)
        return model.encode(img, convert_to_tensor=True)
    else:
        return model.encode(input_data, convert_to_tensor=True)