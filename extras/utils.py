import yaml
import fitz
import os
import pdfplumber


def read_yaml(file_path):
        """
        Reads a YAML file and returns its contents as a Python dictionary.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            dict: Parsed YAML content as a dictionary.
        """
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)  # Safely parse the YAML file
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
        return None

def extract_images(pdf_path, output_folder="extracted_images"):
    doc = fitz.open(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    images_info = []

    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc[page_num].get_images(full=True), start=1):
            xref = img[0]  # XREF index
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{output_folder}/page_{page_num+1}_img_{img_index}.{image_ext}"

            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)

            images_info.append({"page": page_num+1, "image": image_filename})

    return images_info