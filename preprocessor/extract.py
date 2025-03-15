from unstructured.partition.pdf import partition_pdf
import easyocr
import os

class Processor():
    def __init__(self) -> None:
        self.model = easyocr.Reader(['en'])  # Load model    

    def clean_text(self, ocr_output):
        """
        Extracts text from OCR output and applies spell correction.

        Args:
            ocr_output (list): The OCR output containing bounding boxes, text, and probabilities.

        Returns:
            list: A list of corrected text strings.
        """
        corrected_texts = []
        for item in ocr_output:
            if len(item) > 1:  # Ensure the item has text content
                text = item[1]  # Extract the text
                corrected_texts.append(text)
        return " ".join(corrected_texts)
    
    def extract(self, document):
            """
            Extracts text from a document based on its type.
            
            Args:
                document (str): Path to the document file.

            Returns:
                list: Extracted text if the document is an image; None for unsupported types.
            """
            if not os.path.exists(document):
                raise FileNotFoundError(f"The document '{document}' does not exist.")

            file_extension = os.path.splitext(document)[1].lower()

            if file_extension in ['.png', '.jpg', '.jpeg']:  # Supported image formats
                # Use easyOCR for image processing
                #preprocessed_image = self.preprocess_image(document) #Not giving good results
                result = self.model.readtext(document)
                return result

            elif file_extension == '.pdf':
                result = partition_pdf(document,infer_table_structure=True, strategy='hi_res',  languages=["eng"])
                return result
            else:
                print(f"Unsupported file type: {file_extension}")
                return None
