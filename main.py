from extras.constants import CONFIG_PATH
from preprocessor.extract import Processor
from extras.utils import read_yaml
from omission.extract_omission import OmissionExtractor
from omission.check_omission import MedicalOmissionChecker

if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    processor = Processor()
    checker = MedicalOmissionChecker(collection_name=config.get("collection_name"))
    omission_extractor = OmissionExtractor()
    marketing_post_text = processor.extract(config.get("marketing_doc"))
    marketing_post_text_cleaned = processor.clean_text(marketing_post_text)
    
    observation_info = omission_extractor.extract(marketing_post_text_cleaned)
    results = checker.process_observation(marketing_post_text_cleaned,observation_info)
    checker.display_results(results)