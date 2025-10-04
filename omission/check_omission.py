from typing import List, Dict, Tuple
from extras.constants import CONFIG_PATH
from extras.utils import read_yaml
from pydantic import BaseModel
from storage.db import VectorStore
from embedder.multimodal_embedding import get_multimodal_embedding
from omission.models import MedicalOmissionInfo
from colorama import Fore, Style, Back
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
import re

load_dotenv()

class ConsistencyCheck(BaseModel):
    status: Literal["Omission", "Fine", "No documents found"]
    reason: str

class MedicalOmissionChecker:
    def __init__(self, collection_name: str):
        config = read_yaml(CONFIG_PATH)
        self.vector_store = VectorStore(collection_name)
        self.vector_store.set_collection()
        self.client = OpenAI()

    def _query_observation(self, observations: List[str]) -> Dict[str, List[str]]:
        """Query Aperture DB for each claim and return relevant documents."""
        relevant_docs = {}
        for observation in observations:
            embeddings = get_multimodal_embedding(observation)
            documents = self.vector_store.query_embeddings(embeddings)
            relevant_docs[observation] = documents
        return relevant_docs

    def _check_consistency(self, post: str, observation: str, category: str, documents: list[str]) -> ConsistencyCheck:
        """Use an LLM to determine if the observation and documents are consistent."""

        if not documents:
            return ConsistencyCheck(status="No documents found", reason="No supporting references available")

        prompt = f"""
    This is the post: {post}

    Your task is to evaluate whether the post omits important information.

    Observation category: {category}
    Observation: {observation}

    Supporting documents:
    {documents}

    Decide if the documents support the observation or not:
    - If yes, return status "Omission"
    - If not, return status "Fine"

    Provide also a short explanation in plain text.
    """

        response = self.client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": "You are a Medical Legal Reviewer. Output strictly as JSON."},
                {"role": "user", "content": prompt},
            ],
            text_format=ConsistencyCheck,
        )

        return response.output_parsed


    def process_observation(self, post:str, observation_info: MedicalOmissionInfo) -> Dict[str, List[Tuple[str, str]]]:
        """Process all observation, cross-reference with Aperture DB, and check consistency."""
        results = {}
        observation_categories = {
            "omitted_side_effects_and_risks": observation_info.omitted_side_effects_and_risks,
            "omitted_contraindications": observation_info.omitted_contraindications,
            "omitted_safety_information": observation_info.omitted_safety_information,
            "omitted_efficacy_and_limitations": observation_info.omitted_efficacy_and_limitations,
            "omitted_clinical_evidence": observation_info.omitted_clinical_evidence,
        }

        for category, observations in observation_categories.items():
            if not observations:
                results[category] = [("No observation provided", "No documents found")]
                continue

            relevant_docs = self._query_observation(observations)
            category_results = []

            for observation, documents in relevant_docs.items():
                consistency = self._check_consistency(post, observation, category, documents)
                category_results.append((observation, consistency))

            results[category] = category_results

        return results
    
    def display_results(self, results: Dict[str, List[Tuple[str, "ConsistencyCheck"]]]):
        """
        Display flagged claims in the terminal with appropriate colors.
        Expects results in the form:
        """
        for category, observations in results.items():
            print(f"\nCategory: {category}")
            for observation, result in observations:
                status = result.status
                reason = result.reason

                if status == "Omission":
                    print(
                        Fore.RED
                        + f"Observation: {observation}\n"
                        f"   → Status: {status}\n"
                        f"   → Reason: {reason}"
                        + Style.RESET_ALL
                    )
                elif status == "Fine":
                    print(
                        Fore.GREEN
                        + f"Observation: {observation}\n"
                        f"   → Status: {status}\n"
                        f"   → Reason: {reason}"
                        + Style.RESET_ALL
                    )
                else:
                    print(
                        Fore.YELLOW
                        + f"Observation: {observation}\n"
                        f"   → Status: {status}\n"
                        f"   → Reason: {reason}"
                        + Style.RESET_ALL
                    )



