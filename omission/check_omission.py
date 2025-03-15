from typing import List, Dict, Tuple
from extras.constants import CONFIG_PATH
from extras.utils import read_yaml
from pydantic import BaseModel
from storage.db import VectorStore
from embedder.embeddings import get_embeddings
from omission.models import MedicalOmissionInfo
from colorama import Fore, Style, Back
from openai import OpenAI
import re


class MedicalOmissionChecker:
    def __init__(self, collection_name: str):
        config = read_yaml(CONFIG_PATH)
        self.vector_store = VectorStore(host=config.get("db_host"), user=config.get("db_user"), password=config.get("db_password"))
        self.vector_store.set_collection(collection_name)
        self.client = OpenAI()

    def _query_observation(self, observations: List[str]) -> Dict[str, List[str]]:
        """Query Aperture DB for each claim and return relevant documents."""
        relevant_docs = {}
        for observation in observations:
            embeddings = get_embeddings(observation)
            documents = self.vector_store.query_embeddings(embeddings)
            relevant_docs[observation] = documents
        return relevant_docs

    def _check_consistency(self, post:str, observation: str, category: str, documents: List[str]) -> str:
        """Use an LLM to determine if the observation and documents are consistent."""
        if not documents:
            return "No documents found."

        prompt = (
        f"This is the post: {post}\n"
        f"Your task is to evaluate the post to assess whether it misses any important information that should be told.\n"
         f"The following observation was made about the post: '{category}':'{observation}'.\n"
         f"The following documents are provided to guide the reviewer in assessing the observation: {documents}.\n"
        "Evaluate whether the provided documents contains the identified observation:\n"
        "- If it does 'Omission'.\n"
        "- Otherwise 'Fine'.\n"
    "Provide a concise explanation for your conclusion in the following format:\n\n"
    "<answer>\n"
    "[Omission/Fine] - [Reason]\n"
    "</answer>"
    )
        # try:
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Medical Legal Reviewer."},
                {"role": "user", "content": prompt},
            ],
        )
        match = re.search(r"<answer>(.*?)</answer>", completion.choices[0].message.content, re.DOTALL)
        if match:
            return match.group(1).strip()
        # except Exception as e:
        #     return f"Error: {e}"

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
    
    def display_results(self, results: Dict[str, List[Tuple[str, str]]]):
        """Display flagged claims in the terminal with appropriate colors."""
        for category, observations in results.items():
            print(f"\nCategory: {category}")
            for observation, status in observations:
                if "Omission" in status:
                    print(Fore.RED + f"Observation: {observation} -> Status: {status}" + Style.RESET_ALL)
                else:
                    print(Fore.GREEN + f"Observation: {observation} -> Status: {status}" + Style.RESET_ALL)
