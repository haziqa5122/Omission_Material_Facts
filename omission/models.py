# models.py
from pydantic import BaseModel
from typing import List

class MedicalOmissionInfo(BaseModel):
    omitted_side_effects_and_risks: List[str]
    omitted_contraindications: List[str]
    omitted_safety_information: List[str]
    omitted_efficacy_and_limitations: List[str]
    omitted_clinical_evidence: List[str]
