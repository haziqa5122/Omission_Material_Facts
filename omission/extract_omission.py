from openai import OpenAI
from dotenv import load_dotenv
from omission.models import MedicalOmissionInfo

load_dotenv()
# Initialize OpenAI client
client = OpenAI()

class OmissionExtractor:

    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        """
        Initializes the extractor with the specified OpenAI model.
        
        Args:
            model (str): The OpenAI model to use for parsing. Default is 'gpt-4o-2024-08-06'.
        """
        self.model = model

    def extract(self, text: str) -> MedicalOmissionInfo:
        """
        Extracts drug information, claims, and corresponding parts of the text.

        Args:
            text (str): The input text to analyze.

        Returns:
            DrugInfo: A structured representation of the drug-related information.
        """
        prompt = (
    "Analyze the document and assess whether critical information is omitted under the following categories. Provide a general statement for each category about whether omissions are present and what their potential effect might be without relying on specific medical knowledge:\n"
    "- Side Effects and Risks: Evaluate whether any discussion of potential adverse reactions is missing and the implications of this omission for informed decision-making.\n"
    "- Contraindications: Determine if the document excludes mention of specific conditions or populations for whom the drug might be unsafe and explain how this could affect general understanding.\n"
    "- Safety Information: Assess if general safety warnings or advisories are absent and how this might impact the perception of safety.\n"
    "- Efficacy and Limitations: Identify whether information about the drug’s effectiveness or situations where it may not work is omitted and consider the potential effect on setting realistic expectations.\n"
    "- Clinical Evidence and Research: Check if references to studies, data, or regulatory approvals are omitted and discuss how this might influence trust or credibility.\n"
    "For each category where omissions are identified, provide a generalized description of the potential effects without making specific medical assumptions or recommendations."
)


        completion = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            response_format=MedicalOmissionInfo,
        )
        return completion.choices[0].message.parsed