import json

class InterviewGuideline:
    def __init__(self):
        self.guidelines = {
            "general": [
                "Dress formally and be punctual.",
                "Carry all required documents in an organized manner.",
                "Be honest and concise in your answers."
            ],
            "common_questions": [
                "Why do you want to visit the Schengen area?",
                "What is the purpose of your visit?",
                "How long do you plan to stay?",
                "Do you have travel insurance?",
                "Can you provide proof of financial stability?"
            ],
            "embassy_specific": {
                "Germany": [
                    "Be prepared to explain your travel itinerary in detail.",
                    "Know the address of your accommodation."
                ],
                "France": [
                    "Be ready to discuss your travel history.",
                    "Have a clear plan for your stay in France."
                ]
            }
        }

    def get_general_guidelines(self):
        return self.guidelines["general"]

    def get_common_questions(self):
        return self.guidelines["common_questions"]

    def get_embassy_specific_guidelines(self, embassy):
        return self.guidelines["embassy_specific"].get(embassy, [])

    def get_all_guidelines(self, embassy):
        return {
            "general": self.get_general_guidelines(),
            "common_questions": self.get_common_questions(),
            "embassy_specific": self.get_embassy_specific_guidelines(embassy)
        }

if __name__ == "__main__":
    guideline = InterviewGuideline()
    embassy = "France"  # Example embassy
    all_guidelines = guideline.get_all_guidelines(embassy)
    print(json.dumps(all_guidelines, indent=4))