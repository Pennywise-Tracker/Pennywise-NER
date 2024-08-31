from fastapi import FastAPI
from pydantic import BaseModel
import spacy

nlp = spacy.load("D:/Projects/Pennywise/Pennywise-NER/output/model-best")

app = FastAPI()

class TextRequest(BaseModel):
    user_id: str
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    doc = nlp(request.text)

    # Dictionary to store aggregated results by category
    aggregated_results = {}

    # First pass: Collect all amounts and descriptions
    for ent in doc.ents:
        if ent.label_ == "EXPENSE":
            # Store the amounts in a list
            if ent.label_ not in aggregated_results:
                aggregated_results[ent.label_] = {
                    "amounts": [],
                    "descriptions": []
                }
            aggregated_results[ent.label_]["amounts"].append(ent.text)
        else:
            # Store descriptions for non-EXPENSE entities
            if ent.label_ not in aggregated_results:
                aggregated_results[ent.label_] = {
                    "amounts": [],
                    "descriptions": []
                }
            aggregated_results[ent.label_]["descriptions"].append(ent.text)

    # Prepare final results
    results = []
    for category, data in aggregated_results.items():
        if category == "EXPENSE":
            # Only process if there are amounts collected
            continue
        
        result = {
            "user_id": request.user_id,
            "category": category,
            "description": ", ".join(data["descriptions"]),
            "amount": ", ".join(aggregated_results.get("EXPENSE", {}).get("amounts", []))
        }
        results.append(result)

    return results




# RUN COMMAND
# uvicorn main:app --reload

#Output
# {
#     "user_id" : "1"
#     "category" : "",
#     "description": "",
#     "amount" : ""
# }

