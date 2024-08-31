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

    category_descriptions = {}
    category_amounts = {}
    category_order = []
    current_category = None

    for ent in doc.ents:
        if ent.label_ == "EXPENSE":
            if current_category:
                if current_category not in category_amounts:
                    category_amounts[current_category] = 0.0
                try:
                    amount = float(ent.text.replace(',', '').replace('Rs', '').strip())
                    category_amounts[current_category] += amount
                except ValueError:
                    pass
            else:
                if "EXPENSE" not in category_amounts:
                    category_amounts["EXPENSE"] = 0.0
                try:
                    amount = float(ent.text.replace(',', '').replace('Rs', '').strip())
                    category_amounts["EXPENSE"] += amount
                except ValueError:
                    pass
        else:
            if ent.label_ not in category_descriptions:
                category_descriptions[ent.label_] = []
                category_amounts[ent.label_] = 0.0
                category_order.append(ent.label_)
            category_descriptions[ent.label_].append(ent.text)
            current_category = ent.label_

    results = []
    for category in category_order:
        result = {
            "category": category,
            "description": ", ".join(category_descriptions[category]),
            "amount": category_amounts[category]
        }
        results.append(result)

    if "EXPENSE" in category_amounts and category_amounts["EXPENSE"] > 0:
        result = {
            "category": "",
            "description": "",
            "amount": category_amounts["EXPENSE"]
        }
        results.append(result)

    if not results or (len(results) == 1 and "amount" in results[0] and results[0]["amount"] == 0.0):
        response = {
            "user_id": request.user_id,
            "text": "Oops, it looks like you forgot to mention an expense!"
        }
    else:
        all_results_valid = True
        for res in results:
            if not res["category"] or not res["description"]:
                all_results_valid = False
                break
        
        if not all_results_valid:
            response = {
                "user_id": request.user_id,
                "text": "Sorry, I didn't quite get that."
            }
        else:
            response = {
                "user_id": request.user_id,
                "expenses": results
            }

    return response



# RUN COMMAND
# uvicorn main:app --reload