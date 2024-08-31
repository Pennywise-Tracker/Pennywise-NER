from fastapi import FastAPI, HTTPException
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
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return {"user_id":request.user_id ,"entities": entities}

# RUN COMMAND
#uvicorn main:app --reload

#Output
# {
#     "user_id" : ""
#     "category" : "",
#     "description": "",
#     "amount" : ""
# }

