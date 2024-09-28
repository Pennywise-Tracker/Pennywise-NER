import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from spacy.tokens import Span
from typing import List, Dict

# Load the spaCy model
nlp = spacy.load("D:/Projects/Pennywise/Pennywise-NER/output/model-best")

# Create a FastAPI app
app = FastAPI()

# Define a request model
class TextRequest(BaseModel):
    text: str

# Define a response model
class EntityResponse(BaseModel):
    entities: List[Dict[str, str]]

@app.post("/predict", response_model=EntityResponse)
async def predict(request: TextRequest):
    # Process the input text
    doc = nlp(request.text)

    # Create a list to hold the new entities
    new_ents = list(doc.ents)

    # Check for numerical values and label them as "Expense"
    for token in doc:
        if token.like_num:
            new_ent = Span(doc, token.i, token.i + 1, label="EXPENSE")
            new_ents.append(new_ent)

    # Replace the doc's entities with the updated entities
    doc.ents = new_ents

    # Prepare the response
    entities = [{"label": ent.label_, "text": ent.text} for ent in doc.ents]
    return EntityResponse(entities=entities)

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)