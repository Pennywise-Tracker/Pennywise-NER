import json
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

with open('annotations.json', 'r') as f:
    data = json.load(f)

annotations = data["annotations"]

db = DocBin()

for text, annot in tqdm(annotations):
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print(f"Skipping entity: '{text[start:end]}' from position {start} to {end} with label '{label}'")
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("./train.spacy")