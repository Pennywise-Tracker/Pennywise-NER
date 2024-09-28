import json
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.load("en_core_web_trf")
# nlp = spacy.load("en_core_web_lg")

# Load the annotations from the JSON file
with open('annotations.json', 'r') as f:
    data = json.load(f)

annotations = data["annotations"]

# Initialize DocBin for storing processed documents
db = DocBin()

# Iterate over each text and its corresponding annotations
for text, annot in tqdm(annotations):
    print(f"\nProcessing text: {text}")
    doc = nlp.make_doc(text)
    ents = []

    # Iterate over each entity in the annotation
    for start, end, label in annot["entities"]:
        print(f"Entity: '{text[start:end]}' | Start: {start} | End: {end} | Label: {label}")
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        
        if span is None:
            print(f"  Skipping entity: '{text[start:end]}' from position {start} to {end} with label '{label}'")
        else:
            ents.append(span)
    
    # Set the document's entities and add it to the DocBin
    doc.ents = ents
    db.add(doc)

# Save the processed data to disk
db.to_disk("./train.spacy")

# import json
# import spacy
# from spacy.tokens import DocBin
# from tqdm import tqdm

# # Load the spaCy model
# nlp = spacy.load("en_core_web_lg")

# # Load the training annotations from the JSON file
# with open('annotations.json', 'r') as f:
#     data = json.load(f)

# # Initialize DocBin for storing processed documents
# train_db = DocBin()
# valid_db = DocBin()

# # Assuming your JSON structure has both training and validation annotations
# annotations = data["annotations"]
# valid_annotations = data.get("valid_annotations", [])  # Adjust according to your JSON structure

# # Process training annotations
# for text, annot in tqdm(annotations):
#     print(f"\nProcessing training text: {text}")
#     doc = nlp.make_doc(text)
#     ents = []

#     for start, end, label in annot["entities"]:
#         print(f"Entity: '{text[start:end]}' | Start: {start} | End: {end} | Label: {label}")
#         span = doc.char_span(start, end, label=label, alignment_mode="contract")

#         if span is None:
#             print(f"  Skipping entity: '{text[start:end]}' from position {start} to {end} with label '{label}'")
#         else:
#             ents.append(span)

#     doc.ents = ents
#     train_db.add(doc)

# # Save the processed training data to disk
# train_db.to_disk("./train.spacy")

# # Process validation annotations
# for text, annot in tqdm(valid_annotations):
#     print(f"\nProcessing validation text: {text}")
#     doc = nlp.make_doc(text)
#     ents = []

#     for start, end, label in annot["entities"]:
#         print(f"Entity: '{text[start:end]}' | Start: {start} | End: {end} | Label: {label}")
#         span = doc.char_span(start, end, label=label, alignment_mode="contract")

#         if span is None:
#             print(f"  Skipping entity: '{text[start:end]}' from position {start} to {end} with label '{label}'")
#         else:
#             ents.append(span)

#     doc.ents = ents
#     valid_db.add(doc)

# # Save the processed validation data to disk
# valid_db.to_disk("./valid.spacy")
