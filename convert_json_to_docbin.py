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

# # Load your spaCy model
# nlp = spacy.load("en_core_web_trf")

# # Function to process annotations
# def process_annotations(file_path, output_path):
#     # Load the annotations from the JSON file
#     with open(file_path, 'r') as f:
#         data = json.load(f)

#     annotations = data["annotations"]

#     # Initialize DocBin for storing processed documents
#     db = DocBin()

#     # Iterate over each text and its corresponding annotations
#     for text, annot in tqdm(annotations):
#         print(f"\nProcessing text: {text}")
#         doc = nlp.make_doc(text)
#         ents = []

#         # Iterate over each entity in the annotation
#         for start, end, label in annot["entities"]:
#             print(f"Entity: '{text[start:end]}' | Start: {start} | End: {end} | Label: {label}")
#             span = doc.char_span(start, end, label=label, alignment_mode="contract")
            
#             if span is None:
#                 print(f"  Skipping entity: '{text[start:end]}' from position {start} to {end} with label '{label}'")
#             else:
#                 ents.append(span)
        
#         # Set the document's entities and add it to the DocBin
#         doc.ents = ents
#         db.add(doc)

#     # Save the processed data to disk
#     db.to_disk(output_path)

# # Process training data
# process_annotations('train_annotations.json', './train.spacy')

# # Process validation data
# process_annotations('valid_annotations.json', './valid.spacy')

