import spacy
from spacy import displacy

nlp1 = spacy.load(r".\output\model-best")

while True:
    text = input("Enter: ")

    if text.lower() == 'bye':
        print("Exiting.")
        break

    doc = nlp1(text)

    for ent in doc.ents:
        print(f"{ent.label_}: {ent.text}")

    displacy.render(doc, style="ent")