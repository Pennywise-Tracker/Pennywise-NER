import spacy

nlp1 = spacy.load(r".\output\model-best")

text = "I got a Netflix subscription for Rs 700"
# text = "Just renewed my Hotstar for Rs 999"
doc = nlp1(text)

for ent in doc.ents:
    print(f"{ent.label_}: {ent.text}")

from spacy import displacy
displacy.render(doc, style="ent")