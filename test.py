import spacy
from spacy.tokens import Span
from spacy import displacy

# Load your existing trained model
nlp1 = spacy.load(r".\output\model-best")

while True:
    text = input("Enter: ")

    if text.lower() == 'bye':
        print("Exiting.")
        break

    # Process the input text with the model
    doc = nlp1(text)

    # Create a list to hold the new entities
    new_ents = list(doc.ents)

    # Check for numerical values and label them as "EXPENSE" if not already labeled
    for token in doc:
        # Check if the token is already part of an entity
        is_entity = any([token.i >= ent.start and token.i < ent.end for ent in doc.ents])

        if token.like_num and not is_entity:
            # Create a new entity with the label "EXPENSE"
            new_ent = Span(doc, token.i, token.i + 1, label="EXPENSE")
            new_ents.append(new_ent)

    try:
        # Replace the doc's entities with the updated entities, ensuring no overlap
        doc.ents = new_ents

        # Print out the labeled entities
        for ent in doc.ents:
            print(f"{ent.label_}: {ent.text}")

        # Visualize the entities
        displacy.render(doc, style="ent")

    except ValueError as e:
        print(f"Error while setting entities: {e}")