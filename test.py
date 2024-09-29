import spacy
from spacy.tokens import Span
from spacy import displacy

# Load your existing trained model
nlp1 = spacy.load(r".\output\model-best")

# Define greetings and responses
greetings = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! What can I do for you?",
    "how are you": "I'm just a program, but thanks for asking! How can I help you?",
    "good morning": "Good morning! How can I assist you today?",
    "good evening": "Good evening! How can I help you?",
}

while True:
    text = input("Enter: ").lower()  # Convert input to lowercase

    if text == 'bye':
        print("Exiting.")
        break

    # Check for greetings
    if text in greetings:
        print(greetings[text])
        continue  # Skip the spaCy processing

    # Process the input text with the model
    doc = nlp1(text)

    # Step 1: Create a list to hold the new entities and prioritize EXPENSE detection
    new_ents = []

    # Check for numerical values and label them as "EXPENSE"
    for token in doc:
        # If it's a number, it must be labeled as EXPENSE
        if token.like_num:
            new_ent = Span(doc, token.i, token.i + 1, label="EXPENSE")
            new_ents.append(new_ent)

    # Step 2: Handle other entities but remove those overlapping with EXPENSE
    for ent in doc.ents:
        # Check if this entity does not overlap with any of the new_ents
        if not any(ent.start <= exp_ent.start < ent.end for exp_ent in new_ents):
            new_ents.append(ent)

    try:
        # Replace the doc's entities with the updated entities, ensuring no overlap
        doc.ents = new_ents

        # Print out the labeled entities
        if doc.ents:
            for ent in doc.ents:
                print(f"{ent.label_}: {ent.text}")
        else:
            print("Sorry, I did not get that.")
        
        # Visualize the entities
        displacy.render(doc, style="ent")

    except ValueError as e:
        print(f"Error while setting entities: {e}")