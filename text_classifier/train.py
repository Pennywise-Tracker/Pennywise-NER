import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Load the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

df = pd.read_csv('classifier_data.csv')

le = LabelEncoder()
df['encoded-intent'] = le.fit_transform(df['intent'])
le_name_mapping = dict(zip(le.transform(le.classes_),le.classes_))

# @title Description Consolidated


df.groupby('intent').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

#Split Data
X_train, x_test, y_train, y_test = train_test_split(df['description'], df['encoded-intent'], test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

for value in [x_train, x_val, x_test, y_train, y_val, y_test]:
    print(len(value))
    
train_df = Dataset.from_pandas(pd.DataFrame({'text': x_train, 'label': y_train}))
val_df = Dataset.from_pandas(pd.DataFrame({'text': x_val, 'label': y_val}))
test_df = Dataset.from_pandas(pd.DataFrame({'text': x_test, 'label': y_test}))

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_tokenized_df = train_df.map(tokenize_function, batched=True)
val_tokenized_df = val_df.map(tokenize_function, batched=True)
test_tokenized_df = test_df.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to = "none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_df,
    eval_dataset=val_tokenized_df,
)

# Train the model
trainer.train()

# Make Predictions
y_pred = trainer.predict(test_tokenized_df)

predicted_labels = np.argmax(y_pred.predictions, axis=1)


accuracy_score(np.array(y_test), predicted_labels)

trainer.save_model(output_dir="./text-classifier")

# Test with user input
# user_input = input("Enter text: ")
# tokenized_input = tokenizer(user_input, return_tensors='pt').to(device)
# with torch.no_grad():
#         logits = model(**tokenized_input).logits
# predicted_class = logits.argmax().item()
# print(le_name_mapping[predicted_class])