# End-to-end script running the Hugging Face Trainer 
# for sequence classification. Based on the Tasks documentation 
# originally from: https://hf.co/docs/transformers/tasks/sequence_classification
import evaluate
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
    TrainingArguments, Trainer
)

# Constants
model_name = "distilbert-base-uncased"
dataset_name = "imdb"
metric = "accuracy"

# AutoModel requires the label mapping
id2label = {0:"NEGATIVE", 1:"POSITIVE"}
label2id = {v:k for k,v in id2label.items()}

# Load dataset
imdb = load_dataset(dataset_name)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(tokenize_function, batched=True)

# Create an efficient collator which dynamically pads
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Handle computation of our metrics
accuracy = evaluate.load(metric)

def compute_metrics(evaluation_preds):
    predictions, labels = evaluation_preds
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(
        predictions=predictions, references=labels
    )
    
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

# Define the hyperparameters in the TrainingArguments
training_args = TrainingArguments(
    output_dir="results", # Where weights are stored
    learning_rate=2e-5, # The learning rate during training
    per_device_train_batch_size=16, # Number of samples per batch during training
    per_device_eval_batch_size=16, # Number of samples per batch during evaluation
    num_train_epochs=2, # How many iterations through the dataloaders should be done
    weight_decay=0.01, # Regularization penalization
    evaluation_strategy="epoch", # How often metrics on the evaluation dataset should be computed
    save_strategy="epoch", # When to try and save the best model (such as a step number or every iteration)
)

# Create the `Trainer`, passing in the model and arguments
# the datasets to train on, how the data should be collated, 
# and the method for computing our metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Initiate training
trainer.train()

# Performing inference
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
# We need to tokenize the inputs and turn them to PyTorch tensors
encoded_input = tokenizer(text, return_tensors="pt")

# Then we can perform raw torch inference:
model.eval()
with torch.inference_mode():
    logits = model(**encoded_input).logits
    
# Finally, decode our outputs
predicted_class = logits.argmax().item()
print(f'Prediction: {id2label[predicted_class]}')
# Can also use `model.config.id2label` instead