# End-to-end script running the Hugging Face Trainer
# for token classification. Based on the Tasks documentation
# originally from: https://hf.co/docs/transformers/tasks/token_classification
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Constants
dataset_name = "wnut_17"
model_name = "distilbert-base-uncased"
metric = "seqeval"

# AutoModel requires the label mapping
id2label = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
}
label2id = {v: k for k, v in id2label.items()}

# Load dataset
print(f"Downloading dataset ({dataset_name})")
dataset = load_dataset(dataset_name)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    "Realigns tokens and labels and limits sequence length"
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


print(f"Tokenizing dataset for {model_name}...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create an efficient collator which dynamically pads
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Handle computation of our metrics
print(f"Loading metric ({metric})...")
seqeval = evaluate.load(metric)
# Get the tags from the dataset
tags = dataset["train"][0]["ner_tags"]
label_list = dataset["train"].features["ner_tags"].feature.names
labels = [label_list[i] for i in tags]


def compute_metrics(evaluation_preds):
    predictions, labels = evaluation_preds
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Create a model for our problem
print(f"Instantiating model ({model_name})...")
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=13, id2label=id2label, label2id=label2id
)

# Define the hyperparameters in the TrainingArguments
print("Creating training arguments (weights are stored at `results/sequence_classification`)...")
training_args = TrainingArguments(
    output_dir="results/token_classification",  # Where weights are stored
    learning_rate=2e-5,  # The learning rate during training
    per_device_train_batch_size=16,  # Number of samples per batch during training
    per_device_eval_batch_size=16,  # Number of samples per batch during evaluation
    num_train_epochs=2,  # How many iterations through the dataloaders should be done
    weight_decay=0.01,  # Regularization penalization
    evaluation_strategy="epoch",  # How often metrics on the evaluation dataset should be computed
    save_strategy="epoch",  # When to try and save the best model (such as a step number or every iteration)
)

# Create the `Trainer`, passing in the model and arguments
# the datasets to train on, how the data should be collated,
# and the method for computing our metrics
print("Creating `Trainer`...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Initiate training
print("Training...")
trainer.train()

# Performing inference
text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
# We need to tokenize the inputs and turn them to PyTorch tensors
encoded_input = tokenizer(text, return_tensors="pt")

# Then we can perform raw torch inference:
print("Performing inference...")
model.eval()
with torch.inference_mode():
    logits = model(**encoded_input).logits

# Finally, decode our outputs
predictions = logits.argmax(dim=2)
print(f"Prediction: {[id2label[pred] for pred in predictions[0]]}")
# Can also use `model.config.id2label` instead
