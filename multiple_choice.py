# End-to-end script running the Hugging Face Trainer
# for multiple choice. Based on the Tasks documentation
# originally from: https://hf.co/docs/transformers/tasks/multiple_choice
from dataclasses import dataclass
from typing import Optional, Union

import evaluate
import numpy as np
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForMultipleChoice, AutoTokenizer, Trainer, TrainingArguments
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

# Constants
model_name = "bert-base-uncased"
dataset_name = "swag"
metric = "accuracy"

# Load dataset
print(f"Downloading dataset ({dataset_name})")
dataset = load_dataset(dataset_name, "regular", split="train[:8%]", trust_remote_code=True)
dataset = dataset.train_test_split(test_size=0.2)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
ending_names = ["ending0", "ending1", "ending2", "ending3"]


def tokenize_function(examples):
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


print(f"Tokenizing dataset for {model_name}...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Create our own data collator class and use it
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

# Handle computation of our metrics
print(f"Loading metric ({metric})...")
accuracy = evaluate.load(metric)


def compute_metrics(evaluation_preds):
    predictions, labels = evaluation_preds
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


print(f"Instantiating model ({model_name})...")
model = AutoModelForMultipleChoice.from_pretrained(model_name)

# Define the hyperparameters in the TrainingArguments
print("Creating training arguments (weights are stored at `results/multiple_choice`)...")
training_args = TrainingArguments(
    output_dir="results/multiple_choice",  # Where weights are stored
    learning_rate=5e-5,  # The learning rate during training
    per_device_train_batch_size=32,  # Number of samples per batch during training
    per_device_eval_batch_size=32,  # Number of samples per batch during evaluation
    num_train_epochs=2,  # How many iterations through the dataloaders should be done
    weight_decay=0.01,  # Regularization penalization
    evaluation_strategy="epoch",  # How often metrics on the evaluation dataset should be computed
    save_strategy="epoch",  # When to try and save the best model (such as a step number or every iteration)
    fp16=True,  # Whether to use 16-bit precision (mixed precision) instead of 32-bit. Generally faster on T4's
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
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Initiate training
print("Training...")
trainer.train()

# Performing inference
prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law does not apply to croissants and brioche."
candidate2 = "The law applies to baguettes."
# We need to tokenize the inputs and turn them to PyTorch tensors
encoded_input = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
encoded_input = {k: v.unsqueeze(0) for k, v in encoded_input.items()}
labels = torch.tensor(0).unsqueeze(0)

# To move the batch to the right device automatically, use `PartialState().device`
# which will always work no matter the environment
device = PartialState().device
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
# Can also be `.to("cuda")`
labels = labels.to(device)

# Then we can perform raw torch inference:
print("Performing inference...")
model.eval()
with torch.inference_mode():
    logits = model(**encoded_input, labels=labels).logits

# Finally, decode our outputs
predicted_class = logits.argmax().item()
print(f"Predicted answer number: {predicted_class}")
