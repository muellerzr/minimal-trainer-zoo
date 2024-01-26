# End-to-end script running the Hugging Face Trainer
# for masked language modeling. Based on the Tasks documentation
# originally from: https://hf.co/docs/transformers/tasks/masked_language_modeling
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Constants
model_name = "distilroberta-base"
dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"

# Load dataset
print(f"Downloading dataset ({dataset_name})")
dataset = load_dataset(dataset_name, dataset_config, split="train[:500]")
dataset = dataset.train_test_split(test_size=0.2)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"])


print(f"Tokenizing dataset for {model_name}...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# We still need to concatenate our sequences
# and split them into shorter chunks to ease
# minimal RAM usage
block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# And apply
tokenized_dataset = tokenized_dataset.map(group_texts, batched=True)

# Create an efficient collator which dynamically pads
# End-of-sequence as the padding token and mlm=False will
# use the inputs as labels, shifted to the right by one element
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

print(f"Instantiating model ({model_name})...")
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Define the hyperparameters in the TrainingArguments
print("Creating training arguments (weights are stored at `results/causal_language_modeling`)...")
training_args = TrainingArguments(
    output_dir="results/masked_language_modeling",  # Where weights are stored
    learning_rate=2e-5,  # The learning rate during training
    per_device_train_batch_size=8,  # Number of samples per batch during training
    per_device_eval_batch_size=8,  # Number of samples per batch during evaluation
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
)

# Initiate training
print("Training...")
trainer.train()

# Use the evaluate() method to evaluate the model and get its perplexity:
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# Performing inference
text = "The Milky Way is a <mask> galaxy."
# We need to tokenize the inputs and turn them to PyTorch tensors
encoded_input = tokenizer(text, return_tensors="pt")

# To move the batch to the right device automatically, use `PartialState().device`
# which will always work no matter the environment
encoded_input = encoded_input.to(PartialState().device)
# Can also be `encoded_input.to("cuda")`

mask_token_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]

# Then we can perform inference via `model.generate`:
print("Performing inference...")
with torch.inference_mode():
    logits = model(**encoded_input).logits
mask_token_logits = logits[0, mask_token_index, :]

# Finally, decode our outputs
top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()
print("Predictions:")

for token in top_3_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
