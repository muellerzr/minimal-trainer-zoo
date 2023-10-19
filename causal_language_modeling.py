# End-to-end script running the Hugging Face Trainer 
# for causal language modeling. Based on the Tasks documentation 
# originally from: https://hf.co/docs/transformers/tasks/language_modeling
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)

# Constants
model_name = "distilgpt2"
dataset_name = "eli5"

# Load dataset
print(f"Downloading dataset ({dataset_name})")
dataset = load_dataset(dataset_name, split="train_asks[:5000]")
dataset = dataset.train_test_split(test_size=0.2)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

print(f"Tokenizing dataset for {model_name}...")
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=dataset["train"].column_names
)

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

print(f'Instantiating model ({model_name})...')
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the hyperparameters in the TrainingArguments
print(f'Creating training arguments (weights are stored at `results/causal_language_modeling`)...')
training_args = TrainingArguments(
    output_dir="results/causal_language_modeling", # Where weights are stored
    learning_rate=2e-5, # The learning rate during training
    per_device_train_batch_size=8, # Number of samples per batch during training
    per_device_eval_batch_size=8, # Number of samples per batch during evaluation
    num_train_epochs=2, # How many iterations through the dataloaders should be done
    weight_decay=0.01, # Regularization penalization
    evaluation_strategy="epoch", # How often metrics on the evaluation dataset should be computed
    save_strategy="epoch", # When to try and save the best model (such as a step number or every iteration)
)

# Create the `Trainer`, passing in the model and arguments
# the datasets to train on, how the data should be collated, 
# and the method for computing our metrics
print(f'Creating `Trainer`...')
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

# Performing inference
text = "Somatic hypermutation allows the immune system to"
# We need to tokenize the inputs and turn them to PyTorch tensors
encoded_input = tokenizer(text, return_tensors="pt").input_ids

# Then we can perform inference via `model.generate`:
print("Performing inference...")
outputs = model.generate(
    encoded_input, 
    max_new_tokens=100, 
    do_sample=True, 
    top_k=50, 
    top_p=0.95
)
    
# Finally, decode our outputs
print(f'Prediction: {tokenizer.batch_decode(outputs, skip_special_tokens=True)}')