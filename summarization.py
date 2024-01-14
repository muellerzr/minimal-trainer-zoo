# End-to-end script running the Hugging Face Trainer
# for sequence classification. Based on the Tasks documentation
# originally from: https://hf.co/docs/transformers/tasks/sequence_classification
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Constants
model_name = "t5-small"
dataset_name = "billsum"
metric = "rouge"
prefix = "summarize: "

# Load dataset
print(f"Downloading dataset ({dataset_name})")
dataset = load_dataset(dataset_name, split="ca_test")
dataset = dataset.train_test_split(test_size=0.2)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print(f"Tokenizing dataset for {model_name}...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create an efficient collator which dynamically pads
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# Handle computation of our metrics
print(f"Loading metric ({metric})...")
rouge = evaluate.load(metric)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


print(f"Instantiating model ({model_name})...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define the hyperparameters in the TrainingArguments
print("Creating training arguments (weights are stored at `results/sequence_classification`)...")
training_args = Seq2SeqTrainingArguments(
    output_dir="results/summarization",  # Where weights are stored
    learning_rate=2e-5,  # The learning rate during training
    per_device_train_batch_size=8,  # Number of samples per batch during training
    per_device_eval_batch_size=8,  # Number of samples per batch during evaluation
    num_train_epochs=4,  # How many iterations through the dataloaders should be done
    weight_decay=0.01,  # Regularization penalization
    evaluation_strategy="epoch",  # How often metrics on the evaluation dataset should be computed
    save_strategy="epoch",  # When to try and save the best model (such as a step number or every iteration)
    predict_with_generate=True,  # Whether to use `model.generate()` during predictions
)

# Create the `Seq2SeqTrainer`, passing in the model and arguments
# the datasets to train on, how the data should be collated,
# and the method for computing our metrics
print("Creating `Seq2SeqTrainer`...")
trainer = Seq2SeqTrainer(
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
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."  # We need to tokenize the inputs and turn them to PyTorch tensors
encoded_input = tokenizer(text, return_tensors="pt").input_ids

# Then we can perform inference using `model.generate`:
print("Performing inference...")
outputs = model.generate(encoded_input, max_new_tokens=100, do_sample=False)

# Finally, decode our outputs
print(f"Prediction: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
