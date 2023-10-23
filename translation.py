# End-to-end script running the Hugging Face Trainer
# for translation. Based on the Tasks documentation
# originally from: https://hf.co/docs/transformers/tasks/translation
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
dataset_name = "opus_books"
language_set = "en-fr"  # English -> French
source_lang = "en"
target_lang = "fr"
prefix = "translating English to French: "
metric = "sacrebleu"

# Load dataset
print(f"Downloading dataset ({dataset_name})")
dataset = load_dataset(dataset_name, language_set)
dataset = dataset["train"].train_test_split(test_size=0.2)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


print(f"Tokenizing dataset for {model_name}...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create an efficient collator which dynamically pads
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# Handle computation of our metrics
print(f"Loading metric ({metric})...")
sacrebleu = evaluate.load(metric)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


print(f"Instantiating model ({model_name})...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define the hyperparameters in the TrainingArguments
print("Creating training arguments (weights are stored at `results/sequence_classification`)...")
training_args = Seq2SeqTrainingArguments(
    output_dir="results/translation",  # Where weights are stored
    learning_rate=2e-5,  # The learning rate during training
    per_device_train_batch_size=16,  # Number of samples per batch during training
    per_device_eval_batch_size=16,  # Number of samples per batch during evaluation
    num_train_epochs=2,  # How many iterations through the dataloaders should be done
    weight_decay=0.01,  # Regularization penalization
    evaluation_strategy="epoch",  # How often metrics on the evaluation dataset should be computed
    save_strategy="epoch",  # When to try and save the best model (such as a step number or every iteration)
    predict_with_generate=True,  # Whether we should predict using `model.generate()`
)

# Create the `Seq2SeqTrainer`, passing in the model and arguments
# the datasets to train on, how the data should be collated,
# and the method for computing our metrics
print("Creating `Trainer`...")
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
text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
# We need to tokenize the inputs and turn them to PyTorch tensors
encoded_input = tokenizer(text, return_tensors="pt").input_ids

# Then we can perform inference using `model.generate()`:
print("Performing inference...")
outputs = model.generate(encoded_input, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

# Finally, decode our outputs
print(f"Prediction: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
