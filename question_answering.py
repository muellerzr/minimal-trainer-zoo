# End-to-end script running the Hugging Face Trainer 
# for sequence classification. Based on the Tasks documentation 
# originally from: https://hf.co/docs/transformers/tasks/sequence_classification
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator,
    TrainingArguments, Trainer
)

# Constants
model_name = "distilbert-base-uncased"
dataset_name = "squad"

# Load a subset of the dataset
print(f"Downloading dataset ({dataset_name})")
dataset = load_dataset(dataset_name, split="train[:5000]")
dataset = dataset.train_test_split(test_size=0.2)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

print(f"Tokenizing dataset for {model_name}...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Use a basic collator with no preprocessing (like padding)
data_collator = DefaultDataCollator()

print(f'Instantiating model ({model_name})...')
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Define the hyperparameters in the TrainingArguments
print(f'Creating training arguments (weights are stored at `results/sequence_classification`)...')
training_args = TrainingArguments(
    output_dir="results/question_answering", # Where weights are stored
    learning_rate=2e-5, # The learning rate during training
    per_device_train_batch_size=16, # Number of samples per batch during training
    per_device_eval_batch_size=16, # Number of samples per batch during evaluation
    num_train_epochs=3, # How many iterations through the dataloaders should be done
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
    tokenizer=tokenizer,
)

# Initiate training
print("Training...")
trainer.train()

# Performing inference
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
# We need to tokenize the inputs and turn them to PyTorch tensors
encoded_input = tokenizer(question, context, return_tensors="pt")

# Then we can perform raw torch inference:
print("Performing inference...")
model.eval()
with torch.inference_mode():
    outputs = model(**encoded_input)
    
# Finally, decode our outputs
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
predicted_answer_tokens = encoded_input.input_ids[0, answer_start_index : answer_end_index + 1]
print(f'Prediction: {tokenizer.decode(predicted_answer_tokens)}')
