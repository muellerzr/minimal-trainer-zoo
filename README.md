# minimal-trainer-zoo
Minimal example scripts of the Hugging Face Trainer, focused on staying under 150 lines. (and still be readable).

All examples are based on the Tasks documentation from `transformers`

TODO: Link to task docs in readme

## What is this repository?

this collection contains a variety of task scrips for Hugging Face `transformers`, but kept to a minimal level
to enhance readability, hackability, and for learning. 

## How to Run

Each script is self-contained and requires no arguments. Simply:

1. `git clone https://github.com/muellerzr/minimal-trainer-zoo`
2. `cd minimal-trainer-zoo; pip install -r requirements.txt`
3. `python {script.py}` (or `torchrun`/`accelerate launch` instead of `python` if wanting to use DDP)