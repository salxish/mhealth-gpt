import os
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling, training_args
)

os.environ["HF_HUB_OFFLINE"] = "1"
accuracy = evaluate.load("accuracy")

hf_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
raw_dataset = load_dataset("gjyotk/Menstrual-Health-Awareness-Dataset", split="train")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(hf_tokenizer))
model.config.pad_token_id = hf_tokenizer.pad_token_id

def tokenize(example):
    combined_text = [
        f"### Instruction:\n{inst}\n\n### Response:\n{out}{hf_tokenizer.eos_token}"
        for inst, out in zip(example["instruction (string)"], example["output (string)"])
    ]
    return hf_tokenizer(
        combined_text,
        truncation=True,
        max_length=512,
    )

tokenized_dataset = raw_dataset.map(tokenize, batched = True, remove_columns = raw_dataset.column_names)
processed_dataset = tokenized_dataset.train_test_split(test_size=0.2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis = -1)
    return accuracy.compute(
        predictions = predictions,
        references = labels
    )

data_collator = DataCollatorForLanguageModeling(tokenizer = hf_tokenizer, mlm = False)

# noinspection PyRedeclaration
training_args = TrainingArguments(
    output_dir = "distilgpt2-medical",
    learning_rate = 5e-5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    num_train_epochs = 4,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    push_to_hub = True,
)

# noinspection PyTypeChecker
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = processed_dataset["train"],
    eval_dataset = processed_dataset["test"],
    data_collator = data_collator,
)

trainer.train()
trainer.push_to_hub()
