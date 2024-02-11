import torch
import os, sys
import numpy as np

sys.path.append("../..")
import experiments.utils as utils
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from functools import partial
from datasets import Dataset

model_name = "NousResearch/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


data_path = os.path.join(
    os.path.dirname(__file__), "../../data/gbpcad_one_hour_202311210827.csv"
)

torch.manual_seed(42)
num_lags = 5

data = utils.get_data(
    data_path,
    num_lags,
    date_column="barTimestamp",
    price_column="close",
    date_format="%Y-%m-%d %H:%M:%S",
)
data = data.iloc[:20]  # TODO change size
data.drop(["id", "provider", "insertTimestamp", "dayOfWeek"], axis=1, inplace=True)

data = data.dropna()
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
data[cols] = minmax_scale(data[cols])

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction: {instruction}\n Response:"""
answer_template = """{response}"""


def _add_text(rec):
    instruction = str(rec)  # TODO prompt
    response = str(rec["price_delta"])
    rec["prompt"] = prompt_template.format(instruction=instruction)
    rec["answer"] = answer_template.format(response=response)
    rec["text"] = rec["prompt"] + rec["answer"]
    return rec


def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        padding="max_length",
    )


max_length = 1024

data = Dataset.from_pandas(data)
data = data.map(_add_text)


_preprocessing_function = partial(
    preprocess_batch, max_length=max_length, tokenizer=tokenizer
)
data = data.map(
    _preprocessing_function,
    batched=True,
    remove_columns=[
        "barTimestamp",
        "open",
        "close",
        "low",
        "high",
        "volume",
        "ask_open",
        "ask_close",
        "ask_low",
        "ask_high",
        "ask_volume",
        "usdPerPips",
        "spread",
        "close_lag_1",
        "close_lag_2",
        "close_lag_3",
        "close_lag_4",
        "close_lag_5",
        "price_delta",
        "__index_level_0__",
        "prompt",
        "answer",
        "text",
    ],
)

print(data)
split_dataset = data.train_test_split(train_size=0.4)

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
lora_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)  # TODO target modules
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

output_dir = "./results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 2
optim = "paged_adamw_32bit"
save_steps = 1
num_train_epochs = 4
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 20
warmup_ratio = 0.03
lr_scheduler_type = "linear"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_arguments,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
)
trainer.train()

# TODO save
