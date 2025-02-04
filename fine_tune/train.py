from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass
import os, argparse
import torch

@dataclass
class ModelArguments:
    model_name: str
    bucket_name: str

parser = HfArgumentParser((ModelArguments, TrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()
s3_path = f's3://{model_args.bucket_name}/datasets/formatted_dataset.json'
dataset = load_dataset('json', data_files=s3_path, split='train')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_args.model_name,
    max_seq_length=2048,
    load_in_4bit=True,
)

peft_config = {
    "r": 64,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3927
}

model = FastLanguageModel.get_peft_model(model, **peft_config)
training_args = TrainingArguments(
    output_dir=training_args.output_dir,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    learning_rate=training_args.learning_rate,
    max_steps=training_args.max_steps,
    fp16=True,
    optim="adamw_torch", 
    logging_steps=1
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=training_args
)
trainer.train()