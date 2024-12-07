import os

DEVICES = ",".join(map(str, range(17, 32)))
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from trl import SFTTrainer
from datasets import load_dataset


model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
)

model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, lora_config)


def get_prompt(x):
    instruction = x['instruction']
    code_input = x['input']
    
    prompt = f"""
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    
    ### Input:
    {code_input}
    
    ### Output:
    """
    
    return {"text": prompt}


dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
dataset = dataset.map(get_prompt)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    formatting_func=lambda x: x['text'],
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args
)


trainer.train()
trainer.save_model("./exp2")
