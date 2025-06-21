import json
import os
import re
import json
import random
import re
from copy import deepcopy
# import ollama
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import psutil
from unsloth import get_chat_template
# import wandb

FILE_PATH=os.path.dirname(os.path.abspath(__file__))
CODE_PATH= os.path.join(FILE_PATH, '..','..')
DATA_PATH=os.path.join(CODE_PATH, 'data')
MODELS_PATH=os.path.join(CODE_PATH, 'models')
FINETUNED_PATH=os.path.join(MODELS_PATH, 'finetuned')

PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
USED_DATA_PATH=os.path.join(DATA_PATH, 'used')

MODEL_NAME="unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MODEL_NAME="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

def pre_apply_chat_template(example):  
    conversations = example["text"]  
    text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)  
    return {"text": text}  

if __name__ == "__main__":
    # psutil.virtual_memory().available

    # with open(os.path.join(USED_DATA_PATH, 'used_dataset_sys_use_ass.json'), 'r') as f:
    #     used_dataset = json.load(f)
    
    # dataset = Dataset.from_list(used_dataset)  

    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=MODEL_NAME,
    #     max_seq_length=2048,
    #     load_in_4bit=True,
    #     # device_map="cpu",  # Force CPU usage
    # )
    # #OR ELSE EXTRA <s>
    # tokenizer.add_bos_token = False
    # dataset = dataset.map(pre_apply_chat_template)  

    # lora_configs = {
    #     "r": 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    #     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj",],
    #     "lora_alpha": 16,
    #     "lora_dropout": 0, # Supports any, but = 0 is optimized
    #     "bias": "none",
    #     "use_rslora": False, 
    #     "loftq_config": None, 
    # }
    # model = FastLanguageModel.get_peft_model(
    #     model, **lora_configs
    #     # r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    #     # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #     #                 "gate_proj", "up_proj", "down_proj",],
    #     # lora_alpha = 16,
    #     # lora_dropout = 0, # Supports any, but = 0 is optimized
    #     # bias = "none",    # Supports any, but = "none" is optimized
    #     # # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    #     # use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    #     # random_state = 3407,
    #     # use_rslora = False,  # We support rank stabilized LoRA
    #     # loftq_config = None, # And LoftQ
    # )
    # # wandb.login(key="WANDB_API_KEY")
    
    # run=wandb.init(
    #     project="Smart File Finder",
    #     # entity="my_entity",
    #     name="test",
    #     tags=[MODEL_NAME, "finetune"],
    #     config=lora_configs
    # )

    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset,
    #     tokenizer=tokenizer,
    #     dataset_text_field="text",
    #     max_seq_length=2048,
    #     args=TrainingArguments(
    #         per_device_train_batch_size=2,
    #         gradient_accumulation_steps=4,
    #         learning_rate=2e-5,
    #         output_dir=FINETUNED_PATH,
    #         # report_to = "WandB", # Use this for WandB etc

    #     ),

    # )
    # trainer.train()
    # run.finish()
    # torch.cuda.empty_cache()  # Clear any cached memory
    # # model.cpu()  # Move model to CPU
    

    # # Option 1: Save PEFT adapters separately (keeps original 4-bit base model)
    # # model.save_pretrained(os.path.join(FINETUNED_PATH, "lora_adapters"))
    # # tokenizer.save_pretrained(os.path.join(FINETUNED_PATH, "lora_adapters"))

    # model.save_pretrained(os.path.join(FINETUNED_PATH, "test"))
    # tokenizer.save_pretrained(os.path.join(FINETUNED_PATH, "test"))

    #no memory or tuple indices must be integers or slices, not NoneType
    # model.save_pretrained_merged(
    #     os.path.join(FINETUNED_PATH, 'test_merged'), 
    #     tokenizer, 
    #     save_method = "merged_16bit",
    #     maximum_memory_usage=0.01  # Use very little memory     
    # )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.path.join(FINETUNED_PATH, 'latest'),
        max_seq_length=2048,
        load_in_4bit=True,
        # device_map="cpu",  # Force CPU usage
    )

    model.save_pretrained_gguf(
        os.path.join(FINETUNED_PATH, 'test.gguf'),
        tokenizer,
        quantization_method="q4_k_m",
        maximum_memory_usage=0.0001,
    )



