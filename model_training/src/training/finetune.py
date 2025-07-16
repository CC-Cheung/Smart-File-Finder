import json
import os
import json
# import ollama
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import psutil
from unsloth import get_chat_template
import wandb

FILE_PATH=os.path.dirname(os.path.abspath(__file__))
CODE_PATH= os.path.join(FILE_PATH, '..','..')
DATA_PATH=os.path.join(CODE_PATH, 'data')
MODELS_PATH=os.path.join(CODE_PATH, 'models')
FINETUNED_PATH=os.path.join(MODELS_PATH, 'finetuned')

PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
USED_DATA_PATH=os.path.join(DATA_PATH, 'used')
DATASET_PATH=os.path.join(USED_DATA_PATH, 'used_dataset_SUA_pick_number.json')


MODEL_NAME="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
# MODEL_NAME="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# MODEL_NAME=os.path.join(FINETUNED_PATH, 'mistral_SUA_pick_number2')
SAVE_MODEL_NAME=os.path.join(FINETUNED_PATH, "mistral_SUA_pick_number")
def pre_apply_chat_template(example):  
    conversations = example["text"]  
    input_ids = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=False)  
    return {"input_ids": input_ids}  

if __name__ == "__main__":
    # psutil.virtual_memory().available

    with open(DATASET_PATH, 'r') as f:
        used_dataset = json.load(f)
    dataset = Dataset.from_list(used_dataset)  

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    #IMPORTANT?
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "mistral", # change this to the right chat_template name
    )    

    dataset = dataset.map(pre_apply_chat_template)  

    #Comment out if from my pretrained
    lora_configs = {
        "r": 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        "lora_alpha": 16,
        "lora_dropout": 0, # Supports any, but = 0 is optimized
        "bias": "none",
        "use_rslora": False, 
        "loftq_config": None, 
    }

    model = FastLanguageModel.get_peft_model(
        model, **lora_configs       
    )

    #If wandb
    run=wandb.init(
        project="Smart File Finder",
        name="test",
        tags=[
            MODEL_NAME, 
              "finetune"],
        config=lora_configs
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="input_ids",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            
            output_dir=FINETUNED_PATH,
            report_to = "wandb", 
            logging_steps=1,  
            resume_from_checkpoint=True
            # max_steps=5,
        ),

    )
    trainer.train()
    run.finish()
    torch.cuda.empty_cache()  # Clear any cached memory
    # model.cpu()  # Move model to CPU
    

    # Option 1: Save PEFT adapters separately (keeps original 4-bit base model)
    # model.save_pretrained(os.path.join(FINETUNED_PATH, "lora_adapters"))
    # tokenizer.save_pretrained(os.path.join(FINETUNED_PATH, "lora_adapters"))

    model.save_pretrained(SAVE_MODEL_NAME)
    tokenizer.save_pretrained(SAVE_MODEL_NAME)

    # May not work, look at ollamaing.py to focus on model making
    model.save_pretrained_gguf(
        os.path.join(FINETUNED_PATH, 'test.gguf'),
        tokenizer,
        quantization_method="q4_k_m",
        maximum_memory_usage=0.0001,
    )



