from unsloth import FastLanguageModel
import os
import json
from datasets import Dataset
from transformers import TextStreamer
import torch
import pandas as pd
FILE_PATH=os.path.dirname(os.path.abspath(__file__))
CODE_PATH= os.path.join(FILE_PATH, '..')
DATA_PATH=os.path.join(CODE_PATH, 'data')
LOGS_PATH=os.path.join(CODE_PATH, 'logs')

MODELS_PATH=os.path.join(CODE_PATH, 'models')
FINETUNED_PATH=os.path.join(MODELS_PATH, 'finetuned')

PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
USED_DATA_PATH=os.path.join(DATA_PATH, 'used')


def pre_apply_chat_template(example):  
    conversations = example["text"]  
    text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)  
    return {"text": text}  
def pre_apply_chat_template_gen(example):  
    conversations = example["text"][:-1]  
    text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)  
    return {"text": text}  
def pre_apply_chat_template_gen_tokenize(example):  
    conversations = example["text"][:-1]  
    text = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True)  
    return {"text": text}  

def model_text_stream(inputs):
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(inputs['input_ids'], 
                       attention_mask = inputs['attention_mask'], 
                       max_new_tokens = 128, 
                       pad_token_id = tokenizer.eos_token_id,
                       streamer=text_streamer)

    return _
def model_generate_text(inputs):
    outputs = model.generate(inputs['input_ids'], 
                       attention_mask = inputs['attention_mask'], 
                       max_new_tokens = 128, 
                       pad_token_id = tokenizer.eos_token_id,
                       )
    text = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    return text
def model_generate_text_all(inputs, batch_size=10):
    all_texts=[]
    for i in range(inputs['input_ids'].shape[0]//batch_size):
        batch_slice=slice(i*batch_size, min((i+1)*batch_size, inputs['input_ids'].shape[0]))
        outputs = model.generate(inputs['input_ids'][batch_slice], 
                       attention_mask = inputs['attention_mask'][batch_slice], 
                       max_new_tokens = 128, 
                       pad_token_id = tokenizer.eos_token_id,
                       )
        all_texts += tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return all_texts
if __name__ == "__main__":
    # psutil.virtual_memory().available

    with open(os.path.join(USED_DATA_PATH, 'used_dataset_sys_use_ass_list.json'), 'r') as f:
        used_dataset = json.load(f)
    
    dataset = Dataset.from_list(used_dataset)  

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/home/kids/Linux_Coding/Smart-File-Finder/models/finetuned/sys_use_ass_list",
        max_seq_length=2048,
        load_in_4bit=True,
        # device_map="cpu",  # Force CPU usage
    )
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    #     max_seq_length=2048,
    #     load_in_4bit=True,
    #     # device_map="cpu",  # Force CPU usage
    # )
    FastLanguageModel.for_inference(model) 

    # Apply the chat template to the dataset
    dataset_gen = dataset.map(pre_apply_chat_template_gen)

    example_id=None
    example_id=slice(0,100)    
    inputs = tokenizer(
        dataset_gen['text'] if example_id is None else dataset_gen['text'][example_id],
        return_tensors="pt",
        # add_special_tokens=False, #check if correct?
        padding=True,
    ).to("cuda") 

    #one or a few
    # outputs = model_generate_text(inputs)
    # correct_outputs = [example[2]['content'] for example in dataset[example_id]['text']]
    # df=pd.DataFrame({'outputs': outputs, 'correct_outputs': correct_outputs})
    # print(df)

    #all 
    outputs = model_generate_text_all(inputs)
    correct_outputs = [example[2]['content'] for example in dataset[example_id]['text']]
    # example_id=slice(0,df.shape[0])    
    
    split_user_prompts=[example[1]['content'].split('Pick from one of the following: ') for example in dataset[example_id]['text']]    
    df=pd.DataFrame({'outputs': outputs, 
                     'correct_outputs': correct_outputs, 
                     'file_system': [user_prompt[0] for user_prompt in split_user_prompts], 
                     'description': [user_prompt[1] for user_prompt in split_user_prompts]})
    df['match']=df['correct_outputs']==df['outputs']

    print(df)
    df.to_csv(os.path.join(LOGS_PATH,'outputs_sys_use_ass_list.csv'), index=False)

    # df=pd.read_csv(os.path.join(LOGS_PATH,'outputs.csv'))
    # df2=pd.read_csv(os.path.join(LOGS_PATH,'outputs_no_train.csv'))

    pass
    # df.to_csv(os.path.join(LOGS_PATH,'outputs.csv'), index=False)

    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df.to_csv(os.path.join(LOGS_PATH,'outputs.csv'), index=False)
    