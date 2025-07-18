from unsloth import FastLanguageModel
import os
import json
from datasets import Dataset
from transformers import TextStreamer
import re
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
MODEL_NAME="mistral_SUA_pick_number3"

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
    input_ids = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True)  
    return {"input_ids": input_ids}  

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
def model_generate_text_all(data, batch_size=10):
    all_texts=[]
    for i in range(len(data['text'])//batch_size):
        batch_slice=slice(i*batch_size, min((i+1)*batch_size, len(data['text'])))
        inputs=tokenizer(data['text'][batch_slice], return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(model.device)

        outputs = model.generate(inputs['input_ids'], 
                       max_new_tokens = 128, 
                       pad_token_id = tokenizer.eos_token_id,
                       attention_mask = inputs['attention_mask'],
                       )
        all_texts += tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return all_texts
def clean_output(x):
    result = re.search(r'Final Answer: \d+', x)
    if result:
        return result[0]
    else:
        return x
if __name__ == "__main__":
    # psutil.virtual_memory().available

    with open(os.path.join(USED_DATA_PATH, 'used_dataset_SUA_pick_number.json'), 'r') as f:
        used_dataset = json.load(f)
    
    dataset = Dataset.from_list(used_dataset)  

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.path.join(FINETUNED_PATH,  MODEL_NAME),
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

    example_id=None
    example_id=slice(0,100)   

    dataset_gen = dataset.map(pre_apply_chat_template_gen)
    inputs=dataset_gen[example_id]
    

    #one or a few
    # outputs = model_generate_text(inputs)
    # correct_outputs = [example[2]['content'] for example in dataset[example_id]['text']]
    # df=pd.DataFrame({'outputs': outputs, 'correct_outputs': correct_outputs})
    # print(df)

    #all 
    outputs = model_generate_text_all(inputs)
    correct_outputs = [example[2]['content'] for example in dataset[example_id]['text']]
    # example_id=slice(0,df.shape[0])    
    

    split_user_prompts=[example[1]['content'].split('\nHere are the visible items. Choose one of the following: ') for example in dataset[example_id]['text']]  

    df=pd.DataFrame({'outputs': outputs, 
                     'correct_outputs': correct_outputs, 
                     'file_system': [user_prompt[0] for user_prompt in split_user_prompts], 
                     'description': [user_prompt[1] for user_prompt in split_user_prompts]})
    df['outputs']=df['outputs'].map(clean_output) 
    df['match']=df['correct_outputs']==df['outputs']
    print(df)
    df.to_csv(os.path.join(LOGS_PATH,'outputs_'+MODEL_NAME+'.csv'), index=False)

    # df=pd.read_csv(os.path.join(LOGS_PATH,'outputs.csv'))
    # df2=pd.read_csv(os.path.join(LOGS_PATH,'outputs_no_train.csv'))

    pass
    # df.to_csv(os.path.join(LOGS_PATH,'outputs.csv'), index=False)

    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df.to_csv(os.path.join(LOGS_PATH,'outputs.csv'), index=False)
    