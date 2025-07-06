import json
import pandas as pd
import os
from datasets import Dataset
import re
FILE_PATH=os.path.dirname(os.path.abspath(__file__))

CODE_PATH= os.path.join(FILE_PATH, '..',)
DATA_PATH=os.path.join(CODE_PATH, 'data')
MODELS_PATH=os.path.join(CODE_PATH, 'models')
LOGS_PATH=os.path.join(CODE_PATH, 'logs')

FINETUNED_PATH=os.path.join(MODELS_PATH, 'finetuned')

PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
USED_DATA_PATH=os.path.join(DATA_PATH, 'used')


# with open(os.path.join(USED_DATA_PATH, 'used_dataset_SUA_number_list.json'), 'r') as f:
#     used_dataset = json.load(f)
# dataset = Dataset.from_list(used_dataset)  
# df=pd.read_csv(os.path.join(LOGS_PATH,'outputs_SUA_number_list.csv'))

# correct_outputs = [example[2]['content'] for example in dataset[example_id]['text']]
# # example_id=slice(0,df.shape[0])    

# split_user_prompts=[example[1]['content'].split('\nHere are the visible items. Choose one of the following: ') for example in dataset[example_id]['text']]  

# df['file_system']=[user_prompt[0] for user_prompt in split_user_prompts]
# df['correct_outputs']=correct_outputs
# df['description']= [user_prompt[1] for user_prompt in split_user_prompts]
# df['match']=df['correct_outputs']==df['outputs']
# df.to_csv(os.path.join(LOGS_PATH,'outputs_SUA_number_list.csv'), index=False)

def is_final_answer_in_choices(row):
    # Extract choices from prompt
    choices = re.findall(r'\d+\.\s*(.+)', row['description'])
    choices = [choice.strip() for choice in choices]
    # Extract final answer from output
    answer = row['outputs'].lstrip('Final Answer: ')
    return answer in choices
    # Check if final answer is in choices

# all_results = []
df=pd.read_csv(os.path.join(LOGS_PATH,'outputs_SUA_number_list2.csv'))
df2=pd.read_csv(os.path.join(LOGS_PATH,'outputs_SUA_number_list3.csv'))
df3=df.copy()
df3['outputs_2']=df2['outputs']
df3['match_2']=df2['match']
df3['same']=df2['match']==df['match']

df3['is_valid'] = df.apply(is_final_answer_in_choices, axis=1)
df3['is_valid2'] = df2.apply(is_final_answer_in_choices, axis=1)
pass
# df.to_csv(os.path.join(LOGS_PATH,'outputs.csv'), index=False)

# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
