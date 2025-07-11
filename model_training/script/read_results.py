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
    final_line=row['description'].splitlines()[-1]
    largest_number = int(re.match(r'\d+', final_line)[0])
    try:
        choice = int(re.match(r'\d+', row['outputs'].lstrip('Final Answer: '))[0])
    except:
        return None
    # Extract final answer from output
    return largest_number>=choice
    # Check if final answer is in choices
def clean_output(x):
    result = re.search(r'Final Answer: \d+', x, re.IGNORECASE)
    if result:
        return result[0]
    else:
        return x
    return result.group(0)
# all_results = []
df=pd.read_csv(os.path.join(LOGS_PATH,'outputs_mistral_SUA_pick_number.csv'))
df['outputs']=df['outputs'].map(clean_output) 

# df['match'] = df['outputs'] == df['correct_outputs']
df2=pd.read_csv(os.path.join(LOGS_PATH,'outputs_mistral_SUA_pick_number2.csv'))
df2['outputs']=df2['outputs'].map(clean_output)

df3=pd.read_csv(os.path.join(LOGS_PATH,'outputs_mistral_SUA_pick_number3.csv'))
df3['outputs']=df2['outputs'].map(clean_output)

df['is_valid'] = df.apply(is_final_answer_in_choices, axis=1)
df2['is_valid'] = df2.apply(is_final_answer_in_choices, axis=1)
df3['is_valid'] = df3.apply(is_final_answer_in_choices, axis=1)

df4=df.copy()
df4['outputs_2']=df2['outputs']
df4['match_2']=df2['match']
df4['is_valid_2']=df2['is_valid']
df4['outputs_3']=df3['outputs']
df4['match_3']=df3['match']
df4['is_valid_3']=df3['is_valid']

# df4['same']=df2['match']==df['match']


pass
# df.to_csv(os.path.join(LOGS_PATH,'outputs_mistral_SUA_pick_number.csv'), index=False)

# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
