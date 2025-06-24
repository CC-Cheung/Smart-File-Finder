import json
import pandas as pd
import os
from datasets import Dataset
FILE_PATH=os.path.dirname(os.path.abspath(__file__))

CODE_PATH= os.path.join(FILE_PATH, '..',)
DATA_PATH=os.path.join(CODE_PATH, 'data')
MODELS_PATH=os.path.join(CODE_PATH, 'models')
LOGS_PATH=os.path.join(CODE_PATH, 'logs')

FINETUNED_PATH=os.path.join(MODELS_PATH, 'finetuned')

PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
USED_DATA_PATH=os.path.join(DATA_PATH, 'used')


with open(os.path.join(USED_DATA_PATH, 'used_dataset_sys_use_ass_list.json'), 'r') as f:
    used_dataset = json.load(f)
dataset = Dataset.from_list(used_dataset)  

all_results = []
df=pd.read_csv(os.path.join(LOGS_PATH,'outputs_sys_use_ass_list.csv'))
df2=pd.read_csv(os.path.join(LOGS_PATH,'outputs_sys_use_ass_list_2.csv'))
df3=df.copy()
df3['outputs_2']=df2['outputs']
df3['match_2']=df2['match']
df3['same']=df2['match']==df['match']

pass
# df.to_csv(os.path.join(LOGS_PATH,'outputs.csv'), index=False)

# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
