import ollama
import os
import json
import pandas as pd
FILE_PATH=os.path.dirname(os.path.abspath(__file__))

CODE_PATH= os.path.join(FILE_PATH, '..',)
DATA_PATH=os.path.join(CODE_PATH, 'data')
MODELS_PATH=os.path.join(CODE_PATH, 'models')
FINETUNED_PATH=os.path.join(MODELS_PATH, 'finetuned')

PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
USED_DATA_PATH=os.path.join(DATA_PATH, 'used')


with open(os.path.join(USED_DATA_PATH, 'used_dataset.json'), 'r') as f:
    used_dataset = json.load(f)
{
    "text": "### Instruction: Visible Tree: ~/\n\u251c\u2500\u2500 Documents\n\u2502   \u251c\u2500\u2500 Music\n\u2502   \u2502   \u251c\u2500\u2500 Projects\n\u2502   \u2502   \u2514\u2500\u2500 Software Synths\n\u2502   \u2502       \u2514\u2500\u2500 Plugin Presets\n\u2502   \u2502           \u251c\u2500\u2500 Arturia V Collection\n\u2502   \u2502           \u2514\u2500\u2500 Massive\n\u2502   \u2514\u2500\u2500 Travel\n\u2514\u2500\u2500 Videos\n    \u2514\u2500\u2500 Concert Footage\n\nDescription: Music composition project folder.\n### Output: ~//Documents/Music/Projects"
  },
all_results = []
for i,entry in enumerate(used_dataset):
    input, output = entry['text'].split('\n### Output: ')
    vis_tree, desc = input.split('Description: ')
    vis_tree = vis_tree.replace('### Instruction: Visible Tree: ', '')
    all_results.append({
        'vis_tree': vis_tree,
        'desc': desc,
        'out': output,
        'model_response': ollama.chat(
            model='mymodel',
            messages=[{
                'role': 'user', 
                'content': input
                }]
            )['message']['content']
    })
    if i>0:
        break                                
df=pd.DataFrame(all_results)

print(response['message']['content'])  # Should print the path