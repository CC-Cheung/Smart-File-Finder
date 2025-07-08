import json
import os
import json
import random
from treelib import Tree
import re
from copy import deepcopy
import ollama
import numpy as np
import pandas as pd
FILE_PATH=os.path.dirname(os.path.abspath(__file__))
CODE_PATH= os.path.join(FILE_PATH, '..', '..')
DATA_PATH=os.path.join(CODE_PATH, 'data')
PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
USED_DATA_PATH=os.path.join(DATA_PATH, 'used')

MODEL = 'dolphin3:latest'   # Name of your local model (verify with `ollama list`)
NUM_DESIRED = 10
DESCRIPTION_PROMPT_TEMPLATE = """Create precise and concise description of the following file/folder in less than 10 words
{ff}"""
PERCENTAGE = 0.5
NEW_TREE_DENSITY = 1.2
class ParenthesesNotAllowedError(Exception):
    pass
class NoRootError(Exception):
    pass
def generate_description(ff):       
    # Generate file tree
    description_response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": DESCRIPTION_PROMPT_TEMPLATE.format(ff=ff)}]
    )
    description = description_response['message']['content'].strip()    
    return description
#unused
def parse_tree_string(tree_str):
    tree = Tree()
    lines = tree_str.splitlines()
    stack = []

    # Extract and create the root node
    root_name = lines[0].strip()
    tree.create_node(root_name, root_name)
    stack.append((root_name, 0))

    for line in lines[1:]:
        # Remove leading tree characters and get the node name
        stripped_line = line.lstrip('│ ').replace('├── ', '').replace('└── ', '')
        # Each level is 4 characters (│   or spaces)
        level = (len(line) - len(line.lstrip('│ '))) // 4 + 1
        node_name = stripped_line.strip()

        # Pop stack to find correct parent for this level
        while stack and stack[-1][1] >= level:
            stack.pop()

        parent_id = stack[-1][0] if stack else root_name
        node_id = f'{parent_id}/{node_name}'
        tree.create_node(node_name, node_id, parent=parent_id)
        stack.append((node_id, level))

    return tree
def extract_paths(tree_str):
    lines = tree_str.splitlines()
    paths = []
    stack = []

    # Handle root node
    root = lines[0].strip()
    stack = [root]
    paths.append(root)

    for line in lines[1:]:
        # Count the number of leading tree characters to determine depth
        stripped = line.lstrip('│ ')
        # Each level is 4 spaces (or '│   ')
        indent = (len(line) - len(stripped)) // 4 + 1
        name = stripped.replace('├── ', '').replace('└── ', '').strip()

        # Adjust stack to current depth
        stack = stack[:indent]
        stack.append(name)
        # Join stack to make the path
        paths.append('/'.join(stack))

    return paths


def parse_tab_tree(tree_str):
    """Parse tab-indented file system structure into treelib Tree"""
    tree = Tree()    
    lines = [line.rstrip() for line in tree_str.split('\n') 
            if line.strip() and not(
                ('START' in line) or 
                ('END' in line) or 
                (len(line)<2 and not '~' in line)
                )
            ]
    
    for line in lines:
        if '(' in line or ')' in line:  # Check for parentheses in tree_str 
            raise ParenthesesNotAllowedError("Parentheses not allowed in tree_str")    

    # Create root node
    home_idx=0
    if len(lines[0]) - len(lines[0].lstrip()) !=0: 
        raise NoRootError("There must be a root folder")    

    for line in lines[1:]:
        cur_indent = len(line) - len(line.lstrip())
        if cur_indent==0:
            home_idx+=1
        else:
            break
    root_name = lines[home_idx].strip()
    tree.create_node(root_name, root_name)
    parent_stack = pd.DataFrame.from_dict({'node_id':[root_name], 'indent': [0]})  # (parent_id, depth, indent)
    

    # Idea is first use indent to find depth, then use depth to find parent (stack)
    for line in lines[home_idx+1:]:
        cur_indent = len(line) - len(line.lstrip())
        prev_indent=parent_stack['indent'].iloc[-1]
        
        name = line.strip() 

        if (cur_indent> prev_indent):
            parent=parent_stack.iloc[-1]
            cur_depth = parent['indent']+1
            
            
        elif (cur_indent== prev_indent):
            parent=parent_stack.iloc[-2]            
            cur_depth = parent['indent']+1
            
        else:
            cur_depth = parent_stack['indent'].searchsorted(cur_indent, side='left')     
            parent_stack = parent_stack.iloc[:cur_depth] 
            parent=parent_stack.iloc[-1]
            
            
        parent_id = parent['node_id']
        node_id = f"{parent_id}/{name}" if parent_id else name
        tree.create_node(name, node_id, parent=parent_id)
        parent_stack.loc[cur_depth] = {'node_id':node_id, 'indent':cur_indent}

    return tree
#unused
def random_tree_subset(complete_tree, percentage=0.5):
    """Creates a random subset of the tree with approximately `num_nodes` nodes"""
    nodes = complete_tree.all_nodes()
    nodes=[i for i in nodes if i.identifier!='~' and not i.is_leaf()]
    complete_tree_size = complete_tree.size()
    random.shuffle(nodes)
    # subset_tree = Tree(complete_tree.subtree(complete_tree.root), deep=True)
    subset_tree=deepcopy(complete_tree)
    for node in nodes:
        try:
            for child in node.successors(subset_tree.identifier):
                subset_tree.remove_node(child)
        except:
            pass
        if subset_tree.size() < complete_tree_size*percentage:
            break
    return subset_tree

def random_tree_subset_ratio(complete_tree, percentage=0.7, new_tree_density=0.7):
    """Creates a random subset of the tree with approximately `num_nodes` nodes"""
    nodes = complete_tree.all_nodes()
    nodes=[(node, complete_tree.depth(node.identifier)) for node in nodes if node.identifier!='~' and not node.is_leaf()]
    df_nodes=pd.DataFrame(nodes, columns=['node', 'depth'])
    df_nodes['weight']=new_tree_density**df_nodes['depth']   
    df_nodes['weight']=df_nodes['weight']/df_nodes['weight'].sum()
    # subset_tree = Tree(complete_tree.subtree(complete_tree.root), deep=True)
    subset_tree=deepcopy(complete_tree)
    nodes=np.random.choice(df_nodes['node'], size=len(df_nodes), p=df_nodes['weight'], replace=False)
    for node in nodes:
        try:
            for child in node.successors(subset_tree.identifier):
                subset_tree.remove_node(child)
        except:
            pass
        if subset_tree.size() < complete_tree.size()*percentage:
            break
    return subset_tree

def generate_visibility(complete_tree, desired):
    """
    Generates dataset of file system visibility levels given the complete tree and desired node:
   
    Args:
        complete_tree (Tree): Full file system structure
        desired (str): Name of target node to exclude
    
    Returns:
        {visible_tree, desired, deepest_folder}
    """

    visible_tree=random_tree_subset(complete_tree=complete_tree, percentage=0.5)
    # visible_tree=random_tree_subset_ratio(complete_tree=complete_tree, percentage=PERCENTAGE, new_tree_density=NEW_TREE_DENSITY)
    
    #remove desired (OLD)
    # if desired in visible_tree.nodes:
    #     for sibling in visible_tree.siblings(desired):
    #         visible_tree.remove_node(sibling.identifier)
    #     visible_tree.remove_node(desired)
    if desired in visible_tree.nodes:
        deepest_visible=desired
    else: 
        deepest_visible = get_deepest_visible(complete_tree=complete_tree, visible_tree=visible_tree, desired=desired)    
    
    return {
        'visible_tree': str(visible_tree), #visible_tree,
        'desired_description': generate_description(desired),
        # 'desired_description': '',

        'desired_path': desired,
        'deepest_folder': deepest_visible
    }
def get_deepest_visible(complete_tree, visible_tree, desired):
    not_in_visible=[i for i in complete_tree.rsearch(desired, filter=lambda node: node.identifier not in visible_tree.nodes)]
   
    return not_in_visible[-1][:not_in_visible[-1].rfind('/')]

def generate_visibility_data(complete_tree, num_desired=3):
    #don't want ~ removed
    all_desired= random.sample(list(complete_tree.nodes)[1:],num_desired)
    visibility_data=[generate_visibility(complete_tree=complete_tree, desired=desired) for desired in all_desired]
    return visibility_data

def generate_dataset(file_system_dataset, num_desired=3):
    dataset=[]
    for i,example in enumerate(file_system_dataset):        
        try:
            complete_tree=parse_tab_tree(example['tree'])
            visibility_data=generate_visibility_data(complete_tree=complete_tree, num_desired=num_desired)
            dataset.append({
                # 'persona': example['persona'],
                'complete_tree': str(complete_tree),
                'visibility_data': visibility_data
            })
        except ParenthesesNotAllowedError as e:
            print(f"skipped {i} {e}")
        except NoRootError as e:
            print(f"skipped {i} {e}")
        else:
            print(f"generated {i}")
        # if i>4: break
    return dataset
    

def formatting_prompts_func(example, method):    
    if method=="io":
        instruction = (
            "Given the following partial folder tree and file/folder description, "
            "predict which file/folder does the user wants or if not visible, predict the folder to explore.\n"
            f"Folder tree:\n{example['visible_tree']}\n"
            f"File/Folder description: {example['desired_description']}"
        )
        return {
            "instruction": instruction,
            "output": example["deepest_folder"]
        }
    #used for mistral
    elif method=="sys_use_ass":
        return {
            "text": [
                    {"role": "system", 
                     "content": "You are a file retrieval assistant. "
                                "Given the following partial folder tree and file/folder description, "
                                "If the desired file/folder is visible, output it, else, predict the next folder to explore.\n"},

                    {"role": "user", 
                     "content": f"Folder tree:\n{example['visible_tree']}\n"
                                f"File/Folder description: {example['desired_description']}"
                                f"Pick one of the following: {example['visible_tree']}"}, ###

                    {"role": "assistant", "content": example["deepest_folder"]},
                ]            
        }
    
    elif method=="sys_use_ass_list":

        return {
            "text": [
                    {"role": "system", 
                     "content": "You are a file retrieval assistant. "
                                "Given the following file/folder description and a partial folder tree, "
                                "if the desired file/folder is visible, output it, else, predict the next folder to explore.\n"},

                    {"role": "user", 
                     "content": 
f"""File/Folder description: 
{example['desired_description']}
Pick from one of the following:"""+
'\n'.join(extract_paths(example['visible_tree']))},

                    {"role": "assistant", "content": example["deepest_folder"]},
                ]            
        }
    elif method=="SUA_number_list":
        path_str="\n".join([f"{i+1}. {path}" for i,path in enumerate(extract_paths(example['visible_tree']))])
        return {
            "text": [
                {"role": "system", 
                    "content": 
"""You are a file-finder AI. Your task:
- The user provides a file/folder description and a list of visible items.
- You must choose one of those items by the following 2 rules:
    1. If the described item is in the visible list, choose it.         
    2. If not visible, choose the next folder to open.
- The answer is in the format "Final Answer: {path to file/folder}"
- Again, the answer is within the list of visible items.
"""},

                {"role": "user", 
                    "content":                     
f"""Here is the description of what I am searching for: 
{example['desired_description']} 

Here are the visible items. Choose one of the following: 
{path_str}
"""},
                
                {"role": "assistant", "content": "Final Answer: "+ example["deepest_folder"]},
            ]            
        }
# Example usage
if __name__ == "__main__":
    # np.random.seed(42)
    # with open(os.path.join(RAW_DATA_PATH, 'file_systems_dataset.json'), 'r') as f:
    #     file_system_dataset = json.load(f)   

    # all_dataset = generate_dataset(file_system_dataset, num_desired=NUM_DESIRED)
    # # # print(json.dumps(used_dataset[:2], indent=2))  # Print first 2 samples
    # # print(all_dataset)

    # with open(os.path.join(PROCESSED_DATA_PATH, 'all_dataset_test.json'), "w") as f:
    #     f.write(json.dumps(all_dataset, indent=2))
    # pass
    with open(os.path.join(PROCESSED_DATA_PATH, 'all_dataset.json'), "r") as f:
        all_dataset = json.load(f)
    
    used_dataset = []
    method="SUA_number_list"
    with open(os.path.join(USED_DATA_PATH, f"used_dataset_{method}.json"), "w") as f:
        for persona in all_dataset:
            # formatted_data = [formatting_prompts_func(desired) for desired in persona['visibility_data']]
            formatted_data = [formatting_prompts_func(desired, method=method) for desired in persona['visibility_data']]
            used_dataset.extend(formatted_data)
            
        f.write(json.dumps(used_dataset, indent=2))
    print(f"\nDataset saved to used_dataset_{method}.json ({len(all_dataset)} samples)")


