import json
import os
import re
import json
import random
from treelib import Tree
import re
from copy import deepcopy
import ollama

FILE_PATH=os.path.dirname(os.path.abspath(__file__))
CODE_PATH= os.path.join(FILE_PATH, '..', '..')
DATA_PATH=os.path.join(CODE_PATH, 'data')
PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
USED_DATA_PATH=os.path.join(DATA_PATH, 'used')

MODEL = 'llama3.1:8b'   # Name of your local model (verify with `ollama list`)
SLEEP_TIME = 1    # Delay between requests to avoid overloading local model
description_prompt_template = """Create a less than 5 word description of this Linux file/folder. You may mention if it's a file or folder but do not mention its name:
{ff}"""

def generate_description(ff):       
    # Generate file tree
    description_response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": description_prompt_template.format(ff=ff)}]
    )
    description = description_response['message']['content'].strip()    
    return description
# def parse_tree_string(tree_str):
#     """Improved parser handling box-drawing characters and empty lines"""
#     lines = [l for l in tree_str.splitlines() if l.strip() and l.strip() not in ('<START>', '<END>')]
    
#     tree = Tree()
#     stack = []  # (parent_node, current_depth)
    
#     for line in lines:
#         # Calculate indent using box-drawing characters
#         indent = len(re.match(r'[│├└─ ]*', line).group(0))
#         name = re.sub(r'^[│├└─ ]*', '', line).strip()
        
#         # Find parent in stack
#         while stack and stack[-1][1] >= indent:
#             stack.pop()
            
#         parent = stack[-1][0] if stack else None
        
#         # Create node with path-based ID
#         node_id = f"{parent.identifier}/{name}" if parent else name
        
#         if not tree.contains(node_id):
#             tree.create_node(tag=name, identifier=node_id, parent=parent)
            
#         stack.append((tree[node_id], indent))
    
#     return tree

def parse_tab_tree(tree_str):
    """Parse tab-indented file system structure into treelib Tree"""
    tree = Tree()
    lines = [line.rstrip() for line in tree_str.split('\n') 
            if line.strip() and not line.startswith(('<START>', '<END>'))]
    start_idx = 0
    for line in lines:
        if line.startswith('~'):
            break
        else:
            start_idx += 1
    lines=lines[start_idx:]            
    if not lines:
        return tree  # Empty tree
    
    # Create root node
    root_name = lines[0].strip()
    tree.create_node(root_name, root_name)
    stack = [(root_name, 0)]  # (node_id, depth)

    for line in lines[1:]:
        depth = (len(line) - len(line.lstrip(' ')))//4
        name = line.strip()
        
        # Find parent by comparing depths
        while stack and stack[-1][1] >= depth:
            stack.pop()
        
        parent_id = stack[-1][0] if stack else None
        node_id = f"{parent_id}/{name}" if parent_id else name
        
        if not tree.contains(node_id):
            tree.create_node(name, node_id, parent=parent_id)
            stack.append((node_id, depth))

    return tree
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
    if desired in visible_tree.nodes:
        for sibling in visible_tree.siblings(desired):
            visible_tree.remove_node(sibling.identifier)
        visible_tree.remove_node(desired)

    deepest_visible = get_deepest_visible(complete_tree=complete_tree, visible_tree=visible_tree, desired=desired)    
    
    return {
        'visible_tree': str(visible_tree), #visible_tree,
        'desired_description': generate_description(desired),
        'desired_path': desired,
        'deepest_folder': deepest_visible
    }
def get_deepest_visible(complete_tree, visible_tree, desired):
    not_in_visible=[i for i in complete_tree.rsearch(desired, filter=lambda node: node.identifier not in visible_tree.nodes)]
   
    return not_in_visible[-1][:not_in_visible[-1].rfind('/')]

def generate_visibility_data(complete_tree, num_desired=3):
    all_desired= random.sample(list(complete_tree.nodes),num_desired)
    visibility_data=[generate_visibility(complete_tree=complete_tree, desired=desired) for desired in all_desired]
    return visibility_data

def generate_dataset(file_system_dataset):
    dataset=[]
    for i,example in enumerate(file_system_dataset):
        complete_tree=parse_tab_tree(example['tree'])
        visibility_data=generate_visibility_data(complete_tree=complete_tree, num_desired=3)
        dataset.append({
            'persona': example['persona'],
            'complete_tree': str(complete_tree),
            'visibility_data': visibility_data
        })
        # if i>2: break
    return dataset

# Example usage
if __name__ == "__main__":
    # with open(os.path.join(RAW_DATA_PATH, 'file_systems_dataset.json'), 'r') as f:
    #     file_system_dataset = json.load(f)   

    # all_dataset = generate_dataset(file_system_dataset)
    # # print(json.dumps(used_dataset[:2], indent=2))  # Print first 2 samples
    # print(all_dataset)
    # with open(os.path.join(PROCESSED_DATA_PATH, 'all_dataset.json'), 'w') as f:
    #     json.dump(all_dataset, f, indent=2)

    with open(os.path.join(PROCESSED_DATA_PATH, 'all_dataset.json'), 'r') as f:
        all_dataset = json.load(f) 
    used_dataset = []
    with open(os.path.join(USED_DATA_PATH, 'used_dataset.json'), "w") as f:
        for persona in all_dataset:
            for desired in persona['visibility_data']:
                instruction = f"Visible Tree: {desired['visible_tree']}\nDescription: {desired['desired_description']}"
                output = desired["deepest_folder"]       
                used_dataset.append({"text": f"### Instruction: {instruction}\n### Output: {output}"})
        f.write(json.dumps(used_dataset, indent=2))
    print(f"\nDataset saved to used_dataset.json ({len(all_dataset)} samples)")
