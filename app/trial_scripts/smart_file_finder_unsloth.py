import argparse
import os
# import ollama  # Make sure ollama is installed and running
import re
import string
from treelib import Tree
import os
from unsloth import FastLanguageModel
from unsloth import get_chat_template
# Change to your fine-tuned model if needed
MODEL_NAME = 'deepseek-r1:8b'  #perfect but slow af
MODEL_NAME = 'llama3.1:latest'  # alright, pretty fast, but sometimes wrong answer
# MODEL_NAME = 'dolphin3:latest'  # wrong Final Answer: my files/Uni Sanda club tournament pic folder
# MODEL_NAME = 'mistral:latest'  # bad format  Final Answer: 1. my files (assuming that under 'my files' there is a subfolder named "Uni Sanda club tournament pic")
# MODEL_NAME = 'llama3.2' #bad format, only number
# MODEL_NAME = "deepseek-r1:1.5b" #bad answer

MODEL_NAME = "mistral_SUA_number_list3:latest"
MODEL_NAME = "/home/kids/Linux_Coding/Smart-File-Finder/model_training/models/finetuned/mistral_SUA_number_list3"

SELECTION_PHRASE = "The desired file/folder is "
EXPLORATION_PHRASE = "The next folder to open is "
ANSWER_PHRASE= "Final Answer: "
PUNCTUATION = string.punctuation.replace('/', '')
# def list_tree(root, max_depth=2, prefix='~'):
#     """Builds a flat list of files/folders up to max_depth."""
#     tree = []
#     for dirpath, dirnames, filenames in os.walk(root):
#         depth = dirpath[len(root):].count(os.sep)
#         if depth > max_depth:
#             continue
#         rel_dir = os.path.relpath(dirpath, os.path.expanduser('~'))
#         rel_dir = prefix if rel_dir == '.' else f"{prefix}/{rel_dir}"
#         tree.append(rel_dir)
#         for d in dirnames:
#             tree.append(f"{rel_dir}/{d}")
#         for f in filenames:
#             tree.append(f"{rel_dir}/{f}")
#     return sorted(set(tree))

def remove_non_alphanumeric_edges(s: str) -> str:
    # Remove non-alphanumeric characters from the start and end of the string
    return re.sub(r'^\W+|\W+$', '', s)


class FileExplorerTree:
    def __init__(self, full_root_path):
        self.tree = Tree()
        root_path=os.path.basename(full_root_path)
        self.path_before_root = full_root_path.rstrip(root_path)
        # from starting folder
        self.tree.create_node(tag=root_path, identifier=root_path)
          # If the suggestion is a file/folder in the current tree, print and exit
        
      
    def handle_suggestion(self, suggestion):
        """Add the real children of folder_path to the tree, only if folder_path is already in the tree."""
        pattern=f"[{re.escape(PUNCTUATION)}\t\n\r\f\v]"
        path =  re.split(pattern ,suggestion.split(ANSWER_PHRASE)[-1], 1)[-1].strip()
        # path = remove_non_alphanumeric_edges(raw_path)
        full_path = os.path.join(self.path_before_root, path)
        
        if not self.tree.contains(path):
            raise ValueError(f"Folder '{path}' is not in the tree. Cannot explore.")
        elif os.path.isdir(full_path) and self.tree.nodes[path].is_leaf():
            for entry in os.listdir(full_path):
                entry_path = os.path.join(path, entry)
                self.tree.create_node(tag=entry, identifier=entry_path, parent=path)
        else:
            return path
        
        return None  

    def handle_suggestion_streamline(self, suggestion):
        """Add the real children of folder_path to the tree, only if folder_path is already in the tree."""
        raw_path = suggestion.split(SELECTION_PHRASE)[-1].split(EXPLORATION_PHRASE)[-1]
        path = remove_non_alphanumeric_edges(raw_path)

        full_path = os.path.join(self.path_before_root, path)

        if not self.tree.contains(path):
            raise ValueError(f"Folder '{path}' is not in the tree. Cannot explore.")
        
        if SELECTION_PHRASE in suggestion:
            return path
        elif EXPLORATION_PHRASE in suggestion:

            if os.path.isdir(full_path) and self.tree.nodes[path].is_leaf():
                for entry in os.listdir(full_path):
                    entry_path = os.path.join(path, entry)
                    self.tree.create_node(tag=entry, identifier=entry_path, parent=path)
            elif not self.tree.nodes[path].is_leaf():
                raise ValueError(f"'{path}' not leaf, already explored.")
            elif not os.path.isdir(full_path):
                raise ValueError(f"'{full_path}' is not a directory, cannot be explored.")

        
        

        

    def get_partial_view(self, folder_path):
        """Return a dict of immediate children under folder_path."""
        children = self.tree.children(folder_path)
        return {child.tag: child.identifier for child in children}

    def __str__(self, folder_path=None):
        return str(self.tree)
    
    def list_nodes(self, folder_path=None):
        output=""
        for i, key in enumerate(self.tree.nodes.keys()):
            output += f"{i+1}. {key}\n"
        return output

    
def pre_apply_chat_template_gen_tokenize(example):  
    conversations = example["text"][:-1]  
    input_ids = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True)  
    return {"input_ids": input_ids}  



def prompt_model(description, tree):
    """Send prompt to Ollama and get response."""
    system_prompt="""You are a file-finder AI. Your task:
- The user provides a file/folder description and a list of visible items.
- You must choose one of those items by the following 2 rules:
    1. If the described item is in the visible list, choose it.         
    2. If not visible, choose the next folder to open.
- The answer is in the format "Final Answer: {path to file/folder}"
- Again, the answer is within the list of visible items.
"""
    user_prompt=f'''Here is the description of what I am searching for: 
{description} 

Here are the visible items. Choose one of the following: 
{tree}
'''

    messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    
#     response = ollama.chat(
#         model=MODEL_NAME,
#         messages=[
#             {'role': 'system', 'content': system_prompt},
#             {'role': 'user', 'content': user_prompt}
#         ]
#     )
    # return response['message']['content']
    

    inputs=tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True, truncation=True).to(model.device)

    outputs = model.generate(inputs, 
                       max_new_tokens = 128, 
                       pad_token_id = tokenizer.eos_token_id,
                       
                       )

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return outputs
    # return description

def main():
    parser = argparse.ArgumentParser(description="Smart File Finder with Ollama model")
    parser.add_argument("description", help="File/folder description to search for")
    parser.add_argument(
        "--start",
        default=os.getcwd(),
        help="Start folder for the search (default: current working directory)"
    )
    args = parser.parse_args()

    description = args.description
    start = args.start
    visible_tree=FileExplorerTree(start)
    testing=[start, "Hello", "h"]
    for i in range(1,len(testing)):
        testing[i]=os.path.join(testing[i-1], testing[i])
    testing+=[testing[-1]]
    i=0
    result=None
    while True:
        suggestion = prompt_model(description, visible_tree.list_nodes())
        # suggestion = prompt_model(testing[i], str(visible_tree))

        print(f"\nModel suggestion: {suggestion}")
        try:
            # result=visible_tree.handle_suggestion(suggestion)
            result=visible_tree.handle_suggestion(suggestion)

        except ValueError as e:
            print(f"Error: {e}")
            break
        if result is not None:
            break
      
        i+=1
    print(result)
if __name__ == "__main__":

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
        # device_map="cpu",  # Force CPU usage
    )
    main()

# my files
# ├── Hello
# │   ├── h
# │   └── hello.txt
# └── New York
# where should I look for my empire state building photos? Format the answer like "The directory to look in is {path to directory}"