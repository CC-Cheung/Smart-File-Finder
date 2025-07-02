import argparse
import os
import ollama  # Make sure ollama is installed and running
import re

from treelib import Tree
import os

MODEL_NAME = 'deepseek-r1:8b'  # Change to your fine-tuned model if needed
SELECTION_PHRASE = "The desired file/folder is "
EXPLORATION_PHRASE = "The next folder to open is "
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
        raw_path = suggestion.strip(SELECTION_PHRASE).strip(EXPLORATION_PHRASE).rstrip('.')
        path = remove_non_alphanumeric_edges(raw_path)
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

    




def prompt_model(description, tree):
    """Send prompt to Ollama and get response."""
    # prompt = (
    #     f"File/Folder description: {description}\n"
    #     "Pick from one of the following:\n"
    #     + "\n".join(tree)
    # )
    # response = ollama.chat(
    #     model=MODEL_NAME,
    #     messages=[
    #         {"role": "system", "content": (
    #             "You are a file retrieval assistant. Given the following file/folder description and a partial folder tree, "
    #             "if the desired file/folder is visible, output it, else, predict the next folder to explore.\n"
    #         )},
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # return response['message']['content'].strip()
    if MODEL_NAME == 'deepseek-r1:8b':
        system_prompt = """
        You are a file-finder AI. Your task:
        - The user provides a file/folder description and a list of visible items.
        - If the described item is in the visible list:  
        Output EXACTLY: "The desired file/folder is {path to file/folder}"  
        - If not visible:  
        Output EXACTLY: "The next folder to open is {path to file}"
        - Again, the answer is within the list of visible items.
        """
        user_prompt = f"""
        Here is the description of what I am searching for: 
        {description}. 
        
        Here are the visible items. Choose one of the following: 
        {tree}
        """
        
        response = ollama.chat(
            model='deepseek-r1:8b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        return response['message']['content']
    return description

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
            result=visible_tree.handle_suggestion_streamline(suggestion)

        except ValueError as e:
            print(f"Error: {e}")
            break
        if result is not None:
            break
      
        i+=1
    print(result)
if __name__ == "__main__":
    main()

# my files
# ├── Hello
# │   ├── h
# │   └── hello.txt
# └── New York
# where should I look for my empire state building photos? Format the answer like "The directory to look in is {path to directory}"