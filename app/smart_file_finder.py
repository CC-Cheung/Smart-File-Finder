import argparse
import os
import ollama  # Make sure ollama is installed and running
import re
import string
from treelib import Tree
import os
import logging
# Change to your fine-tuned model if needed
MODEL_NAME = "mistral_SUA_number_list3:latest"

SELECTION_PHRASE = "The desired file/folder is "
EXPLORATION_PHRASE = "The next folder to open is "
ANSWER_PHRASE= "Final Answer: "
PUNCTUATION = string.punctuation.replace('/', '')
#logging constants
MAIN_PROGRAM_INFO= 21
MAIN_PROGRAM= 22
class AllowedError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def remove_non_alphanumeric_edges(s: str) -> str:
    # Remove non-alphanumeric characters from the start and end of the string
    return re.sub(r'^\W+|\W+$', '', s)


class FileExplorerTree:
    def __init__(self, full_root_path):
        self.tree = Tree()
        root_path=os.path.basename(full_root_path)
        self.path_before_root = full_root_path.rstrip(root_path)
        self.tree.create_node(tag=root_path, identifier=root_path)
        self.blacklist=[]
    def handle_suggestion(self, suggestion, logger):
        """Add the real children of folder_path to the tree, only if folder_path is already in the tree."""
        path=suggestion.split(ANSWER_PHRASE)[-1].strip("\t\n\r\f\v"+PUNCTUATION)
        path=path.lstrip("\t\n\r\f\v "+PUNCTUATION+string.digits)
        path=path.rstrip("\t\n\r\f\v "+PUNCTUATION)

        logger.main_program_info(f"Model suggestion: {path}")

        full_path = os.path.join(self.path_before_root, path)
        
        if not self.tree.contains(path):
            raise AllowedError(f"Folder '{path}' is not in the tree. Cannot explore.")
        elif os.path.isdir(full_path) and self.tree.nodes[path].is_leaf():
            for entry in os.listdir(full_path):
                entry_path = os.path.join(path, entry)
                self.tree.create_node(tag=entry, identifier=entry_path, parent=path)
        else:
            return path
        
        return None  
    def get_partial_view(self, folder_path):
        """Return a dict of immediate children under folder_path."""
        children = self.tree.children(folder_path)
        return {child.tag: child.identifier for child in children}

    def __str__(self, folder_path=None):
        return str(self.tree)
    
    def list_nodes(self, folder_path=None):
        output=""
        i=0
        
        for key in self.tree.nodes.keys():
            if key in self.blacklist:
                continue
            output += f"{i+1}. {key}\n"
            i+=1
        return output

    
def prompt_model(description, tree):
    # From training
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
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    )
    return response['message']['content']

def get_logger(level=logging.INFO):  

    logging.addLevelName(MAIN_PROGRAM, "MAIN_PROGRAM")
    logging.addLevelName(MAIN_PROGRAM_INFO, "MAIN_PROGRAM_INFO")

    def main_program(self, message, *args, **kwargs):
        if self.isEnabledFor(MAIN_PROGRAM):
            self._log(MAIN_PROGRAM, message, args, **kwargs)
    def main_program_info(self, message, *args, **kwargs):
        if self.isEnabledFor(MAIN_PROGRAM_INFO):
            self._log(MAIN_PROGRAM_INFO, message, args, **kwargs) 

    logging.Logger.main_program = main_program
    logging.Logger.main_program_info = main_program_info

    logger = logging.getLogger(__name__)

    logger.setLevel(level)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
    # return description
def main():
    parser = argparse.ArgumentParser(description="Smart File Finder with Ollama model")
    parser.add_argument("description", help="File/folder description to search for")
    parser.add_argument(
        "--start",
        default=os.getcwd(),
        help="Start folder for the search (default: current working directory)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help="See intermediate folders"
    )


    args = parser.parse_args()
    logger=get_logger(MAIN_PROGRAM_INFO if args.verbose else MAIN_PROGRAM)

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

        try:
            result=visible_tree.handle_suggestion(suggestion, logger)

        except AllowedError as e:
            logger.main_program_info(e)
            num_bad_outputs+=1
            if num_bad_outputs>20:
                logger.main_program("Too many (>20) bad outputs in a row. Try searching yoruself")
                break
            continue
        except Exception as e:
            logger.error(e)
            break
        num_bad_outputs=0
        if result is not None:
            logger.main_program(f"It should be here: {result}")
            continue_search = input("Continue searching? (y/n): ")
            if continue_search.lower() != "y":
                logger.main_program("Goodbye!")
                break  
            visible_tree.blacklist.append(result)
        

        i+=1

if __name__ == "__main__":
    
    main()

# my files
# ├── Hello
# │   ├── h
# │   └── hello.txt
# └── New York
# where should I look for my empire state building photos? Format the answer like "The directory to look in is {path to directory}"