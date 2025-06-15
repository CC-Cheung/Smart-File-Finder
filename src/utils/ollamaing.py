# from unsloth import FastLanguageModel
import os
import subprocess
FILE_PATH=os.path.dirname(os.path.abspath(__file__))
CODE_PATH= os.path.join(FILE_PATH, '..', '..')
MODELS_PATH=os.path.join(CODE_PATH, 'models')
FINETUNED_PATH=os.path.join(MODELS_PATH, 'finetuned')
MODEL_PATH = os.path.join(FINETUNED_PATH, 'Mistral_d66ea65')

GGUF_PATH = os.path.join(MODEL_PATH, 'Mistral_d66ea65.gguf')
ADAPTER_PATH = os.path.join(MODEL_PATH, 'Mistral_d66ea65_adapters')
OLLAMA_PATH = os.path.join(MODEL_PATH, 'ollama')
OLLAMA_MODEL_NAME="Mistral_d66ea65"
# def formatting_prompts_func(example, method):    
#     if method=="io":
#         instruction = (
#             "Given the following partial folder tree and file/folder description, "
#             "predict which file/folder does the user wants or if not visible, predict the folder to explore.\n"
#             f"Folder tree:\n{example['visible_tree']}\n"
#             f"File/Folder description: {example['desired_description']}"
#         )
#         return {
#             "instruction": instruction,
#             "output": example["deepest_folder"]
#         }
#     elif method=="sys_use_ass":
#         return {
#             "text": [
#                     {"role": "system", 
#                      "content": "You are a file retrieval assistant. "
#                                 "Given the following partial folder tree and file/folder description, "
#                                 "If the desired file/folder is visible, output it, else, predict the next folder to explore.\n"},

#                     {"role": "user", 
#                      "content": f"Folder tree:\n{example['visible_tree']}\n"
#                                 f"File/Folder description: {example['desired_description']}"},

#                     {"role": "assistant", "content": example["deepest_folder"]},
#                 ]            
#         }
    
###WEIRD ERROR IF DIRECTLY FROM ADAPTER_PATH
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
#     max_seq_length=2048,
#     load_in_4bit=True,
# )
# Load adapter from directory
# model.load_adapter(ADAPTER_PATH)


# 1. Enable inference mode
# FastLanguageModel.for_inference(model) 

modelfile_content = f"""FROM {GGUF_PATH}
ADAPTER {ADAPTER_PATH}
SYSTEM "You are a file retrieval assistant. Given the following partial folder tree and file/folder description, if the desired file/folder is visible, output it, else, predict the next folder to explore.\n
TEMPLATE \"\"\"<s>[INST] <<SYS>>\\n{{{{ .System }}}}\\n<</SYS>>\\n\\nFolder tree:\\n{{{{ .Prompt }}}}\\n\\nFile/Folder description: {{{{ .Input }}}} [/INST] {{{{ .Response }}}}</s>\"\"\"PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
"""

with open(os.path.join(OLLAMA_PATH, 'Modelfile'), "w") as f:
    f.write(modelfile_content)

subprocess.run([
    "ollama", "create", OLLAMA_MODEL_NAME, "-f", os.path.join(OLLAMA_PATH, 'Modelfile')
], check=True)

print(f"Created Ollama model: {OLLAMA_MODEL_NAME}")
# # Create Ollama model
# ollama create your_model -f Modelfile