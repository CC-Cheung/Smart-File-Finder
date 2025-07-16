from unsloth import FastLanguageModel
import os
import subprocess
from unsloth import get_chat_template
FILE_PATH=os.path.dirname(os.path.abspath(__file__))
CODE_PATH= os.path.join(FILE_PATH, '..', '..')
MODEL_PATH=os.path.join(CODE_PATH, 'models')
FINETUNED_PATH=os.path.join(MODEL_PATH, 'finetuned')
BASE_MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

#Careful about name length
#Careful about debugging
FINETUNED_PATH="/home/kids/Linux_Coding/Smart-File-Finder/model_training/models/finetuned/"
GGUF_PATH = os.path.join(FINETUNED_PATH, 'mistral_SUA_pick_number.gguf')
ADAPTER_PATH = os.path.join(FINETUNED_PATH, 'mistral_SUA_pick_number')

#Double check
OLLAMA_PATH = os.path.join(FINETUNED_PATH, 'ollama')
OLLAMA_MODEL_NAME="mistral_SUA_pick_number"
MODEL_FILE_NAME="Modelfile"

  
# GET GGUF first if not done
# DO NOT model.load_adapter(ADAPTER_PATH). Use adapater path directly with FastLanguageModel.from_pretrained(
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model) 

tokenizer = get_chat_template(
        tokenizer,
        chat_template = "mistral", # change this to the right chat_template name
    )


model.save_pretrained_gguf(
        GGUF_PATH,
        tokenizer,
        quantization_method="q4_k_m",
        maximum_memory_usage=0.0001,
)

pass
# Edit the FROM location to the gguf file
modelfile_content='''FROM /home/kids/Linux_Coding/Smart-File-Finder/model_training/models/finetuned/mistral_SUA_pick_number.gguf/unsloth.Q4_K_M.gguf
TEMPLATE """{{- if .Messages }}
{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}
{{- if and (eq (len (slice $.Messages $index)) 1) $.Tools }}[AVAILABLE_TOOLS] {{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST] {{ if and $.System (eq (len (slice $.Messages $index)) 1) }}{{ $.System }}

{{ end }}{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }}{{ .Content }}
{{- else if .ToolCalls }}[TOOL_CALLS] [
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}]
{{- end }}</s>
{{- else if eq .Role "tool" }}[TOOL_RESULTS] {"content": {{ .Content }}} [/TOOL_RESULTS]
{{- end }}
{{- end }}
{{- else }}[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}[/INST]
{{- end }}{{ .Response }}
{{- if .Response }}</s>
{{- end }}"""
PARAMETER stop [INST]
PARAMETER stop [/INST]
PARAMETER temperature 0.8
'''

with open(os.path.join(OLLAMA_PATH, MODEL_FILE_NAME), "w") as f:
    f.write(modelfile_content)

subprocess.run([
    "ollama", "create", OLLAMA_MODEL_NAME, "-f", os.path.join(OLLAMA_PATH, MODEL_FILE_NAME)
], check=True)

print(f"Created Ollama model: {OLLAMA_MODEL_NAME}")
# Create Ollama model
