# App
## Setup
1. Install ollama.
1. Ensure you have a suitable ollama model. You can download it from [my ollama model repository](https://ollama.com/CC-Cheung/mistral_SUA_pick_number3). The command is ```ollama run CC-Cheung/mistral_SUA_pick_number3```. You may also finetune using the code in the Smart-File-Finder/model_training folder. Ensure the prompt and answer formats are compatible.
2. Ensure you are on/Navigate to this folder **Smart-File-Finder/app**.
3. Run the following in the environment of your choice:
```pip install .```
4. The ```sff``` command should be available in that environment.
5. You can still do it through python {path to smart_file_finder.py} {args} or just running it in and IDE.


## Calling

```sff --help
usage: sff [-h] [-s START] [-v] [-m MODEL] description

Smart File Finder with Ollama model

positional arguments:
  description        File/folder description to search for

options:
  -h, --help         show this help message and exit
  -s, --start START  Start folder for the search (default: current working directory)
  -v, --verbose      See intermediate folders
  -m, --model MODEL  Type ollama list and copy the model name (eg. mymodel:latest)
```
Tips:
1. Edit the defaul model so you don't need to keep inputting it in with ```-m```.
2. Use verbose as it gives an idea what the current partial view is (as well as other info).
