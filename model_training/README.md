# Model Training

The folder includes all the code needed to recreate the app's model. With slight modifications, it can be tailored to fit your particular setup. 

## Workflow
src folder contains the main python code while script folder contains supporting code.

1. src/data/gen_file_system.py to make the various file systems. **Uses Ollama and Faker**
2. src/data/gen_file_data.py to make the training data and format it correctly.**Uses Ollama**
3. src/training/finetune.py to train the model. **Uses Unsloth**
4. script/test_model_unsloth.py to obtain model outputs for training data. **Uses Unsloth**
5. script/read_results.py to visualize and compare different model performance test results.
6. script/utils/ollamaing.py to convert adapters to quantized gguf models and then to ollama models. **Uses Unsloth, calls Ollama**

