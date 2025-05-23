from unsloth import FastLanguageModel
import os
ADAPTER_PATH = "/home/kids/Linux_Coding/LLM_Distillation_test/models/finetuned/lora_adapters"
HF_PATH = "/home/kids/Linux_Coding/LLM_Distillation_test/models/finetuned/hf_models"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = ADAPTER_PATH, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=2048,
    load_in_4bit=True,

)

# 1. Enable inference mode
# FastLanguageModel.for_inference(model) 

# # 2. Load the LoRA adapters

# model.save_pretrained_gguf("/home/kids/Linux_Coding/LLM_Distillation_test/models/finetuned/ollama_ready",
                            # tokenizer, quantization_method = "f16")


# import torch
# torch.cuda.empty_cache()  # Clear any cached memory
# model = model.cpu()
# # Save model to HF format first
# model.save_pretrained(HF_PATH)
# tokenizer.save_pretrained(HF_PATH)
