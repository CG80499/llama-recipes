import os
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel, AutoPeftModelForCausalLM
import torch

output_dir = "llama-recipes/FT-vicuna-13b-v1.5-16k-v3"

model_name = "lmsys/vicuna-13b-v1.5-16k"

output_merged_dir = "vicuna-13b-v1.5-16k-paper-summary-v2"

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)

def load_model(model_name):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

os.makedirs(output_merged_dir, exist_ok=True)
model = load_model("lmsys/vicuna-13b-v1.5-16k")
model = load_peft_model(model, output_dir)
model = model.merge_and_unload()

# state_dict = model.state_dict()

model.save_pretrained(output_merged_dir, safe_serialization=False)

print("model saved!")

# with open("prompt_test.txt", "r") as f:
#     prompt = f.read()

# inputs = tokenizer(prompt, return_tensors="pt")

# generate_ids = model.generate(inputs.input_ids.to("cuda"), max_length=7000)

# print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# model.save_pretrained(output_merged_dir, safe_serialization=True)

#INFO 08-27 13:32:39 llm_engine.py:70] Initializing an LLM engine with config: model='vicuna-13b-v1.5-16k-paper-summary-v0', tokenizer='vicuna-13b-v1.5-16k-paper-summary-v0', tokenizer_mode=auto, trust_remote_code=False, dtype=torch.float16, use_dummy_weights=False, download_dir=None, use_np_weights=False, tensor_parallel_size=1, seed=0)