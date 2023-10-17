from transformers import AutoTokenizer\

model_name = "lmsys/vicuna-13b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def token_lookup(tokenizer, token_id):
    return tokenizer.convert_ids_to_tokens(token_id)

tokens = tokenizer.encode("The student's best guess is incorrect")

print(tokens)

for token in tokens:
    print(token_lookup(tokenizer, token))