import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーのロード
model_name = "cyberagent/open-calm-large"
# model_name = "cyberagent/open-calm-3b"
model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("Clinical AIは、", return_tensors="pt").to(model.device)
with torch.no_grad():
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )
    
output = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(output)