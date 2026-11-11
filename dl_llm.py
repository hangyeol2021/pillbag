from huggingface_hub import snapshot_download
p=snapshot_download("Qwen/Qwen2.5-7B-Instruct", allow_patterns=["*.json","*.safetensors","*.txt","tokenizer*","merges.txt","vocab.json"])
print("OK",p,flush=True)
