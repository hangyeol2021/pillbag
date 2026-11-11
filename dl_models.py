from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
for m in ["gogamza/kobart-base-v2","paust/pko-t5-base"]:
    try:
        AutoTokenizer.from_pretrained(m); AutoModelForSeq2SeqLM.from_pretrained(m); print("OK",m,flush=True)
    except Exception as e: print("FAIL",m,e,flush=True)
