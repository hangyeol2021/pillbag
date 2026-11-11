from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
for m in ["KETI-AIR/ke-t5-base","gogamza/kobart-summarization"]:
    try:
        t=AutoTokenizer.from_pretrained(m); AutoModelForSeq2SeqLM.from_pretrained(m)
        print("OK",m,len(t("1회 1정, 1일 3회 경구 투여한다.").input_ids),flush=True)
    except Exception as e: print("FAIL",m,str(e)[:150],flush=True)
