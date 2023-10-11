import torch
from fastapi import FastAPI
import uvicorn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import mains.summarizer as smr


# Data structure
from schema.item import Item


#define the fastapi
app = FastAPI(title="Summarization API",
            description="API for Summarization",
            version="1.0")


#when the app start, load the model
@app.on_event('startup')
async def load_model():
    smr.model = AutoModelForSeq2SeqLM.from_pretrained("./models/vit5_large_summarization", local_files_only = True)
    smr.tokenizer = AutoTokenizer.from_pretrained("./models/vit5_large_summarization", local_files_only = True)
    smr.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    smr.model.to(smr.device)

  
#when post event happens to /predict
@app.post('/api/v1/summary')
async def get_summarization(item:Item):
    result = ""
    text = item.raw + " </s>"
    encoding = smr.tokenizer(text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to(smr.device)
    outputs = smr.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=4069,
            num_beams=5,
            early_stopping=False
        )
    for output in outputs:
            result += smr.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    item.summarization = result
    return item.summarization


if __name__ == '__main__':
    uvicorn.run(app, port=5000, host='localhost')

