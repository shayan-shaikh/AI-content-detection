from flask import Flask, request
import gradio as gr
import re

app = Flask(__name__)

import torch
from torch import cuda
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = 'cuda' if cuda.is_available() else 'cpu'
model_id = "gpt2"
modelgpt2 = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizergpt2 = GPT2TokenizerFast.from_pretrained(model_id)

def text_to_sentences(text):
    clean_text = text.replace('\n', ' ')
    return re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', clean_text)

def calculatePerplexity(text):
    encodings = tokenizergpt2("\n\n".join([text]), return_tensors="pt")
    max_length = modelgpt2.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = modelgpt2(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    return ppl.item()
    
def calculatePerplexities(text):
    sentences = text_to_sentences(text)
    perplexities = []
    for sentence in sentences:
        perplexity = calculatePerplexity(sentence)
        label = "Human"
        if perplexity<25:
            label = "AI"
        perplexities.append({"sentence": sentence, "perplexity": perplexity, "label": label})
    return perplexities

demo = gr.Interface(
        fn=calculatePerplexities, 
        inputs=gr.Textbox(placeholder="Copy and paste here..."), 
        article = "Visit <a href = \"https://ai-content-detector.online/\">AI Content Detector</a> for better user experience!",
        outputs=gr.outputs.JSON(),
        # interpretation="default",
        examples=["Cristiano Ronaldo is a Portuguese professional soccer player who currently plays as a forward for Manchester United and the Portugal national team. He is widely considered one of the greatest soccer players of all time, having won numerous awards and accolades throughout his career. Ronaldo began his professional career with Sporting CP in Portugal before moving to Manchester United in 2003. He spent six seasons with the club, winning three Premier League titles and one UEFA Champions League title. In 2009, he transferred to Real Madrid for a then-world record transfer fee of $131 million. He spent nine seasons with the club, winning four UEFA Champions League titles, two La Liga titles, and two Copa del Rey titles. In 2018, he transferred to Juventus, where he spent three seasons before returning to Manchester United in 2021. He has also had a successful international career with the Portugal national team, having won the UEFA European Championship in 2016 and the UEFA Nations League in 2019.", "One rule of thumb which applies to everything that we do - professionally and personally : Know what the customer want and deliver. In this case, it is important to know what the organisation what from employee. Connect the same to the KRA. Are you part of a delivery which directly ties to the larger organisational objective. If yes, then the next question is success rate of one’s delivery. If the KRAs are achieved or exceeded, then the employee is entitled for a decent hike."])
demo.launch(show_api=False)