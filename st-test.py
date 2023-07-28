import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("./AI-Content-Detector-V2", local_files_only =True)
model = AutoModelForSequenceClassification.from_pretrained("./AI-Content-Detector-V2", local_files_only = True, num_labels=2)
    
def predict(query):
    tokens = tokenizer.encode(query)
    tokens = tokens[:tokenizer.model_max_length - 2]
    tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
        probs = logits.sigmoid()
    fake, real = probs.detach().cpu().flatten().numpy().tolist() 
    return real

st.title("AI Content Detector")

text = st.text_input("Enter text:")

if st.button("Predict"):

    prediction = predict(text)

    if prediction < 0.80:
        st.write("This text is AI generated 🚨 with probability : "+ str(1 - prediction) )
    else:
        st.write("This text is real ✅ with a probability : " + str(prediction) )
