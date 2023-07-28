import functools

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# from torch import cuda
import gradio as gr
import os
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device

tokenizer = AutoTokenizer.from_pretrained("PirateXX/AI-Content-Detector-V2")

model = AutoModelForSequenceClassification.from_pretrained("PirateXX/AI-Content-Detector-V2")
def text_to_sentences(text):
    clean_text = text.replace('\n', ' ')
    return re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', clean_text)

# function to concatenate sentences into chunks of size 900 or less
def chunks_of_900(text, chunk_size = 900):
    sentences = text_to_sentences(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk + sentence) <= chunk_size:
            if len(current_chunk)!=0:
                current_chunk += " "+sentence
            else:
                current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    chunks.append(current_chunk)
    return chunks
    
def predict(query):
    tokens = tokenizer.encode(query)
    all_tokens = len(tokens)
    tokens = tokens[:tokenizer.model_max_length - 2]
    used_tokens = len(tokens)
    tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)

    with torch.no_grad():
        logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
        probs = logits.softmax(dim=-1)

    #real, fake = probs.detach().cpu().flatten().numpy().tolist() # Hello-SimpleAI/chatgpt-detector-roberta       
    fake, real = probs.detach().cpu().flatten().numpy().tolist() # PirateXX/AI-Content-Detector-V2

    return real

def findRealProb(text):
    chunksOfText = (chunks_of_900(text))
    results = []
    for chunk in chunksOfText:
        output = predict(chunk)
        results.append([output, len(chunk)])
    
    ans = 0
    cnt = 0
    for prob, length in results:
        cnt += length
        ans = ans + prob*length
    realProb = ans/cnt
    return {"Real": realProb, "Fake": 1-realProb}, results


st.markdown(""" <style> .appview-container .main .block-container {
    max-width: 100%;
    padding-top: 1rem;
    padding-right: {1}rem;
    padding-left: {1}rem;
    padding-bottom: {1}rem;
}</style> """, unsafe_allow_html=True)
#Add a logo (optional) in the sidebar
# logo = Image.open(r'C:\Users\13525\Desktop\Insights_Bees_logo.png')
with st.sidebar:
    choose = option_menu("Content Examiner", ["Inspect Content","Generate Content","About", "Contact"],
                        icons=['camera fill', 'kanban', 'book','person lines fill'],
                        menu_icon="app-indicator", default_index=0,
                        styles={
        "container": {"padding": "0 5 5 5 !important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )


if choose == "Inspect Content":
    #Add the cover image for the cover page. Used a little trick to center the image
    st.markdown(""" <style> .font {
        font-size:25px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown('<p class="font">Inspect Content</p>', unsafe_allow_html=True)
        
    with col2:               # To display brand logo                
        st.image('./inspection-1.jpg', width=100 )
        

# txt = st.text_area('Text to analyze', '''
#     It was the best of times, it was the worst of times, it was
#     the age of wisdom, it was the age of foolishness, it was
#     the epoch of belief, it was the epoch of incredulity, it
#     was the season of Light, it was the season of Darkness, it
#     was the spring of hope, it was the winter of despair, (...)
#     ''')

    txt = st.text_area('Add Text here','''
    Cristiano Ronaldo is a Portuguese professional soccer player who currently plays 
    as a forward for Manchester United and the Portugal national team. He is widely 
    considered one of the greatest soccer players of all time, having won numerous 
    awards and accolades throughout his career. Ronaldo began his professional career 
    with Sporting CP in Portugal before moving to Manchester United in 2003. 
    He spent six seasons with the club, winning three Premier League titles 
    and one UEFA Champions League title. In 2009, he transferred to Real Madrid 
    for a then-world record transfer fee of $131 million. He spent nine seasons with 
    the club, winning four UEFA Champions League titles, two La Liga titles, 
    and two Copa del Rey titles. In 2018, he transferred to Juventus, where he spent 
    three seasons before returning to Manchester United in 2021. He has also had 
    a successful international career with the Portugal national team, having won 
    the UEFA European Championship in 2016 and the UEFA Nations League in 2019.
    ''', height=300, max_chars=2000)


    if st.button('Analyze Content'): # st.session_state.input_text is not None
        with st.spinner('Loading the model..'):
            model.to(device)

            st.success(f'Model Loaded!', icon="✅")
            # st.success(f'Reported EER for the selected model {reported_eer}%')
        with st.spinner("Getting prediction..."):
            # print(audio.shape)
            predictions=findRealProb(txt)
            print('prediction_value',predictions)
            if predictions[0]['Fake'] > 0.80:
                # st.error(f"The Sample is spoof: \n Confidence {(prediction_value) }%",  icon="🚨")
                st.error(f"This text is AI generated with confidence: "+str(predictions[0]['Fake']),  icon="🚨")

            else:
                st.success(f"This text is real", icon="✅")
        

# if choose == "Generate Content":
#     st.markdown(""" <style> .font {
#         font-size:25px ; font-family: 'Cooper Black'; color: #FF9633;} 
#         </style> """, unsafe_allow_html=True)
#     st.markdown('<p class="font">Comparison of Models</p>', unsafe_allow_html=True)
#     data_frame = get_data()
#     tab1, tab2 = st.tabs(["EER", "min-TDCF"])
#     with tab1:
#         data_frame["EER ASVS 2019"] = data_frame["EER ASVS 2019"].astype('float64') 
#         data_frame["EER ASVS 2021"] = data_frame["EER ASVS 2021"].astype('float64') 
#         data_frame["Cross-dataset 19-21"] = data_frame["Cross-dataset 19-21"].astype('float64') 

#         data = data_frame[["Model Name","EER ASVS 2019","EER ASVS 2021","Cross-dataset 19-21"]].reset_index(drop=True).melt('Model Name')
#         chart=alt.Chart(data).mark_line().encode(
#             x='Model Name',
#             y='value',
#             color='variable'
#         )
#         st.altair_chart(chart, theme=None, use_container_width=True)
#     with tab2:
#         data_frame["min-TDCF ASVS 2019"] = data_frame["EER ASVS 2019"].astype('float64') 
#         data_frame["min-TDCF ASVS 2021"] = data_frame["EER ASVS 2021"].astype('float64') 
#         data_frame["min-TDCF Cross-dataset"] = data_frame["Cross-dataset 19-21"].astype('float64')

#         data = data_frame[["Model Name","min-TDCF ASVS 2019","min-TDCF ASVS 2021","min-TDCF Cross-dataset"]].reset_index(drop=True).melt('Model Name')
#         chart=alt.Chart(data).mark_line().encode(
#             x='Model Name',
#             y='value',
#             color='variable'
#         )
#         st.altair_chart(chart, theme=None, use_container_width=True)
#     # Data table
#     st.markdown(""" <style> .appview-container .main .block-container {
#         max-width: 100%;
#         padding-top: {1}rem;
#         padding-right: {1}rem;
#         padding-left: {1}rem;
#         padding-bottom: {1}rem;
#         }</style> """, unsafe_allow_html=True)
#     st.dataframe(data_frame, use_container_width=True)



if choose == "About":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About</p>', unsafe_allow_html=True)
if choose == "Contact":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Us</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Your Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        # if submitted:
        #     st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')

