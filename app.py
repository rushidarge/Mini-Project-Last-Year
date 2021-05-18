import streamlit as st
import cv2
import numpy as np
import pandas as pd
import numpy as np
import keras
import keras.layers as L
import keras.models as M
import tensorflow as tf
import os
from keras.utils import Sequence
from keras.models import load_model

st.write("""
         # Handwritten Recognition
         """
         )
st.write("This is a simple Handwritten Recognition web app to predict your handwritten text")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])	

# load weights into new model
model = load_model("prediction_model_ocr.h5",compile=False)
# print("Loaded model from disk")
train=pd.read_csv('https://raw.githubusercontent.com/rushidarge/Mini-Project-Last-Year/main/Data/written_name_train_v2.csv')

characters=set()
train['IDENTITY']=train['IDENTITY'].apply(lambda x: str(x))
for i in train['IDENTITY'].values:
    for j in i :
        if j not in characters :
            characters.add(j)
characters=sorted(characters)

# 2 Dictionaries  :   Turn all ur characters to num and vice versa
char_to_label = {char:label for label,char in enumerate(characters)}
label_to_char = {label:char for label,char in enumerate(characters)}

# A utility to decode the output of the network
def decode_batch_predictions(pred):
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0])*pred.shape[1]
    
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, 
                                        input_length=input_len,
                                        greedy=True)[0][0]
    
    # Iterate over the results and get back the text
    output_text = []
    for res in results.numpy():
        outstr = ''
        for c in res:
            if c < len(characters) and c >=0:
                outstr += label_to_char[c]
        output_text.append(outstr)
    
    # return final text results
    return output_text

def model_predict(img,model):
    # batch_images=np.ones((128,256,64,1),dtype=np.float32)
    img=cv2.imread(img)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(256,64))
    img=(img/255).astype(np.float32)
    img=img.T
    img=np.expand_dims(img,axis=-1)
    a = model.predict(img.reshape(1, 256, 64, 1))
    pred_texts = decode_batch_predictions(a)
    # pred_texts = pred_texts[0]    
    return pred_texts[0]

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    img_array = np.array(image)
    cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    prediction = model_predict('out.jpg', model)
    st.write(prediction)
