from tensorflow.keras.models import load_model
from flask import Flask, render_template, redirect, url_for, request
import os
import cv2
import re as r
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
model=load_model('model.h5')


def preprocess_image(image):
    img=cv2.resize(image,(150,150))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=np.array(img)
    return img

def find_category(image):
    X=[]
    X.append(image)
    X=np.array(X)
    X=X.reshape(len(X),150,150,1)
    prediction=np.argmax(model.predict(X), axis=1)
    if prediction==0:
        return "Human"
    else:
        return "Horse"
    
app=Flask(__name__)

image=-1
image_name='nothing'


@app.route('/',methods=['GET', 'POST'])
def home():
    global image, image_name
    if request.method=="POST":
        file=request.files['file']
        #print(file)
        #print(secure_filename(file.filename))
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/images', secure_filename(file.filename))
        file.save(file_path)
        image_name=str(secure_filename(file.filename))
        #print(image_name)
        #print(file_path)
        image=cv2.imread(os.path.join('static/images', image_name))    
        #print(image)
        #print(os.path.join('static/images', image_name))
        return render_template('index.html', img_path=os.path.join('static/images',image_name), context=1)
    else:
        return render_template('index.html',context=0)
    
                    

@app.route('/resultpage',methods=['GET','POST'])
def predict():
    preprocessed_image=preprocess_image(image)
    answer=find_category(preprocessed_image)
    return render_template('result.html',answer=answer,img_path=os.path.join('static/images',image_name))


if __name__=='__main__':
    app.run(debug=True)