from contextlib import nullcontext
from logging import log
from types import MethodType
from flask import Flask, render_template
from flask.globals import request
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

saved_model=load_model('model')
saved_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    print(request.files)
    if 'image_file' not in request.files:
        return 'there is no image attached in the form!'
    else:
        image_file = request.files['image_file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(path)
        img = cv2.imread(path)
        img = cv2.resize(img,(200,200))
        img = np.reshape(img,[1,200,200,3])
        d = saved_model.predict(img)
        if(d[0][0]>0.8):
            return render_template('braintumour.html')
        elif(d[0][0]>0.5 and d[0][0]<0.8):
            return 'You have symptoms of brain tumour'
        else:
            return render_template('nobraintumour.html')

    return nullcontext

if __name__ == '__main__':
    app.run()