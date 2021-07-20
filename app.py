import os
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
from flask import Flask,render_template,Response,request
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder="./templates",static_folder='./static')
face_classifier = cv2.CascadeClassifier('./model/stream_faceclassifier.xml')
classifier =load_model('./model/model.h5')
class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
# Model saved with Keras model.save()
MODEL_PATH = './model/model.h5'
# Load your trained model
model = load_model(MODEL_PATH,compile=False)
ds_factor=0.6
@app.route('/live')
def live():
    return render_template('live.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()


    def get_frame(self):

        ret,frame = self.video.read()
        frame = cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_rects = face_classifier.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                preds = classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position = (x,y)  
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            break

        ret,jpeg = cv2.imencode('.jpg',frame)
        return jpeg.tobytes()


def model_predict(img_path, model):
    class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    test_img = cv2.imread(img_path,0)
    test_img = cv2.resize(test_img, (48,48))
    test_img = cv2.resize(test_img, (48,48))
    if np.sum([test_img])!=0:
        roi = test_img.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)
    preds = model.predict(roi)[0]
    label=class_labels[preds.argmax()]
    return label
 
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
@app.route('/home', methods=['GET'])
def home():
    # Main page
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'upload/',f.filename)
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        return preds
    return None
if __name__ == '__main__':
    app.run()
