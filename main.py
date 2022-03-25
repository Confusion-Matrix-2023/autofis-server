import os
from flask import Flask, request, json
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask('app')

UPLOAD_FOLDER = '/home/runner/autofis-server/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

d = {
    0: 'Black Sea Sprat',
    1: 'Gilt-Head Bream',
    2: 'Hourse Mackerel',
    3: 'Red Mullet',
    4: 'Red Sea Bream',
    5: 'Sea Bass',
    6: 'Shrimp',
    7: 'Striped Red Mullet',
    8: 'Trout'
}

print(tf.version.VERSION)

fmodel = tf.keras.models.load_model('/home/runner/autofis-server/model/best_fish_model.h5')

def prepare_img(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

@app.route('/')
def hello_world():
  response = app.response_class(
        response=json.dumps([]),
        status=200,
        mimetype='application/json'
    )
  return response;

@app.route('/api/upload', methods=['POST'])
def upload_file():
  file = request.files['file']
  filename = secure_filename(file.filename)
  file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

  img = prepare_img(UPLOAD_FOLDER + '/' + filename)

  results = fmodel.predict(img)

  hashMap = {}
  
  for i in range(len(results[0])):
    hashMap[d[i]] = results[0][i]

  sortedHashMap = sorted(hashMap.items(), key = lambda x: x[1], reverse=True)

  print(sortedHashMap)

  response = app.response_class(
        response=json.dumps({"message": "Successful"}),
        status=200,
        mimetype='application/json'
    )
  return response;
  

app.run(host='0.0.0.0', port=8080)