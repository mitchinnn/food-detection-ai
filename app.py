import numpy as np
from flask import Flask, request, jsonify, send_file, redirect, url_for
import keras
from keras.preprocessing import image
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return send_file('templates/classification.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = keras.models.load_model("final-food-model.h5")
    specific_classes = ['baby_back_ribs', 'baklava', 'beef_carpaccio', 'bruschetta',
                        'beet_salad', 'beignets', 'breakfast_burrito', 'donat', 'churros', 'fried_rice']

    if 'file' not in request.files:
        return jsonify({'error': 'Berkas tidak ditemukan'}), 400

    file = request.files['file']
    img = image.load_img(io.BytesIO(file.read()), target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    if predicted_class_index < len(specific_classes):
        predicted_class = specific_classes[predicted_class_index]
        return jsonify({'predicted_class': predicted_class})
    else:
        return jsonify({'error': 'Gambar tidak cocok dengan kelas yang diharapkan. Silakan coba dengan gambar lain.'}), 400

if __name__ == '__main__':
    app.run()
