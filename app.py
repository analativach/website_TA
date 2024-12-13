import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Tentukan folder untuk menyimpan file yang di-upload
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Fungsi untuk memeriksa ekstensi file yang valid
def allowed_file(filename):
    if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        return True
    return False

# Load the trained model
try:
    model = load_model('model/training_efficiennet.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Daftar nama kelas untuk model
class_names = ['glioma', 'meningiloma', 'notumor', 'pituitary']  # Sesuaikan dengan kelas Anda

# Pastikan folder upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        try:
            # Save the file to the 'uploads' folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load and preprocess the image
            img = image.load_img(file_path, target_size=(224, 224))  # Sesuaikan ukuran dengan model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalisasi jika diperlukan

            # Predict the class
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class = class_names[predicted_class_index]

            # Return the result
            return jsonify({"predicted_class": predicted_class})

        except Exception as e:
            return jsonify({"error": f"Error during prediction: {str(e)}"})
    else:
        return jsonify({"error": "Invalid file type"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
