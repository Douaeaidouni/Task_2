from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('current_model_checkpoint.h5')  

# Class names (update with your dataset labels)
class_names = ['Bear', 'Camel', 'Chicken', 'Elephant', 'Horse', 'Lion', 'Squirrel']

# Path for saving uploaded files
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(img_path, target_size=(128, 128)):
    """Preprocess the image to make it compatible with the model."""
    img = image.load_img(img_path, target_size=target_size)  # Resize image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file selected.')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No file selected.')

        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocess the image and make a prediction
            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_label = class_names[predicted_class]

            # Return the result to the user
            return render_template('index.html',
                                   message='Prediction successful!',
                                   image_path=f'uploads/{file.filename}',
                                   predicted_label=predicted_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
