from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# load trained model
model = tf.keras.models.load_model('/Users/rohith/Cars/model.h5')

# testing endpoint
@app.route('/')
def home():
    return "App is running"


# class mapping
CLASS_NAMES = {0: 'Convertible', 1: 'Coupe', 2: 'Hatchback', 3: 'Minivan', 4: 'SUV', 5: 'Sedan', 6: 'Truck', 7: 'Van', 8: 'Wagon'}


# predictions endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        # Open the image
        image = Image.open(file.stream).convert("RGB")  # Ensure 3 channels (RGB)

        # Resize the image to (224, 224)
        image = image.resize((224, 224))

        # Convert the image to a NumPy array
        image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

        # Add the batch dimension (1, 224, 224, 3)
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])

        predicted_class = np.argmax(prediction[0])

        # Map the predicted class to its label
        class_label = CLASS_NAMES.get(predicted_class, "Unknown")

        return jsonify({
            'predicted_class': int(predicted_class),
            'class_label': class_label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

