import tensorflow as tf
import numpy as np
import cv2
import json

# Load model once
model = tf.keras.models.load_model("trained_model.keras")

# Class names
class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


def predict_disease(image_path):
    try:
        # Load and preprocess image
        image = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(128, 128)
        )
        input_arr = tf.keras.preprocessing.image.img_to_array(image)

        # Normalize if your model was trained with rescale=1./255
        input_arr = input_arr / 255.0

        # Convert to batch
        input_arr = np.expand_dims(input_arr, axis=0)

        # Prediction
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        predicted_class = class_name[result_index]

        # JSON response
        response = {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        }

        return json.dumps(response, indent=4)

    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e)
        }
        return json.dumps(error_response, indent=4)