from fastapi import FastAPI, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import keras
import uvicorn
import os
import numpy as np
import io
from PIL import Image

from class_list import class_name 
from  model import PredictRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

model = keras.models.load_model('trained_model.h5')

# check server status
@app.get("/")
def check_status():
    return {"status": "Server is running"}

# train model predictor route
@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).resize((128,128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    result_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return {
        "predicted_class": class_name[result_index],
        "confidence": round(confidence, 4)
    }
    
# def predict_disease_api(data: PredictRequest):
#     image = keras.preprocessing.image.load_img(
#         data.path_to_image  , target_size=(128, 128)
#     )
#     input_arr = keras.preprocessing.image.img_to_array(image) / 255.0
#     input_arr = np.expand_dims(input_arr, axis=0)

#     prediction = model.predict(input_arr)
#     result_index = np.argmax(prediction)
#     confidence = float(np.max(prediction))

#     return {
#         "predicted_class": class_name[result_index],
#         "confidence": round(confidence, 4)
#     }
    


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
    

if __name__ == "__main__":
    main()