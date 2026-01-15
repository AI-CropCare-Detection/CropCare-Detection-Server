from fastapi import FastAPI, status
import keras
import uvicorn
import os
from  model import PredictRequest
from  process_new_image import processing_new_image

app = FastAPI()
model = keras.models.load_model('trained_model.h5')

# check server status
@app.get("/status")
def check_status():
    return {"status": "Server is running"}

# train model predictor route
@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(data: PredictRequest):
    path_to_image = data.path_to_image

    input_arr = processing_new_image(path_to_image)
    result = model.predict(input_arr)

    return {
        "result": result.tolist(),  # JSON-safe
        "status": "success"
    }

def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
    

if __name__ == "__main__":
    main()