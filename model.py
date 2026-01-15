from pydantic import BaseModel

class PredictRequest(BaseModel):
    path_to_image: str
