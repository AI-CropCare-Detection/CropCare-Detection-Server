# ML Model API

A FastAPI-based machine learning prediction server using Keras.

## Project Structure

- `app.py` - FastAPI application entry point
- `model.py` - Pydantic models for request/response validation
- `process_new_image.py` - Image processing utilities
- `trained_model.h5` - Pre-trained Keras model
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration
- `Procfile` - Process file for deployment
- `runtime.txt` - Python version specification

## Local Development

### Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

- **GET** `/status` - Health check
- **POST** `/predict` - Make predictions
  - Request body: `{"path_to_image": "path/to/image.jpg"}`
  - Response: `{"result": [...], "status": "success"}`

## Deployment on Render

### Prerequisites

1. Push your code to GitHub
2. Create a Render account at https://render.com

### Deployment Steps

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: ml-model-api (or your preferred name)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or choose paid for production)
5. Click "Deploy"

### Alternative: Using render.yaml

If you have a `render.yaml` file in your repository:

1. Click "New +" and select "Web Service"
2. Select "Connect existing code from a Git repository"
3. Choose your repository
4. Click "Deploy" - Render will automatically use `render.yaml`

### Important Notes

- The `trained_model.h5` must be committed to your repository for Render to access it
- If the model file is too large (>100MB), consider:
  - Using Render's persistent disk
  - Downloading the model during build
  - Compressing the model
- Environment variables can be configured in Render dashboard

## Environment Variables

- `PORT` - Server port (default: 8000, set by Render automatically)
- `HOST` - Server host (default: 0.0.0.0)

## Dependencies

Key packages:
- fastapi - Web framework
- uvicorn - ASGI server
- keras - Deep learning framework
- tensorflow - ML framework
- opencv-python - Image processing
- pillow - Image handling
- pydantic - Data validation
