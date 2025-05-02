from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- IMAGE CLASSIFICATION SETUP ----------
image_model_path = "vit_model.onnx"
image_session = ort.InferenceSession(image_model_path)
input_name = image_session.get_inputs()[0].name
output_name = image_session.get_outputs()[0].name
CLASS_LABELS = ["Invoice", "Budget", "Advertisement", "Invoice", "Budget", "Advertisement"]

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = preprocess_image(image)
        output = image_session.run([output_name], {input_name: image})[0][0]
        probabilities = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))
        predicted_class = np.argmax(probabilities)
        return JSONResponse(content={
            "prediction": CLASS_LABELS[predicted_class],
            "confidence": round(float(probabilities[predicted_class]), 2),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- ENTRY POINT FOR UVICORN ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
