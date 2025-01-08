from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Allow requests from any origin for development (you can restrict this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins (e.g., ["http://localhost:63342"]) for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = YOLO("runs/detect/train3/weights/best.pt")


@app.post("/process-image")
async def process_image(file: UploadFile):
    # Read image from the uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform image processing and predictions (simplified for brevity)
    result = model(image)

    # Example bounding box drawing (mock implementation)
    processed_image = image.copy()
    for bbox in result[0].boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 3 )

    # Save the processed image to a file
    processed_image_path = "processed_image.jpg"
    cv2.imwrite(processed_image_path, processed_image)

    # Return the processed image and dummy RGB values
    return JSONResponse({
        "rgb_values": [[255, 0, 0], [0, 255, 0]],  # Example RGB values
        "image_download_url": f"/download/{processed_image_path}",
    })


@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(filename, media_type="image/jpeg", filename=filename)
