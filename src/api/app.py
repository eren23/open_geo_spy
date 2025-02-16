from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from typing import Optional
from datetime import datetime
import uuid

from image_analysis.analyzer import ImageAnalyzer
from geo_data.geo_interface import GeoDataInterface
from reasoning.location_resolver import LocationResolver
from config import CONFIG

app = FastAPI(title="GeoLocator API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the GeoLocator components
image_analyzer = ImageAnalyzer(CONFIG["openrouter_api_key"], CONFIG["app_name"], CONFIG["app_url"])
geo_interface = GeoDataInterface(geonames_username=CONFIG["geonames_username"])
location_resolver = LocationResolver(CONFIG["openrouter_api_key"])

# Ensure upload directory exists
UPLOAD_DIR = os.path.join(os.getenv("IMAGES_DIR", "/app/images"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/locate")
async def locate_image(image: UploadFile, save_image: Optional[bool] = False) -> dict:
    """
    Upload an image and get its predicted location
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_extension = os.path.splitext(image.filename)[1]
    filename = f"upload_{timestamp}_{unique_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Process the image
        features, description = image_analyzer.analyze_image(file_path)
        candidates = geo_interface.search_location_candidates(features)
        final_location = location_resolver.resolve_location(features, candidates, description)

        # Clean up if not saving
        if not save_image:
            os.remove(file_path)

        return {
            "success": True,
            "location": final_location,
            "analysis": {"features": features, "description": description, "candidates": candidates},
            "image_saved": save_image,
            "filename": filename if save_image else None,
        }

    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
