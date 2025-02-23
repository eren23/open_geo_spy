from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from typing import Optional, List
from datetime import datetime
import uuid
import tempfile

# TODO: Currently unused imports
# from PIL import Image
# import io
import base64

from image_analysis.analyzer import ImageAnalyzer
from geo_data.geo_interface import GeoDataInterface
from reasoning.location_resolver import LocationResolver
from config import CONFIG
import google.generativeai as genai

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
image_analyzer = ImageAnalyzer(CONFIG.OPENROUTER_API_KEY, CONFIG.APP_NAME, CONFIG.APP_URL)
geo_interface = GeoDataInterface(geonames_username=CONFIG.GEONAMES_USERNAME)
location_resolver = LocationResolver(CONFIG.OPENROUTER_API_KEY)

# Configure Gemini
if not CONFIG.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required in environment variables")

genai.configure(api_key=CONFIG.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Ensure upload directory exists
UPLOAD_DIR = os.path.join(CONFIG.IMAGE_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/locate")
async def locate_image(image: UploadFile, save_image: Optional[bool] = False, location: Optional[str] = None) -> dict:
    """
    Upload an image and get its predicted location
    Optional location parameter can be used to narrow down the search area
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
        candidates = await geo_interface.search_location_candidates(features, location)
        final_location = location_resolver.resolve_location(features, candidates, description, location)

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


@app.post("/api/analyze-multimodal")
async def analyze_multimodal(files: List[UploadFile] = File(...), save_files: Optional[bool] = Form(False), location: Optional[str] = Form(None)) -> dict:
    """
    Analyze multiple media files (images/videos) to extract location information
    Optional location parameter can be used to narrow down the search area
    """
    try:
        # Create temporary files for the uploads
        temp_files = []
        for file in files:
            # Verify file type
            if not file.content_type or not file.content_type.startswith(("image/", "video/")):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Only images and videos are supported.")

            # Save to temp file
            suffix = "." + (file.filename or "").split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                contents = await file.read()
                tmp.write(contents)
                temp_files.append((tmp.name, file.content_type))

        # Build location analysis prompt
        prompt = """
        Analyze these media files and extract all location-relevant information. Focus on:
        1. Landmarks and buildings (including names, architectural styles)
        2. Street names, addresses, and signage
        3. Geographic features and surroundings
        4. Time period indicators and architectural styles
        5. Weather conditions and time of day
        6. Vegetation and natural features
        7. Any text or symbols that could indicate location
        8. Cultural or regional specific elements
        9. Transportation infrastructure
        10. Business names and commercial signage

        For each identified feature, rate your confidence (0-1) and explain your reasoning.
        Format your response as:
        LANDMARKS: [list with confidence scores]
        SIGNAGE: [list with confidence scores]
        GEOGRAPHY: [description with confidence]
        ARCHITECTURE: [description with confidence]
        CULTURAL_INDICATORS: [list with confidence scores]
        TIME_PERIOD: [estimate with confidence]
        LOCATION_HYPOTHESIS: [your best guess of location with confidence]
        """

        # Process with Gemini
        contents = [prompt]  # Start with the prompt text

        # Add media files
        for temp_file, content_type in temp_files:
            with open(temp_file, "rb") as f:
                file_data = f.read()

            if content_type.startswith("video/"):
                mime_type = content_type
            else:
                mime_type = "image/jpeg"  # Default for images

            contents.append({"inline_data": {"mime_type": mime_type, "data": base64.b64encode(file_data).decode("utf-8")}})

        # Generate response from Gemini
        response = gemini_model.generate_content(contents)

        # Convert Gemini analysis to features format
        features = {
            "landmarks": [],
            "addresses": [],
            "architecture_style": "",
            "vegetation": [],
            "time_of_day": "",
            "weather": "",
            "geographic_features": [],
            "extracted_text": {"street_signs": [], "building_info": [], "business_names": [], "informational": [], "other": []},
            "confidence_scores": {},
        }

        # Parse Gemini response into features
        analysis_text = response.text
        sections = analysis_text.split("\n\n")
        for section in sections:
            if "LANDMARKS:" in section:
                features["landmarks"].extend([item.split("[")[0].strip("* ") for item in section.split("\n")[1:] if item.strip()])
            elif "SIGNAGE:" in section:
                for item in section.split("\n")[1:]:
                    if item.strip():
                        text = item.split("[")[0].strip("* ")
                        if "street" in text.lower():
                            features["extracted_text"]["street_signs"].append(text)
                        elif "building" in text.lower():
                            features["extracted_text"]["building_info"].append(text)
                        else:
                            features["extracted_text"]["other"].append(text)
            elif "GEOGRAPHY:" in section:
                features["geographic_features"].extend([item.split("[")[0].strip("* ") for item in section.split("\n")[1:] if item.strip()])
            elif "ARCHITECTURE:" in section:
                features["architecture_style"] = " ".join([item.split("[")[0].strip("* ") for item in section.split("\n")[1:] if item.strip()])
            elif "CULTURAL_INDICATORS:" in section:
                features["extracted_text"]["informational"].extend([item.split("[")[0].strip("* ") for item in section.split("\n")[1:] if item.strip()])
            elif "TIME_PERIOD:" in section:
                features["time_of_day"] = section.split("\n")[1].split("[")[0].strip("* ")

        # Use existing pipeline to find location
        candidates = geo_interface.search_location_candidates(features, location)
        final_location = location_resolver.resolve_location(features, candidates, analysis_text, location)

        # Save files if requested
        saved_files = []
        if save_files:
            for (temp_file, _), file in zip(temp_files, files):
                filename = (
                    f"{(file.filename or '').split('.')[0]}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                    f"{str(uuid.uuid4())[:8]}."
                    f"{(file.filename or '').split('.')[-1]}"
                )
                file_path = os.path.join(UPLOAD_DIR, filename)
                shutil.copy2(temp_file, file_path)
                saved_files.append(filename)

        # Clean up temp files
        for temp_file, _ in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

        return {
            "success": True,
            "location": final_location,
            "analysis": {"features": features, "description": analysis_text, "candidates": candidates, "gemini_analysis": response.text},
            "files_saved": saved_files if save_files else [],
        }

    except Exception as e:
        # Clean up temp files on error
        for temp_file, _ in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
