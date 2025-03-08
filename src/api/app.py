from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from typing import Optional, List, Dict
from datetime import datetime
import uuid
import tempfile
import json
import re
import base64

# TODO: Currently unused imports
# from PIL import Image
# import io

from image_analysis.analyzer import ImageAnalyzer
from geo_data.geo_interface import GeoDataInterface
from reasoning.location_resolver import LocationResolver
from config import CONFIG
import google.generativeai as genai

app = FastAPI(title="GeoLocator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
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
    Upload an image and get its predicted location with enhanced entity extraction
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

        # Enhance entity extraction with Gemini
        enhanced_features = await _enhance_entity_extraction(file_path, features, description)

        # Merge enhanced features with original features
        for key, value in enhanced_features.items():
            if key not in features or not features[key]:
                features[key] = value
            elif isinstance(features[key], list) and isinstance(value, list):
                features[key].extend([item for item in value if item not in features[key]])
            elif isinstance(features[key], dict) and isinstance(value, dict):
                features[key].update(value)

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


async def _enhance_entity_extraction(image_path: str, features: Dict, description: str) -> Dict:
    """Use Gemini to enhance entity extraction"""
    try:
        # Prepare image for Gemini
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Create prompt for entity extraction
        prompt = f"""
        Analyze this image and extract the following entities with high precision.
        Format your response using EXACTLY these categories, one per line:

        BUSINESSES:
        - List each business name exactly as it appears
        - Include confidence score in [brackets] from 0-1

        STREETS:
        - List each street name exactly as it appears
        - Include confidence score in [brackets] from 0-1

        BUILDINGS:
        - List each building number and identifier
        - Include confidence score in [brackets] from 0-1

        LANDMARKS:
        - List each landmark visible or referenced
        - Include confidence score in [brackets] from 0-1

        LOCATION_CONTEXT:
        - List any city, district, or region names mentioned
        - Include confidence score in [brackets] from 0-1

        Previous analysis found:
        {json.dumps(features, indent=2)}

        Description: {description}

        Remember to format each line as:
        - Exact text [confidence_score]
        """

        # Generate response from Gemini using the new format
        response = gemini_model.generate_content(contents=[{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]}])

        # Parse the response into our feature format
        if not response.text:
            return {}

        result = {
            "extracted_text": {
                "business_names": [],
                "street_signs": [],
                "building_info": [],
            },
            "landmarks": [],
            "entity_locations": {},
        }

        current_category = None
        for line in response.text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for category headers
            if line.endswith(":"):
                category = line[:-1].upper()
                if category in ["BUSINESSES", "STREETS", "BUILDINGS", "LANDMARKS", "LOCATION_CONTEXT"]:
                    current_category = category
                continue

            # Process items under each category
            if current_category and line.startswith("-"):
                # Extract text and confidence score
                text = line.strip("- ")
                confidence_match = re.search(r"\[(0?\.\d+)\]", text)
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                    text = text[: text.find("[")].strip()
                else:
                    confidence = 0.5  # Default confidence if not specified

                if confidence < 0.3:  # Skip low confidence items
                    continue

                # Add to appropriate category
                if current_category == "BUSINESSES":
                    result["extracted_text"]["business_names"].append(text)
                elif current_category == "STREETS":
                    result["extracted_text"]["street_signs"].append(text)
                elif current_category == "BUILDINGS":
                    result["extracted_text"]["building_info"].append(text)
                elif current_category == "LANDMARKS":
                    result["landmarks"].append(text)
                elif current_category == "LOCATION_CONTEXT":
                    # Add location context to all entities
                    for category in ["business_names", "street_signs", "building_info"]:
                        for entity in result["extracted_text"][category]:
                            result["entity_locations"][entity] = text

        return result

    except Exception as e:
        print(f"Error enhancing entity extraction: {e}")
        return {}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/analyze-multimodal")
async def analyze_multimodal(files: List[UploadFile] = File(...), save_files: Optional[bool] = Form(False), location: Optional[str] = Form(None)) -> dict:
    """
    Analyze multiple media files (images/videos) for location information
    """
    temp_files = []
    try:
        # Save uploaded files to temp directory
        for file in files:
            if not file.content_type.startswith(("image/", "video/")):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp()
            temp_files.append((temp_path, file.content_type))

            # Save file content
            with os.fdopen(temp_fd, "wb") as tmp:
                shutil.copyfileobj(file.file, tmp)

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
        contents = []
        contents.append({"text": prompt})

        # Add media files
        for temp_file, content_type in temp_files:
            with open(temp_file, "rb") as f:
                file_data = f.read()
                file_b64 = base64.b64encode(file_data).decode("utf-8")

                mime_type = content_type if content_type.startswith("video/") else "image/jpeg"
                contents.append({"inline_data": {"mime_type": mime_type, "data": file_b64}})

        # Generate response from Gemini
        response = gemini_model.generate_content({"parts": contents})

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
        candidates = await geo_interface.search_location_candidates(features, location)
        final_location = location_resolver.resolve_location(
            features=features, candidates=candidates, description=analysis_text, metadata={"location_hint": location} if location else None
        )

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
