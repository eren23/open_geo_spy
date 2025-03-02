from image_analysis import ImageAnalyzer, MetadataExtractor
from image_analysis.visual_search import VisualSearchEngine
from geo_data import GeoDataInterface
from reasoning import LocationResolver
from config import CONFIG
import os
import asyncio
from PIL import Image
from models.osv5m_predictor import OSV5MPredictor
import re
from typing import Dict, Optional


class GeoLocator:
    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.image_analyzer = ImageAnalyzer(CONFIG.OPENROUTER_API_KEY, CONFIG.APP_NAME, CONFIG.APP_URL)
        self.visual_search = VisualSearchEngine(google_api_key=CONFIG.GOOGLE_API_KEY, bing_api_key=CONFIG.BING_API_KEY)
        self.geo_interface = GeoDataInterface(geonames_username=CONFIG.GEONAMES_USERNAME)
        self.location_resolver = LocationResolver(CONFIG.OPENROUTER_API_KEY)
        self.osv5m_predictor = OSV5MPredictor()

    async def process_image(self, image_path: str, location_hint: str = None):
        """Full processing pipeline with enhanced entity extraction"""
        print("\n=== Starting Image Processing ===")
        print(f"Image: {image_path}")
        if location_hint:
            print(f"Location Hint: {location_hint}")

        # Extract metadata
        metadata = self.metadata_extractor.extract_metadata(image_path)
        metadata["location_hint"] = location_hint  # Add location hint to metadata
        initial_location = self._get_initial_location(metadata)
        if initial_location:
            print("\nInitial Location from Metadata:")
            print(f"Coordinates: {initial_location['lat']}, {initial_location['lon']}")

        # Get OSV5M prediction
        try:
            print("\n=== Attempting OSV5M Prediction ===")
            image = Image.open(image_path)
            osv5m_result, osv5m_confidence = self.osv5m_predictor.predict(image)
            if osv5m_result:
                print("\nOSV5M Model Prediction:")
                print(f"Location: {osv5m_result['name']}")
                print(f"Coordinates: {osv5m_result['lat']}, {osv5m_result['lon']}")
                print(f"Confidence: {osv5m_confidence}")

                # Extract location context from OSV5M prediction
                if osv5m_result.get("metadata", {}).get("city"):
                    initial_location_context = osv5m_result["metadata"]["city"]
                    print(f"Location context from OSV5M: {initial_location_context}")
            else:
                print("! No OSV5M prediction available")
                osv5m_result = None
        except Exception as e:
            print(f"! Error getting OSV5M prediction: {e}")
            osv5m_result = None

        # Analyze image with VLM
        print("\n=== Starting VLM Analysis ===")
        features, description = self.image_analyzer.analyze_image(image_path)
        print("✓ VLM Analysis complete")

        # Add location hint to features
        if location_hint:
            features["location_hint"] = location_hint

        # Extract location context from features
        location_context = self._extract_location_context(features, description)
        if location_hint and not location_context:
            location_context = location_hint
        elif location_hint:
            # Combine location hint with context if they're different
            if location_hint.lower() not in location_context.lower():
                location_context = f"{location_context} ({location_hint})"

        if location_context:
            print(f"✓ Extracted location context: {location_context}")
            features["location_context"] = location_context

        # Get location candidates
        print("\n=== Getting Location Candidates ===")
        candidates = await self.geo_interface.search_location_candidates(features, location_hint=location_context, metadata=metadata)
        print(f"✓ Found {len(candidates)} initial candidates")

        # Add OSV5M prediction to candidates if available
        if osv5m_result:
            osv5m_result["confidence"] = osv5m_confidence
            candidates.append(osv5m_result)
            print("✓ Added OSV5M prediction to candidates")

        # Resolve final location with weighted consideration
        print("\n=== Resolving Final Location ===")
        result = self.location_resolver.resolve_location(
            features=features,
            candidates=candidates,
            description=description,
            metadata=metadata,
            osv5m_prediction=osv5m_result if osv5m_result and osv5m_confidence > 0.7 else None,
        )

        print("\n=== Final Result ===")
        print(f"Location: {result['name']}")
        print(f"Coordinates: {result['lat']}, {result['lon']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Source: {result.get('source', 'unknown')}")
        if result.get("reasoning"):
            print(f"Reasoning: {result['reasoning']}")
        print("===========================\n")

        return result

    def _get_initial_location(self, metadata: dict) -> dict:
        """Extract initial location from metadata"""
        if metadata.get("exif", {}).get("gps_coordinates"):
            lat, lon = metadata["exif"]["gps_coordinates"]
            return {"lat": lat, "lon": lon}
        return None

    async def _get_location_candidates(self, features: dict, initial_location: dict) -> list:
        """Get combined location candidates"""
        candidates = self.geo_interface.search_location_candidates(features)
        visual_matches = await self.visual_search.find_similar_locations(features, initial_location)
        return candidates + visual_matches

    def _extract_location_context(self, features: Dict, description: str) -> Optional[str]:
        """Extract location context from features and description"""
        # Check for license plate region information
        for plate_info in features.get("extracted_text", {}).get("license_plate_info", []):
            if plate_info.get("region_name"):
                return plate_info["region_name"]

        # Check for city names in business names
        for business in features.get("extracted_text", {}).get("business_names", []):
            # Look for city names after commas
            city_match = re.search(r",\s*([A-Z][a-zA-Z\s]+)(?:,|\.|$)", business)
            if city_match:
                return city_match.group(1).strip()

        # Check for city names in description
        city_patterns = [
            r"in ([A-Z][a-zA-Z\s]+)(?:,|\.|$)",
            r"at ([A-Z][a-zA-Z\s]+)(?:,|\.|$)",
            r"near ([A-Z][a-zA-Z\s]+)(?:,|\.|$)",
            r"(?:city|town|village|district) of ([A-Z][a-zA-Z\s]+)(?:,|\.|$)",
        ]

        for pattern in city_patterns:
            match = re.search(pattern, description)
            if match:
                return match.group(1).strip()

        return None


async def process_images_in_directory(directory: str = "/app/images"):
    """Process all images in the specified directory"""
    locator = GeoLocator()

    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Creating images directory: {directory}")
        os.makedirs(directory)
        return

    # Get all image files
    image_files = [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    if not image_files:
        print(f"No images found in {directory}")
        return

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        print(f"\nProcessing image: {image_file}")

        try:
            result = await locator.process_image(image_path)

            print("Predicted Location:")
            print(f"Name: {result['name']}")
            print(f"Coordinates: {result['lat']}, {result['lon']}")
            print(f"Confidence: {result['confidence']}")
            print("\nAnalysis:")
            print(result["analysis"])

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")


def main():
    asyncio.run(process_images_in_directory())


if __name__ == "__main__":
    main()
