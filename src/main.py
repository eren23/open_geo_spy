from image_analysis import ImageAnalyzer, MetadataExtractor
from image_analysis.visual_search import VisualSearchEngine
from geo_data import GeoDataInterface
from reasoning import LocationResolver
from config import CONFIG
import os
import asyncio
from PIL import Image
from models.osv5m_predictor import OSV5MPredictor


class GeoLocator:
    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.image_analyzer = ImageAnalyzer(CONFIG.OPENROUTER_API_KEY, CONFIG.APP_NAME, CONFIG.APP_URL)
        self.visual_search = VisualSearchEngine(google_api_key=CONFIG.GOOGLE_API_KEY, bing_api_key=CONFIG.BING_API_KEY)
        self.geo_interface = GeoDataInterface(geonames_username=CONFIG.GEONAMES_USERNAME)
        self.location_resolver = LocationResolver(CONFIG.OPENROUTER_API_KEY)
        self.osv5m_predictor = OSV5MPredictor()

    async def process_image(self, image_path: str):
        """Full processing pipeline with enhanced model predictions"""
        print("\n=== Starting Image Processing ===")
        print(f"Image: {image_path}")

        # Extract metadata
        metadata = self.metadata_extractor.extract_metadata(image_path)
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
            else:
                print("! No OSV5M prediction available")
        except Exception as e:
            print(f"! Error getting OSV5M prediction: {e}")
            osv5m_result = None

        # Analyze image with VLM
        print("\n=== Starting VLM Analysis ===")
        features, description = self.image_analyzer.analyze_image(image_path)
        print("✓ VLM Analysis complete")

        # Get location candidates
        print("\n=== Getting Location Candidates ===")
        candidates = await self._get_location_candidates(features, initial_location)
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
