from image_analysis.analyzer import ImageAnalyzer
from image_analysis.metadata_extractor import MetadataExtractor
from image_analysis.visual_search import VisualSearchEngine
from geo_data.geo_interface import GeoDataInterface
from reasoning.location_resolver import LocationResolver
from config import CONFIG
import os


class GeoLocator:
    def __init__(self):
        self.image_analyzer = ImageAnalyzer(CONFIG["openrouter_api_key"], CONFIG["app_name"], CONFIG["app_url"])
        self.metadata_extractor = MetadataExtractor()
        self.visual_search = VisualSearchEngine(google_api_key=CONFIG.get("GOOGLE_API_KEY"), bing_api_key=CONFIG.get("BING_API_KEY"))
        self.geo_interface = GeoDataInterface(geonames_username=CONFIG["geonames_username"])
        self.location_resolver = LocationResolver(CONFIG["openrouter_api_key"])

    async def locate_image(self, image_path: str):
        # 1. Extract metadata
        metadata = self.metadata_extractor.extract_metadata(image_path)
        initial_location = None
        if metadata.get("exif", {}).get("gps_coordinates"):
            initial_location = {"lat": metadata["exif"]["gps_coordinates"][0], "lon": metadata["exif"]["gps_coordinates"][1]}

        # 2. Analyze image
        features, description = self.image_analyzer.analyze_image(image_path)

        # 3. Search for candidate locations
        candidates = self.geo_interface.search_location_candidates(features)

        # 4. Find visually similar locations
        visual_matches = await self.visual_search.find_similar_locations(image_path, initial_location)

        # Combine candidates with visual matches
        candidates.extend(visual_matches)

        # 5. Reason about most likely location
        final_location = self.location_resolver.resolve_location(features, candidates, description, metadata=metadata)

        return {
            "location": final_location,
            "analysis": {"features": features, "description": description, "candidates": candidates, "metadata": metadata, "visual_matches": visual_matches},
        }


def process_images_in_directory(directory: str = "/app/images"):
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
            result = locator.locate_image(image_path)

            print("Predicted Location:")
            print(f"Name: {result['location']['name']}")
            print(f"Coordinates: {result['location']['lat']}, {result['location']['lon']}")
            print(f"Confidence: {result['location']['confidence']}")
            print("\nAnalysis:")
            print(result["analysis"]["description"])

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")


def main():
    process_images_in_directory()


if __name__ == "__main__":
    main()
