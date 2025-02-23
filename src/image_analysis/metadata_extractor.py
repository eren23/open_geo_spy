from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, Optional
import datetime
import piexif
import requests
from io import BytesIO
import os


class MetadataExtractor:
    def __init__(self):
        self.timezone_api = "http://api.geonames.org/timezoneJSON"

    def extract_metadata(self, image_path: str) -> Dict:
        """Extract all available metadata from image"""
        try:
            if image_path.startswith(("http://", "https://")):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)

            metadata = {
                "dimensions": image.size,
                "format": image.format,
                "color_mode": image.mode,
                "exif": self._extract_exif(image),
                "creation_time": None,
                "device_info": None,
                "estimated_location": None,
                "filename": os.path.basename(image_path),
            }

            if "exif" in metadata and metadata["exif"]:
                metadata.update(self._analyze_exif_data(metadata["exif"]))

            # Debug logging
            print("\n=== Image Metadata ===")
            print(f"File: {metadata['filename']}")
            print(f"Dimensions: {metadata['dimensions']}")
            print(f"Format: {metadata['format']}")
            print(f"Color Mode: {metadata['color_mode']}")
            if metadata.get("exif"):
                print("\nEXIF Data:")
                for key, value in metadata["exif"].items():
                    print(f"  {key}: {value}")
            if metadata.get("gps_coordinates"):
                print(f"\nGPS Coordinates: {metadata['gps_coordinates']}")
            if metadata.get("timestamp"):
                print(f"Timestamp: {metadata['timestamp']}")
            if metadata.get("camera_model"):
                print(f"Camera: {metadata['camera_model']}")
            print("=====================\n")

            return metadata
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {}

    def _extract_exif(self, image: Image) -> Optional[Dict]:
        """Extract EXIF data from image"""
        try:
            exif_dict = {}
            info = image.getexif()

            if not info:
                return None

            for tag_id in info:
                tag = TAGS.get(tag_id, tag_id)
                data = info.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode()
                exif_dict[tag] = data

            # Get GPS Info
            if piexif.ImageIFD.GPSTag in info:
                gps_info = {}
                for key in GPSTAGS.keys():
                    if key in info[piexif.ImageIFD.GPSTag]:
                        gps_info[GPSTAGS[key]] = info[piexif.ImageIFD.GPSTag][key]
                exif_dict["GPSInfo"] = gps_info

            return exif_dict
        except Exception as e:
            print(f"Error extracting EXIF data: {e}")
            return None

    def _analyze_exif_data(self, exif: Dict) -> Dict:
        """Analyze EXIF data for location hints"""
        analysis = {"timestamp": None, "camera_model": None, "gps_coordinates": None, "estimated_timezone": None}

        # Extract timestamp
        if "DateTime" in exif:
            try:
                analysis["timestamp"] = datetime.datetime.strptime(exif["DateTime"], "%Y:%m:%d %H:%M:%S")
            except ValueError as e:
                print(f"Error parsing timestamp: {e}")

        # Extract camera info
        if "Make" in exif and "Model" in exif:
            analysis["camera_model"] = f"{exif['Make']} {exif['Model']}"

        # Extract GPS coordinates
        if "GPSInfo" in exif:
            try:
                gps = exif["GPSInfo"]
                if "GPSLatitude" in gps and "GPSLongitude" in gps:
                    lat = self._convert_to_degrees(gps["GPSLatitude"])
                    lon = self._convert_to_degrees(gps["GPSLongitude"])

                    if gps.get("GPSLatitudeRef", "N") == "S":
                        lat = -lat
                    if gps.get("GPSLongitudeRef", "E") == "W":
                        lon = -lon

                    analysis["gps_coordinates"] = (lat, lon)
            except:
                pass

        return analysis

    def _convert_to_degrees(self, value) -> float:
        """Convert GPS coordinates to degrees"""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
