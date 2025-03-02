from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, Optional, List, Tuple
import datetime
import piexif
import requests
from io import BytesIO
import os
import cv2
import base64
import numpy as np
import mimetypes


class MetadataExtractor:
    def __init__(self):
        self.timezone_api = "http://api.geonames.org/timezoneJSON"

    def extract_metadata(self, file_path: str) -> Dict:
        """Extract all available metadata from image or video file"""
        try:
            # Get file system metadata first
            file_metadata = self._get_file_metadata(file_path)

            if file_path.startswith(("http://", "https://")):
                response = requests.get(file_path)
                content = BytesIO(response.content)
                is_video = self._is_video_file(file_path)
                if is_video:
                    return self._extract_video_metadata(content, file_metadata)
                image = Image.open(content)
            else:
                is_video = self._is_video_file(file_path)
                if is_video:
                    return self._extract_video_metadata(file_path, file_metadata)
                image = Image.open(file_path)

            # Extract image metadata
            metadata = {
                **file_metadata,  # Include file system metadata
                "media_type": "image",
                "dimensions": image.size,
                "format": image.format,
                "color_mode": image.mode,
                "color_profile": self._get_color_profile(image),
                "bit_depth": self._get_bit_depth(image),
                "compression": self._get_compression_info(image),
                "exif": self._extract_exif(image),
                "creation_time": None,
                "device_info": None,
                "estimated_location": None,
            }

            # Extract additional EXIF data if available
            if metadata["exif"]:
                exif_analysis = self._analyze_exif_data(metadata["exif"])
                metadata.update(exif_analysis)

                # Extract advanced camera settings
                metadata["camera_settings"] = self._extract_camera_settings(metadata["exif"])

            # Calculate image hash for duplicate detection
            metadata["image_hash"] = self._calculate_image_hash(image)

            # Analyze image content
            metadata["content_analysis"] = self._analyze_image_content(image)

            self._log_metadata(metadata)
            return metadata

        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {}

    def _get_file_metadata(self, file_path: str) -> Dict:
        """Get file system metadata"""
        try:
            if file_path.startswith(("http://", "https://")):
                return {"filename": os.path.basename(file_path), "file_size": None, "file_type": self._get_file_type(file_path), "last_modified": None}

            stat = os.stat(file_path)
            return {
                "filename": os.path.basename(file_path),
                "file_size": stat.st_size,
                "file_type": self._get_file_type(file_path),
                "creation_time": datetime.datetime.fromtimestamp(stat.st_ctime),
                "last_modified": datetime.datetime.fromtimestamp(stat.st_mtime),
                "last_accessed": datetime.datetime.fromtimestamp(stat.st_atime),
                "file_permissions": stat.st_mode,
                "file_owner": stat.st_uid,
                "file_group": stat.st_gid,
            }
        except Exception as e:
            print(f"Error getting file metadata: {e}")
            return {}

    def _extract_video_metadata(self, file_path: str, file_metadata: Dict) -> Dict:
        """Extract metadata from video file"""
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            metadata = {
                **file_metadata,
                "media_type": "video",
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "dimensions": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                "video_codec": self._get_codec_info(cap),
                "audio_streams": self._get_audio_info(file_path),
                "bitrate": self._get_video_bitrate(file_path),
            }

            # Extract thumbnail
            ret, frame = cap.read()
            if ret:
                metadata["thumbnail"] = self._create_thumbnail(frame)
                # Try to extract any metadata from first frame
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_exif = self._extract_exif(pil_image)
                if frame_exif:
                    metadata["frame_metadata"] = self._analyze_exif_data(frame_exif)

            cap.release()
            return metadata

        except Exception as e:
            print(f"Error extracting video metadata: {e}")
            return file_metadata

    def _get_codec_info(self, cap) -> str:
        """Get video codec information"""
        try:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        except:
            return "unknown"

    def _get_audio_info(self, file_path: str) -> List[Dict]:
        """Get audio stream information using ffprobe"""
        try:
            import ffmpeg

            probe = ffmpeg.probe(file_path)
            return [stream for stream in probe["streams"] if stream["codec_type"] == "audio"]
        except:
            return []

    def _get_video_bitrate(self, file_path: str) -> Optional[int]:
        """Get video bitrate using ffprobe"""
        try:
            import ffmpeg

            probe = ffmpeg.probe(file_path)
            return int(probe["format"]["bit_rate"])
        except:
            return None

    def _create_thumbnail(self, frame: np.ndarray, size: Tuple[int, int] = (320, 240)) -> str:
        """Create base64 thumbnail from video frame"""
        try:
            # Resize frame
            h, w = frame.shape[:2]
            aspect = w / h
            if w > h:
                new_w = size[0]
                new_h = int(new_w / aspect)
            else:
                new_h = size[1]
                new_w = int(new_h * aspect)
            thumbnail = cv2.resize(frame, (new_w, new_h))

            # Convert to base64
            _, buffer = cv2.imencode(".jpg", thumbnail)
            return base64.b64encode(buffer).decode()
        except:
            return ""

    def _get_color_profile(self, image: Image) -> Optional[str]:
        """Get image color profile information"""
        try:
            if "icc_profile" in image.info:
                return "ICC profile present"
            return None
        except:
            return None

    def _get_bit_depth(self, image: Image) -> Optional[int]:
        """Get image bit depth"""
        try:
            return image.bits if hasattr(image, "bits") else None
        except:
            return None

    def _get_compression_info(self, image: Image) -> Optional[str]:
        """Get image compression information"""
        try:
            return image.info.get("compression", None)
        except:
            return None

    def _calculate_image_hash(self, image: Image) -> str:
        """Calculate perceptual hash of image for duplicate detection"""
        try:
            # Convert to grayscale and resize to 8x8
            small_image = image.convert("L").resize((8, 8), Image.LANCZOS)
            pixels = list(small_image.getdata())
            avg = sum(pixels) / len(pixels)
            # Create binary hash
            bits = "".join(["1" if pixel > avg else "0" for pixel in pixels])
            return hex(int(bits, 2))[2:].zfill(16)
        except:
            return ""

    def _analyze_image_content(self, image: Image) -> Dict:
        """Analyze basic image content characteristics"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)

            # Calculate basic statistics
            analysis = {
                "mean_brightness": float(np.mean(img_array)),
                "std_brightness": float(np.std(img_array)),
                "is_grayscale": len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1),
            }

            if len(img_array.shape) == 3:
                analysis["channel_means"] = [float(np.mean(img_array[:, :, i])) for i in range(img_array.shape[2])]

            # Detect if image might be a screenshot
            analysis["might_be_screenshot"] = self._detect_screenshot(img_array)

            return analysis
        except:
            return {}

    def _detect_screenshot(self, img_array: np.ndarray) -> bool:
        """Detect if image might be a screenshot based on characteristics"""
        try:
            # Screenshots often have very uniform edges
            edges = np.mean(np.abs(np.diff(img_array[:, 0]))) + np.mean(np.abs(np.diff(img_array[0, :])))
            return edges < 10  # Arbitrary threshold
        except:
            return False

    def _get_file_type(self, file_path: str) -> str:
        """Get detailed file type information"""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "unknown"

    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a video based on extension or mime type"""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
        ext = os.path.splitext(file_path)[1].lower()
        mime_type = self._get_file_type(file_path)
        return ext in video_extensions or (mime_type and mime_type.startswith("video/"))

    def _extract_camera_settings(self, exif: Dict) -> Dict:
        """Extract detailed camera settings from EXIF data"""
        settings = {}

        # Map of EXIF tags to human-readable names
        setting_tags = {
            "FNumber": "f_stop",
            "ExposureTime": "exposure_time",
            "ISOSpeedRatings": "iso",
            "FocalLength": "focal_length",
            "ExposureProgram": "exposure_program",
            "MeteringMode": "metering_mode",
            "Flash": "flash",
            "WhiteBalance": "white_balance",
        }

        for tag, name in setting_tags.items():
            if tag in exif:
                settings[name] = exif[tag]

        return settings

    def _log_metadata(self, metadata: Dict) -> None:
        """Log extracted metadata in a structured format"""
        print("\n=== Detailed Metadata Analysis ===")

        # File information
        print("\nFile Information:")
        for key in ["filename", "file_size", "file_type", "media_type"]:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")

        # Media properties
        print("\nMedia Properties:")
        for key in ["dimensions", "format", "color_mode", "bit_depth"]:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")

        # Temporal information
        print("\nTemporal Information:")
        for key in ["creation_time", "last_modified", "last_accessed"]:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")

        # Camera information
        if "camera_settings" in metadata:
            print("\nCamera Settings:")
            for key, value in metadata["camera_settings"].items():
                print(f"  {key}: {value}")

        # Location information
        if "gps_coordinates" in metadata:
            print("\nLocation Information:")
            print(f"  GPS Coordinates: {metadata['gps_coordinates']}")

        # Content analysis
        if "content_analysis" in metadata:
            print("\nContent Analysis:")
            for key, value in metadata["content_analysis"].items():
                print(f"  {key}: {value}")

        print("===============================\n")

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
