"""EXIF/metadata extraction, cleaned up from original metadata_extractor.py.

Keeps: EXIF GPS extraction, DMS conversion, image hashing, video metadata.
Removes: bare excepts, print-based logging.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import mimetypes
import os
from io import BytesIO
from typing import Any, Optional

import numpy as np
from loguru import logger
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from src.evidence.chain import Evidence, EvidenceSource


class MetadataExtractor:
    """Extract EXIF, file system, and image content metadata."""

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

    def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract all available metadata from image or video file."""
        try:
            file_metadata = self._get_file_metadata(file_path)

            if self._is_video_file(file_path):
                return self._extract_video_metadata(file_path, file_metadata)

            image = Image.open(file_path)

            metadata: dict[str, Any] = {
                **file_metadata,
                "media_type": "image",
                "dimensions": image.size,
                "format": image.format,
                "color_mode": image.mode,
                "exif": self._extract_exif(image),
                "creation_time": None,
                "gps_coordinates": None,
                "camera_model": None,
            }

            if metadata["exif"]:
                exif_analysis = self._analyze_exif_data(metadata["exif"])
                metadata.update(exif_analysis)
                metadata["camera_settings"] = self._extract_camera_settings(metadata["exif"])

            metadata["image_hash"] = self._calculate_image_hash(image)
            metadata["content_analysis"] = self._analyze_image_content(image)

            return metadata

        except Exception as e:
            logger.error("Error extracting metadata from {}: {}", file_path, e)
            return {"filename": os.path.basename(file_path), "error": str(e)}

    def to_evidence(self, metadata: dict) -> list[Evidence]:
        """Convert metadata to Evidence objects."""
        evidences = []

        # GPS coordinates from EXIF
        gps = metadata.get("gps_coordinates")
        if gps and len(gps) == 2:
            evidences.append(
                Evidence(
                    source=EvidenceSource.EXIF,
                    content=f"EXIF GPS coordinates: {gps[0]:.6f}, {gps[1]:.6f}",
                    confidence=0.95,  # EXIF GPS is highly reliable when present
                    latitude=gps[0],
                    longitude=gps[1],
                    metadata={"camera_model": metadata.get("camera_model", "")},
                )
            )

        # Timestamp can provide timezone hints
        ts = metadata.get("timestamp")
        if ts:
            evidences.append(
                Evidence(
                    source=EvidenceSource.EXIF,
                    content=f"Image captured at {ts}",
                    confidence=0.5,
                    metadata={"timestamp": str(ts)},
                )
            )

        return evidences

    # --- Private methods ---

    def _get_file_metadata(self, file_path: str) -> dict[str, Any]:
        try:
            stat = os.stat(file_path)
            return {
                "filename": os.path.basename(file_path),
                "file_size": stat.st_size,
                "file_type": mimetypes.guess_type(file_path)[0] or "unknown",
                "last_modified": datetime.datetime.fromtimestamp(stat.st_mtime),
            }
        except OSError as e:
            logger.warning("Could not read file metadata: {}", e)
            return {"filename": os.path.basename(file_path)}

    def _extract_exif(self, image: Image.Image) -> Optional[dict]:
        try:
            exif_dict = {}
            info = image.getexif()
            if not info:
                return None

            for tag_id in info:
                tag = TAGS.get(tag_id, tag_id)
                data = info.get(tag_id)
                if isinstance(data, bytes):
                    try:
                        data = data.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                exif_dict[tag] = data

            # GPS Info
            try:
                import piexif

                if piexif.ImageIFD.GPSTag in info:
                    gps_info = {}
                    for key in GPSTAGS:
                        if key in info[piexif.ImageIFD.GPSTag]:
                            gps_info[GPSTAGS[key]] = info[piexif.ImageIFD.GPSTag][key]
                    exif_dict["GPSInfo"] = gps_info
            except (ImportError, KeyError, TypeError):
                pass

            return exif_dict
        except Exception as e:
            logger.debug("EXIF extraction failed: {}", e)
            return None

    def _analyze_exif_data(self, exif: dict) -> dict[str, Any]:
        analysis: dict[str, Any] = {
            "timestamp": None,
            "camera_model": None,
            "gps_coordinates": None,
        }

        # Timestamp
        if "DateTime" in exif:
            try:
                analysis["timestamp"] = datetime.datetime.strptime(exif["DateTime"], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass

        # Camera
        make = exif.get("Make", "")
        model = exif.get("Model", "")
        if make or model:
            analysis["camera_model"] = f"{make} {model}".strip()

        # GPS
        gps = exif.get("GPSInfo")
        if gps and "GPSLatitude" in gps and "GPSLongitude" in gps:
            try:
                lat = self._convert_to_degrees(gps["GPSLatitude"])
                lon = self._convert_to_degrees(gps["GPSLongitude"])
                if gps.get("GPSLatitudeRef", "N") == "S":
                    lat = -lat
                if gps.get("GPSLongitudeRef", "E") == "W":
                    lon = -lon
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    analysis["gps_coordinates"] = (lat, lon)
            except (TypeError, ValueError, ZeroDivisionError) as e:
                logger.debug("GPS conversion failed: {}", e)

        return analysis

    def _convert_to_degrees(self, value: tuple) -> float:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)

    def _extract_camera_settings(self, exif: dict) -> dict[str, Any]:
        tag_map = {
            "FNumber": "f_stop",
            "ExposureTime": "exposure_time",
            "ISOSpeedRatings": "iso",
            "FocalLength": "focal_length",
            "Flash": "flash",
            "WhiteBalance": "white_balance",
        }
        return {name: exif[tag] for tag, name in tag_map.items() if tag in exif}

    def _calculate_image_hash(self, image: Image.Image) -> str:
        try:
            small = image.convert("L").resize((8, 8), Image.LANCZOS)
            pixels = list(small.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if p > avg else "0" for p in pixels)
            return hex(int(bits, 2))[2:].zfill(16)
        except Exception:
            return ""

    def _analyze_image_content(self, image: Image.Image) -> dict[str, Any]:
        try:
            arr = np.array(image)
            analysis = {
                "mean_brightness": float(np.mean(arr)),
                "is_grayscale": len(arr.shape) == 2 or (len(arr.shape) == 3 and arr.shape[2] == 1),
            }
            if len(arr.shape) == 3:
                analysis["channel_means"] = [float(np.mean(arr[:, :, i])) for i in range(arr.shape[2])]
            return analysis
        except Exception:
            return {}

    def _is_video_file(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        mime = mimetypes.guess_type(file_path)[0]
        return ext in self.VIDEO_EXTENSIONS or (mime is not None and mime.startswith("video/"))

    def _extract_video_metadata(self, file_path: str, file_metadata: dict) -> dict[str, Any]:
        try:
            import cv2

            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return {**file_metadata, "media_type": "video", "error": "Could not open video"}

            fps = cap.get(cv2.CAP_PROP_FPS) or 1
            metadata = {
                **file_metadata,
                "media_type": "video",
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps,
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": fps,
                "dimensions": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            }
            cap.release()
            return metadata
        except Exception as e:
            logger.error("Video metadata extraction failed: {}", e)
            return {**file_metadata, "media_type": "video"}
