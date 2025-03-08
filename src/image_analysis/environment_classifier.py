from typing import Dict, List, Tuple
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum, auto


class EnvironmentType(Enum):
    URBAN = auto()
    SUBURBAN = auto()
    RURAL = auto()
    INDUSTRIAL = auto()
    AIRPORT = auto()
    COASTAL = auto()
    FOREST = auto()
    MOUNTAIN = auto()
    DESERT = auto()
    PARK = auto()
    HIGHWAY = auto()
    UNKNOWN = auto()


@dataclass
class EnvironmentInfo:
    primary_type: EnvironmentType
    secondary_types: List[EnvironmentType]
    confidence: float
    features: Dict[str, float]


class EnvironmentClassifier:
    def __init__(self):
        # Define feature weights for each environment type
        self.type_features = {
            EnvironmentType.URBAN: {"building_density": 0.3, "road_density": 0.2, "vegetation_density": -0.1, "text_density": 0.2, "vehicle_density": 0.2},
            EnvironmentType.SUBURBAN: {"building_density": 0.2, "road_density": 0.15, "vegetation_density": 0.3, "text_density": 0.1, "vehicle_density": 0.1},
            EnvironmentType.RURAL: {"building_density": -0.2, "road_density": -0.1, "vegetation_density": 0.4, "natural_features": 0.3},
            EnvironmentType.INDUSTRIAL: {"building_density": 0.3, "large_structures": 0.3, "vehicle_density": 0.2, "vegetation_density": -0.2},
            EnvironmentType.AIRPORT: {"runway_features": 0.4, "large_structures": 0.2, "open_space": 0.2, "aircraft_presence": 0.2},
            EnvironmentType.COASTAL: {"water_presence": 0.4, "beach_features": 0.3, "building_density": 0.1, "vegetation_type": 0.2},
            EnvironmentType.FOREST: {"vegetation_density": 0.5, "tree_coverage": 0.3, "natural_features": 0.2},
            EnvironmentType.MOUNTAIN: {"elevation_features": 0.4, "terrain_roughness": 0.3, "vegetation_type": 0.2},
            EnvironmentType.DESERT: {"sand_features": 0.4, "vegetation_density": -0.2, "terrain_type": 0.3},
            EnvironmentType.PARK: {"vegetation_density": 0.3, "recreational_features": 0.3, "path_density": 0.2},
            EnvironmentType.HIGHWAY: {"road_width": 0.3, "vehicle_density": 0.3, "road_markings": 0.2, "building_density": -0.1},
        }

    def classify_environment(self, image: np.ndarray, features: Dict) -> EnvironmentInfo:
        """Classify the environment type based on image analysis and extracted features"""
        # Extract environment-specific features
        env_features = self._extract_environment_features(image, features)

        # Calculate scores for each environment type
        type_scores = {}
        for env_type, weights in self.type_features.items():
            score = 0
            for feature, weight in weights.items():
                if feature in env_features:
                    score += weight * env_features[feature]
            type_scores[env_type] = max(0, min(1, score))  # Normalize to 0-1

        # Get primary and secondary types
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        primary_type = sorted_types[0][0]
        secondary_types = [t[0] for t in sorted_types[1:3] if t[1] > 0.3]  # Get next 2 types with score > 0.3

        return EnvironmentInfo(primary_type=primary_type, secondary_types=secondary_types, confidence=sorted_types[0][1], features=env_features)

    def _extract_environment_features(self, image: np.ndarray, features: Dict) -> Dict[str, float]:
        """Extract normalized feature scores for environment classification"""
        env_features = {}

        # Building density (0-1)
        if features.get("building_density"):
            env_features["building_density"] = {"high_density": 0.9, "medium_density": 0.5, "low_density": 0.2}.get(features["building_density"], 0.0)

        # Vegetation density (0-1)
        if features.get("vegetation_density"):
            env_features["vegetation_density"] = {"high": 0.9, "medium": 0.5, "low": 0.2}.get(features["vegetation_density"], 0.0)

        # Road features
        road_types = features.get("road_types", [])
        env_features["road_density"] = min(1.0, len(road_types) * 0.2)
        env_features["road_width"] = 1.0 if "major_road" in road_types else (0.6 if "street" in road_types else 0.3)

        # Water features
        water_bodies = features.get("water_bodies", [])
        env_features["water_presence"] = 1.0 if "large_water_body" in water_bodies else (0.5 if "small_water_body" in water_bodies else 0.0)

        # Terrain features
        terrain_types = features.get("terrain_type", [])
        env_features["terrain_roughness"] = 1.0 if "mountain" in terrain_types else (0.7 if "hill" in terrain_types else 0.3)
        env_features["sand_features"] = 1.0 if "sand" in terrain_types else 0.0
        env_features["beach_features"] = 1.0 if "beach" in terrain_types else 0.0

        # Text density from extracted text
        extracted_text = features.get("extracted_text", {})
        text_items = len(extracted_text.get("street_signs", [])) + len(extracted_text.get("building_info", [])) + len(extracted_text.get("business_names", []))
        env_features["text_density"] = min(1.0, text_items * 0.1)  # Normalize to 0-1

        # Detect vehicles and aircraft
        env_features["vehicle_density"] = self._detect_vehicles(image)
        env_features["aircraft_presence"] = self._detect_aircraft(image)

        # Detect runways and large structures
        env_features["runway_features"] = self._detect_runways(image)
        env_features["large_structures"] = self._detect_large_structures(image)

        # Natural features
        env_features["natural_features"] = self._analyze_natural_features(image, terrain_types)
        env_features["tree_coverage"] = self._analyze_tree_coverage(image)

        # Recreational features for parks
        env_features["recreational_features"] = self._detect_recreational_features(image)
        env_features["path_density"] = self._analyze_path_density(image)

        return env_features

    def _detect_vehicles(self, image: np.ndarray) -> float:
        """Detect presence of vehicles in the image"""
        # TODO: Implement more sophisticated vehicle detection
        # For now, use a simple edge detection and contour analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vehicle_like_objects = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 1.2 < aspect_ratio < 3.0 and 100 < w < 500:  # Typical vehicle aspect ratios
                vehicle_like_objects += 1

        return min(1.0, vehicle_like_objects * 0.1)

    def _detect_aircraft(self, image: np.ndarray) -> float:
        """Detect presence of aircraft in the image"""
        # TODO: Implement aircraft detection
        # For now, use a simple large object detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_objects = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > image.shape[0] * image.shape[1] * 0.1:  # Large objects
                large_objects += 1

        return min(1.0, large_objects * 0.2)

    def _detect_runways(self, image: np.ndarray) -> float:
        """Detect runway-like features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return 0.0

        parallel_lines = 0
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                angle1 = np.arctan2(lines[i][0][3] - lines[i][0][1], lines[i][0][2] - lines[i][0][0])
                angle2 = np.arctan2(lines[j][0][3] - lines[j][0][1], lines[j][0][2] - lines[j][0][0])
                if abs(angle1 - angle2) < 0.1:  # Nearly parallel lines
                    parallel_lines += 1

        return min(1.0, parallel_lines * 0.05)

    def _detect_large_structures(self, image: np.ndarray) -> float:
        """Detect presence of large structures"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_structures = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > image.shape[0] * image.shape[1] * 0.15:  # Very large objects
                large_structures += 1

        return min(1.0, large_structures * 0.3)

    def _analyze_natural_features(self, image: np.ndarray, terrain_types: List[str]) -> float:
        """Analyze presence of natural features"""
        # Combine terrain analysis with color-based natural feature detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define natural color ranges (green vegetation, brown earth, etc.)
        natural_colors = [((35, 50, 50), (85, 255, 255)), ((10, 50, 50), (20, 255, 255)), ((0, 0, 50), (180, 50, 200))]  # Green  # Brown  # Gray (rocks)

        natural_pixels = 0
        total_pixels = image.shape[0] * image.shape[1]

        for lower, upper in natural_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            natural_pixels += np.sum(mask > 0)

        color_score = natural_pixels / (total_pixels * 3)  # Normalize by total possible
        terrain_score = len(terrain_types) * 0.2

        return min(1.0, (color_score + terrain_score) / 2)

    def _analyze_tree_coverage(self, image: np.ndarray) -> float:
        """Analyze tree coverage in the image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green color range for trees
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Use contour analysis to identify tree-like shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tree_like_objects = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum size threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(h) / w
                if 1.5 < aspect_ratio < 4.0:  # Typical tree aspect ratios
                    tree_like_objects += 1

        return min(1.0, tree_like_objects * 0.05)

    def _detect_recreational_features(self, image: np.ndarray) -> float:
        """Detect recreational features like playgrounds, sports fields, etc."""
        # TODO: Implement more sophisticated recreational feature detection
        # For now, use color and shape analysis for common recreational features

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Look for typical playground/sports field colors
        color_ranges = [((0, 50, 50), (10, 255, 255)), ((20, 50, 50), (30, 255, 255)), ((100, 50, 50), (130, 255, 255))]  # Red  # Yellow  # Blue

        recreational_pixels = 0
        total_pixels = image.shape[0] * image.shape[1]

        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            recreational_pixels += np.sum(mask > 0)

        return min(1.0, recreational_pixels / (total_pixels * 3))

    def _analyze_path_density(self, image: np.ndarray) -> float:
        """Analyze density of paths and walkways"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use adaptive thresholding to detect paths
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Look for long, thin structures
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        path_like_objects = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:  # Minimum size threshold
                aspect_ratio = float(w) / h
                if aspect_ratio > 3.0 or aspect_ratio < 0.33:  # Long, thin objects
                    path_like_objects += 1

        return min(1.0, path_like_objects * 0.1)
