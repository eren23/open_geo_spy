from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
import cv2
from openai import OpenAI
import base64
import requests
import io
from math import ceil


class ImageAnalyzer:
    def __init__(self, openrouter_api_key: str, app_name: str, app_url: str):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
        self.headers = {"HTTP-Referer": app_url, "X-Title": app_name}
        self.chunk_size = (800, 800)  # Size of each chunk
        self.overlap = 100  # Overlap between chunks in pixels

        # Add new analysis categories
        self.analysis_categories = {
            "driving_side": None,  # Left/Right
            "camera_generation": None,  # Gen 1-4 or "shitcam"
            "camera_height": None,  # Normal/Low
            "season": None,
            "time_of_day": None,
            "weather": None,
            "infrastructure": {
                "road_type": None,  # Concrete/Asphalt/Dirt etc
                "road_lines": None,  # Color and pattern
                "bollards": [],  # List of detected bollard types
                "utility_poles": [],  # List of detected pole types
                "signs": [],  # Traffic/Street signs detected
            },
            "environment": {
                "terrain_type": [],
                "vegetation_type": [],
                "soil_color": None,
                "building_density": None,
                "architecture_style": None,
            },
        }

        # Add infrastructure detection parameters
        self.road_hsv_ranges = {
            "asphalt": [(0, 0, 0), (180, 30, 80)],  # Dark gray
            "concrete": [(0, 0, 150), (180, 30, 255)],  # Light gray
            "dirt": [(20, 20, 20), (30, 255, 200)],  # Brown
        }

        self.line_colors = {"white": [(0, 0, 200), (180, 30, 255)], "yellow": [(20, 100, 100), (30, 255, 255)]}

    def analyze_image(self, image_path: str) -> Tuple[Dict, str]:
        """Enhanced image analysis incorporating GeoGuessr techniques"""
        # Load and prepare image
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path)
            image = Image.open(io.BytesIO(response.content))
            cv_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = Image.open(image_path)
            cv_image = cv2.imread(image_path)

        # Analyze camera characteristics
        self._analyze_camera_meta(cv_image)

        # Analyze driving side
        self._analyze_driving_side(cv_image)

        # Analyze environment
        self._analyze_environment(cv_image)

        # Analyze infrastructure
        self._analyze_infrastructure(cv_image)

        # Use VLM for high-level analysis
        vlm_features = self._analyze_with_vlm(image)

        # Combine all analyses
        features = {**self.analysis_categories, **vlm_features}

        return features, self._generate_analysis_description(features)

    def _create_image_chunks(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Create overlapping chunks of the image"""
        height, width = image.shape[:2]
        chunks = []

        for y in range(0, height, self.chunk_size[1] - self.overlap):
            for x in range(0, width, self.chunk_size[0] - self.overlap):
                # Calculate chunk boundaries
                x2 = min(x + self.chunk_size[0], width)
                y2 = min(y + self.chunk_size[1], height)

                # Extract chunk
                chunk = image[y:y2, x:x2]

                # Only include chunks that are large enough
                if chunk.shape[0] > 100 and chunk.shape[1] > 100:
                    chunks.append((chunk, (x, y, x2, y2)))

        return chunks

    def _extract_text_from_chunk(self, chunk: np.ndarray, full_image: Image, coords: Tuple[int, int, int, int]) -> str:
        """Extract text from an image chunk using Qwen-VL"""
        try:
            # Convert chunk to base64
            chunk_pil = Image.fromarray(cv2.cvtColor(chunk, cv2.COLOR_BGR2RGB))
            buffered = io.BytesIO()
            chunk_pil.save(buffered, format="JPEG")
            chunk_base64 = base64.b64encode(buffered.getvalue()).decode()
            chunk_url = f"data:image/jpeg;base64,{chunk_base64}"

            # Convert full image for context
            full_buffered = io.BytesIO()
            full_image.save(full_buffered, format="JPEG")
            full_base64 = base64.b64encode(full_buffered.getvalue()).decode()
            full_url = f"data:image/jpeg;base64,{full_base64}"

            # Create prompt with chunk and full image context
            prompt = f"""
            I'm showing you a section of a larger image (coordinates: {coords}).
            Focus on finding and reading any text in this section.
            Look for:
            1. Signs and labels
            2. Street names
            3. Building numbers
            4. Business names
            5. Any other text

            List all text you can read, with confidence levels (0-1).
            Format: "Text: [text] (confidence: [0-1])"
            """

            completion = self.client.chat.completions.create(
                model="qwen/qwen-vl-plus:free",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": chunk_url}},
                            {"type": "image_url", "image_url": {"url": full_url}},
                        ],
                    }
                ],
                extra_headers=self.headers,
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error processing chunk at {coords}: {e}")
            return ""

    def _combine_chunk_results(self, chunk_results: List[str]) -> str:
        """Combine and deduplicate text findings from chunks"""
        text_findings = {"STREET_SIGNS": set(), "BUILDING_INFO": set(), "BUSINESS_NAMES": set(), "INFORMATIONAL": set(), "OTHER_TEXT": set()}

        for result in chunk_results:
            for line in result.split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue

                text, confidence = self._parse_text_line(line)
                if text and confidence > 0.5:  # Only include high confidence findings
                    # Categorize the text
                    if any(word in text.lower() for word in ["street", "st.", "avenue", "ave", "road", "rd"]):
                        text_findings["STREET_SIGNS"].add(text)
                    elif any(word in text.lower() for word in ["building", "#", "floor", "suite"]):
                        text_findings["BUILDING_INFO"].add(text)
                    elif any(word in text for word in ["Inc.", "LLC", "Ltd.", "Co."]):
                        text_findings["BUSINESS_NAMES"].add(text)
                    elif len(text.split()) > 3:  # Longer text likely informational
                        text_findings["INFORMATIONAL"].add(text)
                    else:
                        text_findings["OTHER_TEXT"].add(text)

        # Format the combined results
        result = ""
        for category, texts in text_findings.items():
            if texts:
                result += f"{category}:\n"
                for text in texts:
                    result += f"- {text}\n"
                result += "\n"

        return result

    def _parse_text_line(self, line: str) -> Tuple[str, float]:
        """Parse a line of text with confidence score"""
        try:
            if "confidence:" in line.lower():
                text = line.split("confidence:")[0].strip()
                confidence = float(line.split("confidence:")[1].strip().strip("()"))
                return text, confidence
            return line, 0.0
        except:
            return "", 0.0

    def _analyze_location_features(self, image: Image, text_features: Dict) -> str:
        """Analyze location features using the full image"""
        # Convert image for API
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/jpeg;base64,{image_base64}"

        # Add license plate analysis to the prompt
        location_prompt = f"""
        Analyze this image and extract all relevant location information.
        IMPORTANT: Focus on finding the MOST SPECIFIC location possible - prefer city/district level over country level.
        
        Previously extracted text:
        {text_features}
        
        License Plates Found:
        {', '.join(text_features.get('license_plates', []))}
        
        Please identify with high precision:
        1. Exact city/district name from license plates (e.g. if plate starts with 16, it's Bursa)
        2. Specific neighborhood or area within the city
        3. Landmarks and buildings (including any text/names found)
        4. Exact street names and addresses from visible signs
        5. Geographic features and surroundings
        6. Architectural styles and time period indicators
        7. Weather conditions and time of day
        8. Vegetation types
        9. Any additional location clues from the text
        10. Traffic signs and road markings style
        
        For each identified feature:
        - Rate your confidence (0-1)
        - Explain your reasoning
        - Start with the most specific location indicator (e.g. license plate city code)
        - Only fall back to broader regions if specific location cannot be determined
        """

        completion = self.client.chat.completions.create(
            model="google/gemini-2.0-flash-001",  # Keep using Gemini for general analysis
            messages=[{"role": "user", "content": [{"type": "text", "text": location_prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}],
            extra_headers=self.headers,
        )

        return completion.choices[0].message.content

    def _parse_text_analysis(self, text_analysis: str) -> Dict[str, List[str]]:
        """Parse the text extraction response into structured data"""
        features = {
            "street_signs": [],
            "building_info": [],
            "business_names": [],
            "informational": [],
            "other_text": [],
            "license_plates": [],  # List of raw plate numbers
            "license_plate_info": [],  # List of dicts with detailed plate info
        }

        current_category = None
        for line in text_analysis.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.endswith(":"):
                category = line[:-1].lower()
                if category in features:
                    current_category = category
                continue

            if current_category and line.startswith("-"):
                text = line[1:].strip()
                # Check for license plate patterns
                is_plate, plate_info = self._is_license_plate(text)
                if is_plate:
                    features["license_plates"].append(text)
                    if plate_info:
                        features["license_plate_info"].append({"plate_number": text, **plate_info})
                else:
                    features[current_category].append(text)

        return features

    def _is_license_plate(self, text: str) -> Tuple[bool, Optional[Dict[str, str]]]:
        """Detect if text matches common license plate patterns and extract region info"""
        import re

        # Common license plate patterns by region with city/area codes
        patterns = {
            "USA": {
                "patterns": [
                    (r"^([A-Z]{1,3})\s*\d{3,4}$", "state_prefix"),  # ABC 123 - State prefix
                    (r"^\d{3}\s*([A-Z]{2,3})$", "state_suffix"),  # 123 ABC - State suffix
                    (r"^([A-Z0-9]{1,3})[A-Z0-9]{2,5}$", "state_mixed"),  # Generic US plate
                ],
                "region_codes": {"NY": "New York", "CA": "California", "TX": "Texas", "FL": "Florida", "IL": "Illinois", "PA": "Pennsylvania"},
            },
            "Europe": {
                "patterns": [
                    # German format: City code + numbers + letters
                    (r"^([A-Z]{1,3})-[A-Z]{1,2}\s*\d{1,4}$", "german"),
                    # UK format: Area code + year + letters
                    (r"^([A-Z]{2})\d{2}\s*[A-Z]{3}$", "uk"),
                    # French format: Department number + letters + numbers
                    (r"^(\d{2,3})-[A-Z]{3}-\d{2,3}$", "french"),
                ],
                "region_codes": {
                    # German cities
                    "B": "Berlin",
                    "M": "Munich",
                    "K": "Cologne",
                    "F": "Frankfurt",
                    "HH": "Hamburg",
                    "S": "Stuttgart",
                    # UK areas
                    "LA": "London",
                    "MA": "Manchester",
                    "LV": "Liverpool",
                    "BI": "Birmingham",
                    "ED": "Edinburgh",
                    # French departments
                    "75": "Paris",
                    "69": "Lyon",
                    "13": "Marseille",
                    "33": "Bordeaux",
                    "31": "Toulouse",
                },
            },
            "Turkey": {
                "patterns": [
                    # Turkish format: City code (1-81) + Letter(s) + Numbers
                    (r"^(\d{2})\s*[A-Z]{1,3}\s*\d{2,4}$", "turkish"),  # 34 ABC 123
                    (r"^(\d{2})\s*[A-Z]{1,3}\s*\d{2,4}$", "turkish_new"),  # Modern format
                ],
                "region_codes": {
                    "01": "Adana",
                    "06": "Ankara",
                    "07": "Antalya",
                    "16": "Bursa",
                    "26": "Eskişehir",
                    "27": "Gaziantep",
                    "34": "Istanbul",
                    "35": "Izmir",
                    "38": "Kayseri",
                    "41": "Kocaeli",
                    "42": "Konya",
                    "55": "Samsun",
                    "61": "Trabzon",
                    "65": "Van",
                    "33": "Mersin",
                    "20": "Denizli",
                    "09": "Aydın",
                    "48": "Muğla",
                    "10": "Balıkesir",
                    "45": "Manisa",
                    "31": "Hatay",
                    "44": "Malatya",
                    "25": "Erzurum",
                    "23": "Elazığ",
                    "21": "Diyarbakır",
                    "46": "Kahramanmaraş",
                    "52": "Ordu",
                    "53": "Rize",
                    "54": "Sakarya",
                    "63": "Şanlıurfa",
                    # Add more Turkish cities as needed
                },
                "districts": {
                    "16": {  # Bursa districts
                        "central": ["Osmangazi", "Yıldırım", "Nilüfer", "Gürsu", "Kestel"],
                        "outer": ["Mudanya", "Gemlik", "İnegöl", "Orhangazi", "İznik"],
                    },
                    # Add other cities' districts as needed
                },
            },
            "Asia": {
                "patterns": [
                    # Indian format: State code + district code + number series
                    (r"^([A-Z]{2})\s*\d{1,2}\s*[A-Z]{1,2}\s*\d{4}$", "indian"),
                    # Korean format: Area code + vehicle type + numbers
                    (r"^(\d{2,3})\s*[가-힣]\s*\d{4}$", "korean"),
                    # Japanese format: Area name + classification + numbers
                    (r"^((?:東京|名古屋|大阪|横浜))\s*\d{3,4}$", "japanese"),
                ],
                "region_codes": {
                    # Indian states
                    "MH": "Maharashtra",
                    "DL": "Delhi",
                    "KA": "Karnataka",
                    "TN": "Tamil Nadu",
                    "UP": "Uttar Pradesh",
                    # Korean regions
                    "11": "Seoul",
                    "21": "Busan",
                    "22": "Daegu",
                    "23": "Incheon",
                    "24": "Gwangju",
                    # Japanese cities
                    "東京": "Tokyo",
                    "名古屋": "Nagoya",
                    "大阪": "Osaka",
                    "横浜": "Yokohama",
                },
            },
        }

        # Clean the text
        text = text.upper().replace("-", "").strip()

        # Check against all patterns
        for region, region_data in patterns.items():
            for pattern, pattern_type in region_data["patterns"]:
                match = re.match(pattern, text)
                if match:
                    # Extract the region code from the match
                    region_code = match.group(1)
                    # Look up the city/region name
                    location = region_data["region_codes"].get(region_code)
                    return True, {
                        "country": region,
                        "region_code": region_code,
                        "region_name": location if location else "Unknown",
                        "pattern_type": pattern_type,
                    }

        return False, None

    def _parse_analysis(self, analysis: str, text_features: Dict[str, List[str]]) -> Dict:
        """Parse the location analysis into structured features"""
        features = {
            "landmarks": [],
            "addresses": [],
            "architecture_style": "",
            "vegetation": [],
            "time_of_day": "",
            "weather": "",
            "geographic_features": [],
            "extracted_text": {
                "street_signs": text_features["street_signs"],
                "building_info": text_features["building_info"],
                "business_names": text_features["business_names"],
                "informational": text_features["informational"],
                "other": text_features["other_text"],
            },
            "confidence_scores": {},
        }

        # Parse the analysis text to populate features
        current_section = None
        for line in analysis.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Look for confidence indicators
            if "confidence:" in line.lower():
                try:
                    feature, score = line.split("confidence:", 1)
                    feature = feature.strip().lower()
                    score = float(score.strip().split()[0])
                    features["confidence_scores"][feature] = score
                except:
                    continue

            # Extract features based on context
            if "landmark:" in line.lower() or "building:" in line.lower():
                features["landmarks"].append(line.split(":", 1)[1].strip())
            elif "address:" in line.lower():
                features["addresses"].append(line.split(":", 1)[1].strip())
            elif "architecture:" in line.lower():
                features["architecture_style"] = line.split(":", 1)[1].strip()
            elif "vegetation:" in line.lower():
                features["vegetation"].extend([v.strip() for v in line.split(":", 1)[1].split(",")])
            elif "geographic:" in line.lower():
                features["geographic_features"].append(line.split(":", 1)[1].strip())

        return features

    def _analyze_shadows(self, cv_image: np.ndarray) -> Dict[str, float]:
        """Analyze shadows to estimate time of day"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply threshold to separate shadows
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

            # Find contours of shadows
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return {"time_confidence": 0.0, "estimated_hour": None}

            # Find the longest shadow
            longest_shadow = max(contours, key=cv2.contourArea)

            # Get the orientation of the shadow
            rect = cv2.minAreaRect(longest_shadow)
            angle = rect[2]

            # Normalize angle to 0-360 range
            angle = angle % 360
            if angle < 0:
                angle += 360

            # Calculate shadow length ratio
            shadow_length = max(rect[1])
            object_height = min(rect[1])
            if object_height == 0:
                return {"time_confidence": 0.0, "estimated_hour": None}

            length_ratio = shadow_length / object_height

            # Estimate time based on shadow angle and length
            # Morning: shadows point west (270°)
            # Noon: shortest shadows
            # Evening: shadows point east (90°)

            if 225 <= angle <= 315:  # Morning
                estimated_hour = 7 + (length_ratio - 3) / 0.5  # Rough mapping of length to hour
            elif 45 <= angle <= 135:  # Evening
                estimated_hour = 17 + (length_ratio - 3) / 0.5
            else:  # Mid-day
                estimated_hour = 12 + (length_ratio - 1) * 2

            # Constrain to reasonable daylight hours
            estimated_hour = max(6, min(20, estimated_hour))

            # Calculate confidence based on shadow clarity
            shadow_area = cv2.contourArea(longest_shadow)
            image_area = cv_image.shape[0] * cv_image.shape[1]
            shadow_clarity = shadow_area / image_area

            confidence = min(1.0, shadow_clarity * 5)  # Scale up but cap at 1.0

            return {"time_confidence": confidence, "estimated_hour": round(estimated_hour, 1), "shadow_angle": angle, "shadow_length_ratio": length_ratio}

        except Exception as e:
            print(f"Error analyzing shadows: {e}")
            return {"time_confidence": 0.0, "estimated_hour": None}

    def _estimate_time_of_day(self, cv_image: np.ndarray, metadata: Dict) -> Dict:
        """Estimate time of day using multiple methods"""
        # Get shadow-based estimate
        shadow_estimate = self._analyze_shadows(cv_image)

        # Get brightness-based estimate
        brightness = cv2.mean(cv_image)[0] / 255.0  # Normalize to 0-1

        # Time ranges based on brightness
        # Dawn: 0.2-0.4
        # Day: 0.4-0.8
        # Dusk: 0.2-0.4
        # Night: 0-0.2

        time_ranges = {(0.0, 0.2): "night", (0.2, 0.4): "dawn/dusk", (0.4, 0.8): "day", (0.8, 1.0): "bright day"}

        time_of_day = next((label for (low, high), label in time_ranges.items() if low <= brightness <= high), "unknown")

        # Combine estimates
        result = {
            "time_of_day": time_of_day,
            "brightness": brightness,
            "shadow_analysis": shadow_estimate,
            "confidence": max(0.3, (shadow_estimate["time_confidence"] + 0.7 * brightness) / 2),
        }

        # Add specific hour if available
        if shadow_estimate["estimated_hour"] is not None:
            result["estimated_hour"] = shadow_estimate["estimated_hour"]

        # Add time period
        if shadow_estimate["estimated_hour"] is not None:
            hour = shadow_estimate["estimated_hour"]
            if 5 <= hour < 12:
                result["period"] = "morning"
            elif 12 <= hour < 17:
                result["period"] = "afternoon"
            elif 17 <= hour < 21:
                result["period"] = "evening"
            else:
                result["period"] = "night"

        return result

    def _analyze_environment(self, cv_image: np.ndarray) -> Dict:
        """Analyze environmental features in the image"""
        features = {
            "terrain_type": [],
            "water_bodies": [],
            "sky_features": [],
            "building_density": "unknown",
            "road_types": [],
            "vegetation_density": "unknown",
        }

        try:
            # Analyze terrain using color segmentation
            terrain_features = self._analyze_terrain(cv_image)
            features["terrain_type"] = terrain_features

            # Detect water bodies
            water_features = self._detect_water_bodies(cv_image)
            features["water_bodies"] = water_features

            # Analyze sky features
            sky_features = self._analyze_sky(cv_image)
            features["sky_features"] = sky_features

            # Estimate building density
            features["building_density"] = self._estimate_building_density(cv_image)

            # Detect road types
            features["road_types"] = self._detect_road_types(cv_image)

            # Analyze vegetation density
            features["vegetation_density"] = self._analyze_vegetation_density(cv_image)

        except Exception as e:
            print(f"Error in environment analysis: {e}")

        return features

    def _analyze_terrain(self, image: np.ndarray) -> List[str]:
        """Analyze terrain types using color and texture analysis"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different terrain types
        terrain_colors = {
            "sand": ([20, 50, 50], [40, 255, 255]),
            "grass": ([35, 50, 50], [85, 255, 255]),
            "snow": ([0, 0, 200], [180, 30, 255]),
            "rock": ([0, 0, 50], [180, 50, 200]),
        }

        terrain_types = []
        for terrain, (lower, upper) in terrain_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if np.sum(mask) > 0.1 * mask.size:  # If more than 10% of image matches
                terrain_types.append(terrain)

        return terrain_types

    def _detect_water_bodies(self, image: np.ndarray) -> List[str]:
        """Detect presence of water bodies"""
        # Convert to HSV for water detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Water color ranges (blue/green hues)
        water_lower = np.array([90, 50, 50])
        water_upper = np.array([130, 255, 255])

        # Create mask for water colors
        mask = cv2.inRange(hsv, water_lower, water_upper)

        water_types = []
        water_area = np.sum(mask) / mask.size

        if water_area > 0.3:  # Large water body
            water_types.append("large_water_body")
        elif water_area > 0.1:  # Small water body
            water_types.append("small_water_body")

        return water_types

    def _analyze_sky(self, image: np.ndarray) -> List[str]:
        """Analyze sky features"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Sky detection (upper third of image)
        height = image.shape[0]
        sky_region = hsv[: height // 3, :, :]

        features = []

        # Check for clear sky
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([140, 255, 255])
        blue_mask = cv2.inRange(sky_region, blue_lower, blue_upper)

        # Check for clouds
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(sky_region, white_lower, white_upper)

        if np.sum(blue_mask) > 0.6 * blue_mask.size:
            features.append("clear_sky")
        if np.sum(white_mask) > 0.3 * white_mask.size:
            features.append("cloudy")

        return features

    def _estimate_building_density(self, image: np.ndarray) -> str:
        """Estimate building density in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge density
        edge_density = np.sum(edges) / edges.size

        if edge_density > 0.1:
            return "high_density"
        elif edge_density > 0.05:
            return "medium_density"
        else:
            return "low_density"

    def _detect_road_types(self, image: np.ndarray) -> List[str]:
        """Detect types of roads present"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Line detection using HoughLinesP
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        road_types = []
        if lines is not None:
            if len(lines) > 20:
                road_types.append("major_road")
            elif len(lines) > 10:
                road_types.append("street")
            else:
                road_types.append("path")

        return road_types

    def _analyze_vegetation_density(self, image: np.ndarray) -> str:
        """Analyze vegetation density using green color detection"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green color range for vegetation
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        # Create mask for vegetation
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Calculate vegetation density
        density = np.sum(mask) / mask.size

        if density > 0.3:
            return "high"
        elif density > 0.1:
            return "medium"
        else:
            return "low"

    def _analyze_camera_meta(self, image: np.ndarray):
        """Analyze camera generation and height"""
        # Check for circular blur patterns characteristic of Gen 2
        # Check for color saturation levels for Gen 3 vs 4
        # Check for low camera height indicators
        # Implementation details...

    def _analyze_driving_side(self, image: np.ndarray):
        """Determine which side of the road vehicles drive on"""
        # Look for:
        # - Visible vehicles and their direction
        # - Road signs placement
        # - Road markings direction
        # Implementation details...

    def _analyze_with_vlm(self, image: Image) -> Dict:
        """Use VLM to analyze high-level features"""
        # Convert image for API
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/jpeg;base64,{image_base64}"

        prompt = """Analyze this image for location-identifying features. Focus on:
        1. Architecture style and building materials
        2. Vegetation types and patterns
        3. Road infrastructure (signs, poles, bollards)
        4. Terrain and landscape features
        5. Any visible text or signage
        6. Weather conditions and time of day
        7. Cultural indicators (clothing, vehicles, etc)
        
        Format your response in this structure:
        ARCHITECTURE:
        - Style: [style name]
        - Materials: [list materials]
        - Notable features: [list features]
        
        VEGETATION:
        - Types: [list vegetation types]
        - Density: [low/medium/high]
        
        INFRASTRUCTURE:
        - Road type: [type]
        - Signs: [list signs]
        - Street furniture: [list items]
        
        TERRAIN:
        - Type: [describe terrain]
        - Features: [list notable features]
        
        TEXT:
        - Signs: [list text from signs]
        - Business names: [list business names]
        - Other text: [list other visible text]
        
        CULTURAL:
        - Vehicles: [describe vehicles]
        - People: [describe clothing/activities]
        - Other indicators: [list other cultural elements]
        
        WEATHER/TIME:
        - Weather: [describe conditions]
        - Time of day: [estimate]
        - Season: [if apparent]
        
        For each item, include a confidence score (0-1)."""

        try:
            completion = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}],
                extra_headers=self.headers,
            )

            # Parse the response into structured data
            response_text = completion.choices[0].message.content
            features = self._parse_vlm_response(response_text)

            return features

        except Exception as e:
            print(f"Error in VLM analysis: {e}")
            # Return empty features rather than None
            return {
                "architecture_style": None,
                "vegetation": [],
                "infrastructure_features": [],
                "terrain_features": [],
                "extracted_text": {"signs": [], "business_names": [], "other": []},
                "cultural_indicators": [],
                "weather": None,
                "time_of_day": None,
                "season": None,
                "confidence_scores": {},
            }

    def _parse_vlm_response(self, response: str) -> Dict:
        """Parse the VLM response into structured features"""
        features = {
            "architecture_style": None,
            "vegetation": [],
            "infrastructure_features": [],
            "terrain_features": [],
            "extracted_text": {"signs": [], "business_names": [], "other": []},
            "cultural_indicators": [],
            "landmarks": [],  # Add landmarks list
            "addresses": [],  # Add addresses list
            "geographic_features": [],  # Add geographic features
            "weather": None,
            "time_of_day": None,
            "season": None,
            "confidence_scores": {},
        }

        try:
            current_section = None
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Check for section headers
                if line.endswith(":"):
                    current_section = line[:-1].lower()
                    continue

                # Parse content based on section
                if line.startswith("-"):
                    content = line[1:].strip()
                    confidence = 1.0  # Default confidence

                    # Extract confidence if present
                    if "(confidence:" in content.lower():
                        parts = content.lower().split("(confidence:")
                        content = parts[0].strip()
                        try:
                            confidence = float(parts[1].strip(")"))
                        except ValueError:
                            pass

                    # Add content to appropriate section
                    if current_section == "architecture":
                        if "style:" in content.lower():
                            features["architecture_style"] = content.split(":", 1)[1].strip()
                            features["confidence_scores"]["architecture"] = confidence
                        elif "notable features:" in content.lower():
                            features["landmarks"].extend([f.strip() for f in content.split(":", 1)[1].split(",")])
                    elif current_section == "vegetation":
                        if "types:" in content.lower():
                            features["vegetation"].extend([v.strip() for v in content.split(":", 1)[1].split(",")])
                            features["confidence_scores"]["vegetation"] = confidence
                    elif current_section == "infrastructure":
                        if any(k in content.lower() for k in ["road type:", "signs:", "street furniture:"]):
                            features["infrastructure_features"].append(content)
                    elif current_section == "terrain":
                        features["terrain_features"].append(content)
                        if "type:" in content.lower():
                            features["confidence_scores"]["terrain"] = confidence
                    elif current_section == "text":
                        if "business names:" in content.lower():
                            features["extracted_text"]["business_names"].extend([b.strip() for b in content.split(":", 1)[1].split(",")])
                        elif "signs:" in content.lower():
                            features["extracted_text"]["signs"].extend([s.strip() for s in content.split(":", 1)[1].split(",")])
                        else:
                            features["extracted_text"]["other"].append(content)
                        features["confidence_scores"]["text"] = confidence
                    elif current_section == "cultural":
                        features["cultural_indicators"].append(content)
                    elif current_section == "weather/time":
                        if "weather:" in content.lower():
                            features["weather"] = content.split(":", 1)[1].strip()
                        elif "time of day:" in content.lower():
                            features["time_of_day"] = content.split(":", 1)[1].strip()
                        elif "season:" in content.lower():
                            features["season"] = content.split(":", 1)[1].strip()

        except Exception as e:
            print(f"Error parsing VLM response: {e}")
            # Don't override the default empty features on error

        return features

    def _generate_analysis_description(self, features: Dict) -> str:
        """Generate human-readable description of analysis results"""
        description_parts = []

        # Add architecture information
        if features.get("architecture_style"):
            description_parts.append(f"Architecture Style: {features['architecture_style']}")

        # Add landmarks
        if features.get("landmarks"):
            description_parts.append(f"Landmarks: {', '.join(features['landmarks'])}")

        # Add business names
        if features.get("extracted_text", {}).get("business_names"):
            description_parts.append(f"Businesses: {', '.join(features['extracted_text']['business_names'])}")

        # Add signs
        if features.get("extracted_text", {}).get("signs"):
            description_parts.append(f"Signs: {', '.join(features['extracted_text']['signs'])}")

        # Add infrastructure features
        if features.get("infrastructure_features"):
            description_parts.append(f"Infrastructure: {', '.join(features['infrastructure_features'])}")

        # Add terrain features
        if features.get("terrain_features"):
            description_parts.append(f"Terrain: {', '.join(features['terrain_features'])}")

        # Add weather and time
        if features.get("weather"):
            description_parts.append(f"Weather: {features['weather']}")
        if features.get("time_of_day"):
            description_parts.append(f"Time: {features['time_of_day']}")

        # Add confidence scores
        if features.get("confidence_scores"):
            scores = [f"{k.title()}: {v:.2f}" for k, v in features["confidence_scores"].items()]
            description_parts.append(f"Confidence Scores: {', '.join(scores)}")

        return "\n".join(description_parts)

    def _analyze_infrastructure(self, image: np.ndarray) -> None:
        """Analyze infrastructure elements in the image"""
        # Analyze road type
        self.analysis_categories["infrastructure"]["road_type"] = self._detect_road_type(image)

        # Analyze road lines
        self.analysis_categories["infrastructure"]["road_lines"] = self._detect_road_lines(image)

        # Detect bollards
        self.analysis_categories["infrastructure"]["bollards"] = self._detect_bollards(image)

        # Detect utility poles
        self.analysis_categories["infrastructure"]["utility_poles"] = self._detect_utility_poles(image)

        # Detect traffic signs
        self.analysis_categories["infrastructure"]["signs"] = self._detect_traffic_signs(image)

    def _detect_road_type(self, image: np.ndarray) -> str:
        """Detect the type of road surface"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Get lower third of image where road is likely to be
        height = image.shape[0]
        road_region = hsv[int(height * 2 / 3) : height, :]

        max_pixels = 0
        road_type = "unknown"

        for rtype, (lower, upper) in self.road_hsv_ranges.items():
            mask = cv2.inRange(road_region, np.array(lower), np.array(upper))
            pixel_count = np.sum(mask > 0)
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                road_type = rtype

        return road_type

    def _detect_road_lines(self, image: np.ndarray) -> List[str]:
        """Detect road line markings"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_lines = []

        for color, (lower, upper) in self.line_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Apply edge detection
            edges = cv2.Canny(mask, 50, 150)

            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

            if lines is not None and len(lines) > 0:
                detected_lines.append(f"{color}_lines")

        return detected_lines

    def _detect_bollards(self, image: np.ndarray) -> List[str]:
        """Detect bollards in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bollards = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on aspect ratio and size
            aspect_ratio = h / w if w > 0 else 0
            if 2 < aspect_ratio < 8 and 20 < h < 200:
                # Get color of potential bollard
                roi = image[y : y + h, x : x + w]
                avg_color = np.mean(roi, axis=(0, 1))

                # Classify bollard type based on color and shape
                bollard_type = self._classify_bollard(avg_color, aspect_ratio)
                if bollard_type:
                    bollards.append(bollard_type)

        return list(set(bollards))  # Remove duplicates

    def _detect_utility_poles(self, image: np.ndarray) -> List[str]:
        """Detect utility poles in the image"""
        # Similar approach to bollard detection but with different parameters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect vertical lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=20)

        poles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                # If nearly vertical
                if angle > 80:
                    # Get region around line
                    min_x = max(0, min(x1, x2) - 10)
                    max_x = min(image.shape[1], max(x1, x2) + 10)
                    roi = image[min(y1, y2) : max(y1, y2), min_x:max_x]

                    if roi.size > 0:
                        pole_type = self._classify_pole(roi)
                        if pole_type:
                            poles.append(pole_type)

        return list(set(poles))

    def _detect_traffic_signs(self, image: np.ndarray) -> List[str]:
        """Detect traffic signs in the image"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for common sign colors
        color_ranges = {"red": [(0, 100, 100), (10, 255, 255)], "blue": [(100, 100, 100), (130, 255, 255)], "yellow": [(20, 100, 100), (30, 255, 255)]}

        signs = []
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Get shape properties
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small detections
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

                    # Classify sign based on shape
                    sign_type = self._classify_sign(len(approx), color)
                    if sign_type:
                        signs.append(sign_type)

        return list(set(signs))

    def _classify_bollard(self, color: np.ndarray, aspect_ratio: float) -> Optional[str]:
        """Classify bollard type based on color and shape"""
        # Simple classification based on color
        r, g, b = color

        if r > 200 and g > 200 and b > 200:
            return "white_bollard"
        elif r > 200 and g < 100 and b < 100:
            return "red_bollard"
        elif r > 200 and g > 150 and b < 100:
            return "yellow_bollard"

        return None

    def _classify_pole(self, roi: np.ndarray) -> Optional[str]:
        """Classify utility pole type based on appearance"""
        # Simple classification based on average color and texture
        avg_color = np.mean(roi, axis=(0, 1))

        if avg_color[0] > 100:  # Concrete/metal pole
            return "concrete_pole"
        else:  # Wooden pole
            return "wooden_pole"

    def _classify_sign(self, vertices: int, color: str) -> Optional[str]:
        """Classify traffic sign based on shape and color"""
        if vertices == 3:
            return f"{color}_triangle_sign"
        elif vertices == 4:
            return f"{color}_rectangle_sign"
        elif vertices > 7:  # Approximately circular
            return f"{color}_circle_sign"

        return None
