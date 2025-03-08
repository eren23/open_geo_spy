from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
import cv2
from openai import OpenAI
import base64
import requests
import io
from math import ceil
import re
from .environment_classifier import EnvironmentClassifier, EnvironmentType, EnvironmentInfo


class ImageAnalyzer:
    def __init__(self, openrouter_api_key: str, app_name: str, app_url: str):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
        self.headers = {"HTTP-Referer": app_url, "X-Title": app_name}
        self.chunk_size = (800, 800)  # Size of each chunk
        self.overlap = 100  # Overlap between chunks in pixels
        self.environment_classifier = EnvironmentClassifier()

    def analyze_image(self, image_path: str) -> Tuple[Dict, str]:
        """Analyze image with enhanced text and visual understanding"""
        # Load and prepare image
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path)
            image = Image.open(io.BytesIO(response.content))
            cv_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = Image.open(image_path)
            cv_image = cv2.imread(image_path)

        # Get image chunks
        chunks = self._create_image_chunks(cv_image)

        # Extract text from each chunk
        all_text = []
        for i, (chunk, coords) in enumerate(chunks):
            chunk_text = self._extract_text_from_chunk(chunk, image, coords)
            if chunk_text:
                all_text.append(chunk_text)

        # Combine and deduplicate text findings
        combined_text = self._combine_chunk_results(all_text)
        text_features = self._parse_text_analysis(combined_text)

        # Analyze environmental features
        env_features = self._analyze_environment(cv_image)

        # Classify environment type
        env_info = self.environment_classifier.classify_environment(cv_image, env_features)

        # Add environment classification to features
        env_features["environment_type"] = env_info.primary_type.name
        env_features["secondary_environments"] = [t.name for t in env_info.secondary_types]
        env_features["environment_confidence"] = env_info.confidence

        # Second pass: Full image analysis with text context and environment type
        location_analysis = self._analyze_location_features(image, text_features, env_info)
        features = self._parse_analysis(location_analysis, text_features)

        # Merge environmental features
        features.update(env_features)

        # Add time analysis
        time_analysis = self._estimate_time_of_day(cv_image, {})
        features["time_analysis"] = time_analysis

        if time_analysis.get("estimated_hour"):
            # Format time as HH:MM, ensuring consistent decimal places
            hour = int(time_analysis["estimated_hour"])
            minute = int((time_analysis["estimated_hour"] % 1) * 60)
            features["time_of_day"] = f"{hour:02d}:{minute:02d}"
        else:
            features["time_of_day"] = time_analysis["time_of_day"]

        return features, f"{combined_text}\n\n{location_analysis}"

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

    def _analyze_location_features(self, image: Image, text_features: Dict, env_info: EnvironmentInfo) -> str:
        """Analyze location features using the full image"""
        # Convert image for API
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/jpeg;base64,{image_base64}"

        # Build environment-aware prompt
        location_prompt = f"""
        Analyze this image and extract all relevant location information.
        IMPORTANT: Focus on finding the MOST SPECIFIC location possible - prefer city/district level over country level.
        
        Environment Type: {env_info.primary_type.name} (confidence: {env_info.confidence:.2f})
        Secondary Environments: {', '.join(t.name for t in env_info.secondary_types)}
        
        Previously extracted text:
        {text_features}
        
        License Plates Found:
        {', '.join(text_features.get('license_plates', []))}
        
        Please identify with high precision, considering the environment type:
        1. {self._get_priority_instructions(env_info.primary_type)}
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
        - Start with the most specific location indicator
        - Only fall back to broader regions if specific location cannot be determined
        
        Consider the environment type when weighing evidence:
        {self._get_environment_specific_guidance(env_info.primary_type)}
        """

        completion = self.client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[{"role": "user", "content": [{"type": "text", "text": location_prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}],
            extra_headers=self.headers,
        )

        return completion.choices[0].message.content

    def _get_priority_instructions(self, env_type: EnvironmentType) -> str:
        """Get environment-specific priority instructions"""
        instructions = {
            EnvironmentType.URBAN: "Exact city/district name from license plates and business signs",
            EnvironmentType.SUBURBAN: "Neighborhood identifiers and residential area names",
            EnvironmentType.RURAL: "Nearby town names and geographic features",
            EnvironmentType.INDUSTRIAL: "Industrial zone names and company identifiers",
            EnvironmentType.AIRPORT: "Airport name and terminal identifiers",
            EnvironmentType.COASTAL: "Beach/port names and coastal landmarks",
            EnvironmentType.FOREST: "Forest name and nearby settlements",
            EnvironmentType.MOUNTAIN: "Mountain range and peak names",
            EnvironmentType.DESERT: "Desert region and nearest settlement",
            EnvironmentType.PARK: "Park name and surrounding urban area",
            EnvironmentType.HIGHWAY: "Highway number and nearest exits/intersections",
            EnvironmentType.UNKNOWN: "Most prominent location indicators visible",
        }
        return instructions.get(env_type, instructions[EnvironmentType.UNKNOWN])

    def _get_environment_specific_guidance(self, env_type: EnvironmentType) -> str:
        """Get environment-specific guidance for location analysis"""
        guidance = {
            EnvironmentType.URBAN: """
                - License plates and business signs are highly reliable indicators
                - Street signs and addresses provide precise location data
                - Building numbers and business names are key reference points
                - Pay attention to transit system branding and station names
            """,
            EnvironmentType.SUBURBAN: """
                - Focus on neighborhood entrance signs and community markers
                - School zones and local business names are good indicators
                - Street name patterns can identify specific developments
                - Look for postal codes on visible mail boxes or signs
            """,
            EnvironmentType.RURAL: """
                - Geographic features are more reliable than man-made markers
                - Farm/ranch names can provide regional context
                - Distance markers to nearest towns are valuable
                - Local business types can indicate specific regions
            """,
            EnvironmentType.INDUSTRIAL: """
                - Company names and industrial park signs are primary indicators
                - Look for shipping addresses and facility numbers
                - Industrial zone classifications can narrow down location
                - Transport infrastructure can indicate specific industrial areas
            """,
            EnvironmentType.AIRPORT: """
                - Terminal signs and gate numbers are highly specific
                - Airline branding can indicate the hub location
                - Airport codes on vehicles or equipment are reliable
                - Focus less on license plates unless on service vehicles
            """,
            EnvironmentType.COASTAL: """
                - Beach names and marina signage are primary indicators
                - Port numbers and dock identifiers are specific
                - Coastal business names often include location
                - Look for marine navigation markers
            """,
            EnvironmentType.FOREST: """
                - Forest service signs and trail markers are key
                - Look for ranger station identifiers
                - Trail names often include regional information
                - Focus on natural landmarks more than human infrastructure
            """,
            EnvironmentType.MOUNTAIN: """
                - Peak names and elevation markers are primary
                - Trail signs and route markers provide context
                - Ski resort branding can be location-specific
                - Look for geological formation names
            """,
            EnvironmentType.DESERT: """
                - Focus on distance markers to settlements
                - Protected area names and boundaries
                - Geographic feature names are key
                - Look for research station or military installation markers
            """,
            EnvironmentType.PARK: """
                - Park name and entrance signs are primary
                - Look for municipal recreation department branding
                - Trail markers and facility numbers help
                - Nearby street names provide urban context
            """,
            EnvironmentType.HIGHWAY: """
                - Mile markers and exit numbers are highly specific
                - Highway numbers and intersection signs are key
                - Service area names provide regional context
                - Focus on official road signage over business signs
            """,
            EnvironmentType.UNKNOWN: """
                - Consider all visible location indicators equally
                - Look for the most specific and official markers
                - Cross-reference multiple types of evidence
                - Consider both natural and man-made features
            """,
        }
        return guidance.get(env_type, guidance[EnvironmentType.UNKNOWN])

    def _parse_text_analysis(self, text_analysis: str) -> Dict[str, List[str]]:
        """Parse the text extraction response into structured data with improved entity categorization"""
        features = {
            "street_signs": [],
            "building_info": [],
            "business_names": [],
            "informational": [],
            "other_text": [],
            "license_plates": [],  # List of raw plate numbers
            "license_plate_info": [],  # List of dicts with detailed plate info
            "entity_types": {},  # Store entity type classifications
        }

        current_category = None
        for line in text_analysis.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for category headers
            if line.endswith(":") and line.isupper():
                current_category = line[:-1].lower()
                continue

            # Process text within a category
            if current_category and ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    text = parts[1].strip()
                    if text:
                        # Detect entity type with more nuanced rules
                        entity_type = self._classify_entity_type(text)
                        features["entity_types"][text] = entity_type

                        # Categorize based on entity type
                        if entity_type == "business":
                            features["business_names"].append(text)
                        elif entity_type == "street":
                            features["street_signs"].append(text)
                        elif entity_type == "building":
                            features["building_info"].append(text)
                        elif entity_type == "license_plate":
                            is_plate, plate_info = self._is_license_plate(text)
                            features["license_plates"].append(text)
                            if plate_info:
                                features["license_plate_info"].append({"plate_number": text, **plate_info})
                        else:
                            features[current_category].append(text)

        return features

    def _classify_entity_type(self, text: str) -> str:
        """Classify text into entity types using pattern matching and contextual clues"""
        text_lower = text.lower()

        # Business indicators
        business_indicators = [
            "gmbh",
            "ltd",
            "inc",
            "llc",
            "co.",
            "kg",
            "ag",
            "restaurant",
            "café",
            "cafe",
            "hotel",
            "shop",
            "store",
            "markt",
            "market",
            "supermarket",
            "biomarkt",
            "bakery",
            "bäckerei",
            "apotheke",
            "pharmacy",
            "bank",
        ]

        # Street indicators
        street_indicators = [
            "straße",
            "strasse",
            "str.",
            "street",
            "avenue",
            "ave",
            "road",
            "rd",
            "boulevard",
            "blvd",
            "lane",
            "ln",
            "way",
            "allee",
            "platz",
            "square",
        ]

        # Building indicators
        building_indicators = ["building", "gebäude", "haus", "house", "apartment", "apt", "suite", "floor", "etage", "no.", "nr.", "number", "hausnummer"]

        # Check for business patterns
        if any(indicator in text_lower for indicator in business_indicators):
            return "business"

        # Check for street patterns
        if any(indicator in text_lower for indicator in street_indicators):
            return "street"

        # Check for building patterns
        if any(indicator in text_lower for indicator in building_indicators) or re.search(r"\d+\s*[a-zA-Z]?$", text):
            return "building"

        # Check for license plate patterns (simplified check)
        if re.search(r"^[A-Z]{1,3}[-\s]?[0-9]{1,4}[-\s]?[A-Z]{0,3}$", text):
            return "license_plate"

        # Default to other
        return "other"

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

    def analyze_video(self, video_path: str) -> Tuple[Dict, str]:
        """Analyze video for text and location features"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            # Initialize scene detection
            scene_detector = cv2.createBackgroundSubtractorMOG2()
            last_scene_time = 0
            scene_threshold = 0.3  # Minimum scene change threshold
            min_scene_duration = 2.0  # Minimum seconds between scenes

            text_findings = []
            scene_descriptions = []
            current_frame = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = current_frame / fps

                # Apply scene detection
                fgmask = scene_detector.apply(frame)
                scene_score = np.mean(fgmask) / 255.0

                # Check if this is a new scene
                if scene_score > scene_threshold and current_time - last_scene_time > min_scene_duration:

                    # Convert frame for analysis
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Get scene description
                    scene_desc = self._analyze_scene(pil_image, current_time)
                    scene_descriptions.append({"timestamp": current_time, "description": scene_desc})

                    # Extract text from scene
                    text_result = self._extract_text_from_frame(frame, current_time)
                    if text_result:
                        text_findings.append(text_result)

                    last_scene_time = current_time

                current_frame += int(fps)  # Skip frames for efficiency
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

            cap.release()

            # Combine results
            combined_text = self._combine_video_findings(text_findings)
            scene_summary = self._summarize_scenes(scene_descriptions)

            return {
                "text_findings": combined_text,
                "scene_analysis": scene_summary,
                "duration": duration,
                "frame_count": frame_count,
                "fps": fps,
            }, scene_summary

        except Exception as e:
            print(f"Error analyzing video: {e}")
            return {}, ""

    def _analyze_scene(self, image: Image, timestamp: float) -> str:
        """Analyze a video scene using Gemini"""
        try:
            # Convert image for API
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_url = f"data:image/jpeg;base64,{image_base64}"

            prompt = f"""
            Analyze this video frame at timestamp {timestamp:.2f} seconds.
            Describe:
            1. The main scene content and setting
            2. Any significant objects or actions
            3. Environmental conditions
            4. Any text or signs visible
            5. Notable location indicators
            
            Be concise but specific.
            """

            completion = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}],
                extra_headers=self.headers,
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error analyzing scene: {e}")
            return ""

    def _extract_text_from_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Extract text from a video frame with improved chunking"""
        height, width = frame.shape[:2]

        # Create overlapping chunks with dynamic sizing
        chunk_size = min(width, height) // 2
        overlap = chunk_size // 3
        chunks = []

        for y in range(0, height - overlap, chunk_size - overlap):
            for x in range(0, width - overlap, chunk_size - overlap):
                x2 = min(x + chunk_size, width)
                y2 = min(y + chunk_size, height)
                chunk = frame[y:y2, x:x2]
                chunks.append((chunk, (x, y, x2, y2)))

        # Process chunks in parallel
        chunk_results = []
        for chunk, coords in chunks:
            result = self._extract_text_from_chunk(chunk, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), coords)
            if result:
                chunk_results.append(result)

        return {"timestamp": timestamp, "text": self._combine_chunk_results(chunk_results)}

    def _combine_video_findings(self, findings: List[Dict]) -> Dict:
        """Combine and track text findings across video frames"""
        text_tracks = {"STREET_SIGNS": {}, "BUILDING_INFO": {}, "BUSINESS_NAMES": {}, "INFORMATIONAL": {}, "OTHER_TEXT": {}}

        # Track text occurrences over time
        for finding in findings:
            timestamp = finding["timestamp"]
            text_data = self._parse_text_analysis(finding["text"])

            for category in text_tracks:
                if category.lower() in text_data:
                    for text in text_data[category.lower()]:
                        if text not in text_tracks[category]:
                            text_tracks[category][text] = []
                        text_tracks[category][text].append(timestamp)

        # Filter and combine results
        combined = {}
        for category, tracks in text_tracks.items():
            filtered_tracks = {}
            for text, timestamps in tracks.items():
                # Only keep text that appears in multiple frames or with high confidence
                if len(timestamps) > 1:
                    filtered_tracks[text] = {"timestamps": timestamps, "duration": timestamps[-1] - timestamps[0], "occurrences": len(timestamps)}
            if filtered_tracks:
                combined[category] = filtered_tracks

        return combined

    def _summarize_scenes(self, scenes: List[Dict]) -> str:
        """Generate a temporal summary of video scenes"""
        summary = "Video Scene Analysis:\n\n"

        for i, scene in enumerate(scenes, 1):
            summary += f"Scene {i} (at {scene['timestamp']:.2f}s):\n"
            summary += f"{scene['description']}\n\n"

        return summary
