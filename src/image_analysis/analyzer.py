from typing import Dict, List, Tuple
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

        # Second pass: Full image analysis with text context
        location_analysis = self._analyze_location_features(image, text_features)
        features = self._parse_analysis(location_analysis, text_features)

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

        # Update prompt with time information
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

        Time Analysis:
        - Time of Day: {features['time_of_day']}
        - Period: {time_analysis.get('period', 'unknown')}
        - Confidence: {time_analysis['confidence']:.2f}
        """

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

    def _analyze_location_features(self, image: Image, text_features: Dict) -> str:
        """Analyze location features using the full image"""
        # Convert image for API
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/jpeg;base64,{image_base64}"

        location_prompt = f"""
        Analyze this image and extract all relevant location information.
        
        Previously extracted text:
        {text_features}
        
        Please identify:
        1. Landmarks and buildings (including any text/names found)
        2. Exact street names and addresses from visible signs
        3. Geographic features and surroundings
        4. Architectural styles and time period indicators
        5. Weather conditions and time of day
        6. Vegetation types
        7. Any additional location clues from the text
        
        For each identified feature, rate your confidence (0-1) and explain your reasoning.
        """

        completion = self.client.chat.completions.create(
            model="google/gemini-2.0-flash-001",  # Keep using Gemini for general analysis
            messages=[{"role": "user", "content": [{"type": "text", "text": location_prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}],
            extra_headers=self.headers,
        )

        return completion.choices[0].message.content

    def _parse_text_analysis(self, text_analysis: str) -> Dict[str, List[str]]:
        """Parse the text extraction response into structured data"""
        features = {"street_signs": [], "building_info": [], "business_names": [], "informational": [], "other_text": []}

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
                features[current_category].append(line[1:].strip())

        return features

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
