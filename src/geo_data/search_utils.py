from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re


class LocationSearcher:
    def __init__(self):
        self.search_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def search_location_info(self, query: str, city: str) -> List[Dict]:
        """Search for location information using Google and parse results"""
        search_query = f"{query} {city} address location"
        results = []

        try:
            # Perform Google search
            search_results = search(search_query, num_results=5)

            for url in search_results:
                # Skip certain domains that usually don't have useful address info
                if any(domain in url.lower() for domain in ["facebook.com", "twitter.com", "instagram.com"]):
                    continue

                page_info = self._extract_page_info(url)
                if page_info:
                    results.append(page_info)

        except Exception as e:
            print(f"Error performing location search: {e}")

        return results

    def _extract_page_info(self, url: str) -> Optional[Dict]:
        """Extract location information from a webpage"""
        try:
            response = requests.get(url, headers=self.search_headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")

            # Look for address patterns
            text = soup.get_text()
            addresses = self._find_addresses(text)

            if addresses:
                return {"url": url, "title": soup.title.string if soup.title else "", "addresses": addresses, "source": "web_search"}

        except Exception as e:
            print(f"Error extracting info from {url}: {e}")

        return None

    def _find_addresses(self, text: str) -> List[str]:
        """Find potential address patterns in text"""
        addresses = []

        # Common address patterns
        patterns = [
            # Street number + name + optional unit
            r"\d{1,5}\s+[A-Za-z\s]{1,30}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Circle|Cir|Place|Pl)[.,]?(?:\s+(?:Apt|Unit|Suite|#)\s*[A-Za-z0-9-]+)?",
            # European style addresses
            r"(?:[A-Za-z\s]{1,30}(?:strasse|straße|strada|rue|via|calle|platz|plaza)\s+\d{1,5})",
            # PO Box
            r"P\.?O\.?\s*Box\s+\d{1,5}",
            # Building names with numbers
            r"(?:Building|Bldg|Tower)\s+[A-Za-z0-9-]+,\s*[A-Za-z\s]{1,50}",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                addr = match.group().strip()
                if len(addr) > 10:  # Filter out very short matches
                    addresses.append(addr)

        return list(set(addresses))  # Remove duplicates
