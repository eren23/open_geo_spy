import requests
import sys
import os
import traceback
from typing import List, Dict, Any, Optional
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("geolocator_client")


class GeoLocatorClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.logger = logger

    async def locate_image(self, image_path: str, save: bool = False) -> Dict[str, Any]:
        """
        Locate an image using the API.

        Args:
            image_path: Path to the image file
            save: Whether to save debug information

        Returns:
            API response as dictionary

        Raises:
            FileNotFoundError: If image file doesn't exist
            RequestException: If API request fails
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as f:
                self.logger.info(f"Sending locate request for image: {image_path}")
                response = await requests.post(f"{self.base_url}/api/locate", files={"image": f}, params={"save_image": str(save).lower()})

                if response.status_code != 200:
                    self.logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    response.raise_for_status()

                return response.json()

        except FileNotFoundError as e:
            self.logger.error(f"File error: {str(e)}")
            raise
        except requests.RequestException as e:
            self.logger.error(f"API request error: {str(e)}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise

    async def analyze_multimodal(self, files: List[str], save: bool = False) -> Dict[str, Any]:
        """
        Analyze multiple files using the API.

        Args:
            files: List of file paths to analyze
            save: Whether to save debug information

        Returns:
            API response as dictionary

        Raises:
            FileNotFoundError: If any file doesn't exist
            RequestException: If API request fails
        """
        try:
            # Validate files exist
            for file_path in files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

            # Prepare file objects
            file_objs = []
            for f in files:
                try:
                    file_objs.append(("files", (os.path.basename(f), open(f, "rb"), self._get_mime_type(f))))
                except Exception as e:
                    self.logger.error(f"Error preparing file {f}: {str(e)}")
                    # Clean up any opened files
                    for obj in file_objs:
                        try:
                            obj[1][1].close()
                        except:
                            pass
                    raise

            self.logger.info(f"Sending multimodal analysis request for {len(files)} files")
            response = await requests.post(f"{self.base_url}/api/analyze-multimodal", files=file_objs, data={"save_files": str(save).lower()})

            # Clean up files
            for obj in file_objs:
                try:
                    obj[1][1].close()
                except:
                    pass

            if response.status_code != 200:
                self.logger.error(f"API request failed with status {response.status_code}: {response.text}")
                response.raise_for_status()

            return response.json()

        except FileNotFoundError as e:
            self.logger.error(f"File error: {str(e)}")
            raise
        except requests.RequestException as e:
            self.logger.error(f"API request error: {str(e)}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type for a file based on extension"""
        ext = filename.split(".")[-1].lower()
        if ext in {"jpg", "jpeg"}:
            return "image/jpeg"
        elif ext == "png":
            return "image/png"
        elif ext in {"mp4", "mov"}:
            return "video/mp4"
        else:
            self.logger.warning(f"Unknown file type for extension: {ext}")
            return "application/octet-stream"


if __name__ == "__main__":

    async def main():
        try:
            client = GeoLocatorClient()

            if len(sys.argv) < 3:
                print("Usage: python test_api.py <command> <args>")
                print("Commands:")
                print("  image <image_path> [--save]")
                print("  multimodal <file1> [file2 ...] [--save]")
                sys.exit(1)

            command, *args = sys.argv[1:]

            if command == "image":
                if not args:
                    print("Error: Missing image path")
                    sys.exit(1)
                result = await client.locate_image(args[0], "--save" in args)
                print("Response:", result)

            elif command == "multimodal":
                files = [a for a in args if not a.startswith("--")]
                if not files:
                    print("Error: No files specified")
                    sys.exit(1)
                result = await client.analyze_multimodal(files, "--save" in args)
                print("Response:", result)

            else:
                print(f"Error: Unknown command '{command}'")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            sys.exit(1)

    asyncio.run(main())
