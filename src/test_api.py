import requests
import sys
import os
from typing import List
import asyncio


class GeoLocatorClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def locate_image(self, image_path: str, save: bool = False) -> dict:
        with open(image_path, "rb") as f:
            response = await requests.post(f"{self.base_url}/api/locate", files={"image": f}, params={"save_image": str(save).lower()})
            return response.json()

    async def analyze_multimodal(self, files: List[str], save: bool = False) -> dict:
        file_objs = [(os.path.basename(f), open(f, "rb"), self._get_mime_type(f)) for f in files]
        response = await requests.post(
            f"{self.base_url}/api/analyze-multimodal", files=[("files", f) for f in file_objs], data={"save_files": str(save).lower()}
        )
        return response.json()

    def _get_mime_type(self, filename: str) -> str:
        ext = filename.split(".")[-1].lower()
        return "image/jpeg" if ext in {"jpg", "jpeg", "png"} else "video/mp4"


if __name__ == "__main__":

    async def main():
        client = GeoLocatorClient()

        if len(sys.argv) < 3:
            print("Usage: python test_api.py <command> <args>")
            sys.exit(1)

        command, *args = sys.argv[1:]
        if command == "image":
            print(await client.locate_image(args[0], "--save" in args))
        elif command == "multimodal":
            files = [a for a in args if not a.startswith("--")]
            print(await client.analyze_multimodal(files, "--save" in args))

    asyncio.run(main())
