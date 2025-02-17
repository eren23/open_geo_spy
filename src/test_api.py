import requests
import sys
import json
import os
from typing import List


def test_image_location(image_path: str, save_image: bool = False):
    """Test the image location API endpoint"""
    url = "http://localhost:8000/api/locate"

    # Prepare the file
    with open(image_path, "rb") as f:
        files = {"image": (image_path, f, "image/jpeg")}
        params = {"save_image": "true" if save_image else "false"}

        # Make the request
        response = requests.post(url, files=files, params=params)

    # Print the results
    if response.status_code == 200:
        result = response.json()
        print("\nLocation Results:")
        print(f"Name: {result['location']['name']}")
        print(f"Coordinates: {result['location']['lat']}, {result['location']['lon']}")
        print(f"Confidence: {result['location']['confidence']}")
        print("\nAnalysis:")
        print(result["analysis"]["description"])

        if result["image_saved"]:
            print(f"\nImage saved as: {result['filename']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_multimodal_analysis(file_paths: List[str], save_files: bool = False):
    """Test the multimodal analysis API endpoint"""
    url = "http://localhost:8000/api/analyze-multimodal"

    # Prepare the files
    files = []
    for path in file_paths:
        with open(path, "rb") as f:
            mime_type = "image/jpeg" if path.endswith((".jpg", ".jpeg", ".png")) else "video/mp4"
            files.append(("files", (os.path.basename(path), f, mime_type)))

    # Add the save_files parameter
    data = {"save_files": str(save_files).lower()}

    # Make the request
    response = requests.post(url, files=files, data=data)

    # Print the results
    if response.status_code == 200:
        result = response.json()
        print("\nAnalysis Results:")
        print(result["analysis"])

        if result["files_saved"]:
            print("\nSaved files:")
            for filename in result["files_saved"]:
                print(f"- {filename}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_api.py <command> <args>")
        print("Commands:")
        print("  image <image_path> [--save]")
        print("  multimodal <file1> [file2...] [--save]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "image":
        image_path = sys.argv[2]
        save_image = "--save" in sys.argv
        test_image_location(image_path, save_image)
    elif command == "multimodal":
        files = []
        save_files = "--save" in sys.argv

        i = 2
        while i < len(sys.argv):
            if sys.argv[i] not in ("--save"):
                files.append(sys.argv[i])
            i += 1

        if not files:
            print("Error: Must provide at least one file")
            sys.exit(1)

        test_multimodal_analysis(files, save_files)
