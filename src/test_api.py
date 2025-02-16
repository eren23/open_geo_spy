import requests
import sys
import json


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path> [--save]")
        sys.exit(1)

    image_path = sys.argv[1]
    save_image = "--save" in sys.argv

    test_image_location(image_path, save_image)
