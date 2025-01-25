import cv2
import numpy as np
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, render_template

app = Flask(__name__)

load_dotenv()

client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

def encode_image(image):
    """
    Encode an OpenCV image to base64 for API submission
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Encode image to JPEG
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def detect_glasses(image, face_x, face_y, face_w, face_h):
    """
    Detect if glasses are present in the image
    
    Args:
    - image: OpenCV image
    - face_x, face_y: Top-left coordinates of the detected face
    - face_w, face_h: Width and height of the detected face
    
    Returns:
    - Boolean indicating presence of glasses
    """
    try:
        # Crop the face region
        face_region = image[face_y:face_y+face_h, face_x:face_x+face_w]
        
        # Encode the face region
        base64_image = encode_image(face_region)
        
        # Call OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Are there glasses covering the eyes in this image? Respond with only 'Yes' or 'No'.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=10
        )
        
        # Extract the response
        glasses_response = response.choices[0].message.content.lower().strip()
        
        # Return True if glasses are detected
        return glasses_response == 'yes'
    
    except Exception as e:
        print(f"Error in glasses detection: {e}")
        return False

def detect_cap(image, face_y, face_h):
    """
    Detect if a cap or headwear is present in the image
    
    Args:
    - image: OpenCV image
    - face_y: Y-coordinate of the face
    - face_h: Height of the face
    
    Returns:
    - Boolean indicating presence of a cap
    """
    try:
        # Crop the top region above the face
        top_region_height = int(face_h * 0.5)  # Take half the face height as top region
        top_region = image[max(0, face_y - top_region_height):face_y, :]
        
        # Encode the top region
        base64_image = encode_image(top_region)
        
        # Call OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Is there a cap, hat, or any headwear in this image? Respond with only 'Yes' or 'No'.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=10
        )
        
        # Extract the response
        cap_response = response.choices[0].message.content.lower().strip()
        
        # Return True if cap is detected
        return cap_response == 'yes'
    
    except Exception as e:
        print(f"Error in cap detection: {e}")
        return False

@app.route('/process_image/<image_path>', method=['GET'])
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    print(f"Original image shape: {image.shape}")

    # Check if the image is grayscale
    if len(image.shape) == 2 or (len(image.shape) == 3 and np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2])):
        print("The image is black and white (grayscale).")
        return
    else:
        print("The image is colored (RGB).")

    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) != 1:
        print(f"Error: Detected {len(faces)} faces. The image must contain exactly one face.")
        return

    # Get face coordinates
    x, y, w, h = faces[0]
    print(f"Face detected at x={x}, y={y}, w={w}, h={h}")

    # Check for cap
    if detect_cap(image, y, h):
        print("Error: Cap detected. Please remove any headwear for passport photo.")
        return
    
    # Check for glasses
    if detect_glasses(image, x, y, w, h):
        print("Error: Glasses detected. Please remove glasses for passport photo.")
        return

    # Create a mask for the face
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1

    # Ensure 70-80% of the image is the face
    padding = int(min(h, w) * 0.1)  # Add 10% padding around the face
    x_start = max(x - padding, 0)
    y_start = max(y - padding, 0)
    x_end = min(x + w + padding, image.shape[1])
    y_end = min(y + h + padding, image.shape[0])
    cropped_image = image[y_start:y_end, x_start:x_end]

    # Resize the image to 35mm x 45mm (350x450 pixels for high resolution)
    new_height = 450
    new_width = 350
    resized_image = cv2.resize(cropped_image, (new_width, new_height))

    # Display the processed image
    cv2.imshow('Processed Passport Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resized_image

if __name__ == "__main__":
    # Set the working directory
    default_directory = "F:/6th Semester/PassportMee/image-processing/"
    if not os.path.exists(default_directory):
        print(f"Default directory not found: {default_directory}")
        default_directory = input("Please enter a valid directory path: ")
    os.chdir(default_directory)

    # Load the image
    # image = cv2.imread('rayhan_cap.jpg')
    # processed_image = process_image(image)

    # if processed_image is not None:
    #     # Save the processed image
    #     cv2.imwrite('processed_image.jpg', processed_image)
    #     print("Processed image saved as 'processed_image.jpg'")

    app.run(debug=True)