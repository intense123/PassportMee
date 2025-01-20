import cv2
import numpy as np
import os

def detect_glasses(image, face_x, face_y, face_w, face_h):
    """
    Detect if a person is wearing glasses using a combination of eye cascade
    and edge detection in the eye region
    Returns: bool indicating if glasses are detected
    """
    # Load eye cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    
    # Define the eye region (middle third of face)
    eye_region_y = face_y + int(face_h * 0.2)  # Start 20% down from top of face
    eye_region_h = int(face_h * 0.3)  # Take 30% of face height
    eye_region = image[eye_region_y:eye_region_y + eye_region_h, face_x:face_x + face_w]
    
    # Convert to grayscale for eye detection
    if len(image.shape) == 3:
        gray_eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_eye_region = eye_region

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray_eye_region, 
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(int(face_w/6), int(face_h/6)))
    
    # If we detect 2 eyes, apply edge detection for glasses
    if len(eyes) >= 2:
        # Apply edge detection to eye region
        edges = cv2.Canny(gray_eye_region, 30, 150)
        
        # Calculate edge density in eye region
        edge_pixels = np.count_nonzero(edges)
        region_area = gray_eye_region.shape[0] * gray_eye_region.shape[1]
        edge_density = edge_pixels / region_area
        
        # Higher edge density in eye region often indicates glasses
        return edge_density > 0.15
    
    return False

def detect_cap(image, face_y, face_h):
    """
    Detect if a person is wearing a cap by analyzing the area above their face
    Returns: bool indicating if a cap is detected
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Define the region of interest (ROI) above the face
    cap_region_height = int(face_h * 0.5)  # Check area above face
    cap_region_y = max(0, face_y - cap_region_height)
    
    # Extract the ROI
    roi = gray[cap_region_y:face_y, :]
    
    # Apply edge detection
    edges = cv2.Canny(roi, 50, 150)
    
    # Count the number of edge pixels
    edge_pixels = np.count_nonzero(edges)
    
    # Calculate edge density
    roi_area = roi.shape[0] * roi.shape[1]
    edge_density = edge_pixels / roi_area
    
    # If edge density is below threshold, likely a cap is present
    return edge_density < 0.1  # Adjust threshold as needed

def process_image(image):
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
    default_directory = "F:/6th Semester/Passport me/image-processing/"
    os.chdir(default_directory)

    # Load the image
    image = cv2.imread('image_rayhan.jpg')
    processed_image = process_image(image)

    if processed_image is not None:
        # Save the processed image
        cv2.imwrite('processed_image.jpg', processed_image)
        print("Processed image saved as 'processed_image.jpg'")