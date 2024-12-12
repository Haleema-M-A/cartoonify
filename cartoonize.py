import cv2
import numpy as np
from PIL import Image
import os

def cartoonize(input_path, output_path):
    """Convert a photo to cartoon style using OpenCV"""
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return False
    
    print("Processing image...")
    
    # Convert to RGB (from BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply bilateral filter for smoothing while preserving edges
    smooth = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    
    # Convert to grayscale
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    
    # Apply median blur
    gray = cv2.medianBlur(gray, 5)
    
    # Detect and enhance edges
    edges = cv2.adaptiveThreshold(gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 9, 9)
    
    # Convert back to RGB
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Cartoon effect
    cartoon = cv2.bitwise_and(smooth, edges)
    
    # Enhance colors
    cartoon = cv2.convertScaleAbs(cartoon, alpha=1.2, beta=10)
    
    # Convert to PIL Image and save
    cartoon_pil = Image.fromarray(cartoon)
    cartoon_pil.save(output_path)
    print(f"Cartoon image saved to: {output_path}")
    return True

def process_directory(input_dir, output_dir):
    """Process all images in a directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"cartoon_{filename}")
            print(f"\nProcessing {filename}...")
            cartoonize(input_path, output_path)

if __name__ == "__main__":
    # Create directories for input and output images
    input_dir = "input_images"
    output_dir = "output_images"
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input directory: {input_dir}")
        print("Please place your images in the 'input_images' folder")
    
    if os.path.exists(input_dir) and os.listdir(input_dir):
        process_directory(input_dir, output_dir)
        print("\nAll images processed! Check the 'output_images' folder for results.")
    else:
        print("\nNo images found in input directory.")
        print("Please place your images in the 'input_images' folder and run the script again.")
