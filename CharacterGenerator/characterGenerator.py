"""
    This module generates random character images.
    Characters 'I', 'O', 'Q', 'U' are not allowed. 
"""

import os
import cv2
import numpy as np

from tqdm import tqdm
from common import apply_trfs
from PIL import Image, ImageDraw, ImageFont

# Define colors
TEXT_RESET = '\033[0m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'

# Define image dimensions and costants
image_width = 20
initial_image_height = 36
final_image_height = 40
font_size = 48
stroke_width = 0
initial_point = (-1, 1)

# Define paths
output_path = 'characters/'
font_path = 'assets/plates1999.ttf'

# Define possible characters
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
# Function to apply affine transformations to images
def affine_transform(img:cv2.Mat) -> cv2.Mat:
    rows, cols = img.shape[0], img.shape[1]

    p1 = [4, 4]
    p2 = [4, rows - 4]
    p3 = [cols - 4, 4]
    pts1 = np.float32([p1, p2, p3])
    
    delta = 2
    rand = np.random.randint(-delta, delta, 6)
    p1r = [p1[0] + rand[0], p1[1] + rand[1]]
    p2r = [p2[0] + rand[2], p2[1] + rand[3]]
    p3r = [p3[0] + rand[4], p3[1] + rand[5]]
    pts2 = np.float32([p1r, p2r, p3r])
    
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=255)
    return dst

# Function to generate random characters
def generate_characters(char:str, num_images:int) -> None:
    # Create a font object
    font = ImageFont.truetype(font_path, size=font_size)

    # Change image width if the character is "1"
    if char == '1': image_w = 10
    else: image_w = image_width

    # Generate "num_images" images with "char" character
    for i in tqdm(range(num_images)):
        # Create a new image of the specified dimensions
        img = Image.new(mode='L', size=(image_w, initial_image_height), color=255)
        # Create a draw object
        draw = ImageDraw.Draw(img)

        # Draw the text
        draw.text(initial_point, char, font=font, stroke_width=stroke_width)

        # Apply transformations
        img = apply_trfs(img, True)
        img = cv2.resize(img, (image_width, final_image_height))

        # Apply affine transformations
        img = affine_transform(img)

        # Save the image
        cv2.imwrite(output_path + char + '/' + char + '-' + str(i) + '.png', img)

    return

# Main function
def main(num_chars:int) -> None:
    # Create output directory if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create output subdirectories if not exist and generate images
    for letter in letters:
        if not os.path.exists(output_path + letter):
            os.makedirs(output_path + letter)
        generate_characters(letter, num_chars)
    for number in numbers:
        if not os.path.exists(output_path + number):
            os.makedirs(output_path + number)
        generate_characters(number, num_chars)
        
    return

# Driver function
def driver() -> None:
    choice = 1

    while choice != '0':
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Generate all characters.')
        print('  0. Exit.')
        choice = input(TEXT_YELLOW + 'Enter your choice: ' + TEXT_RESET)

        # Exit
        if choice == '0':
            print(TEXT_YELLOW + 'Exiting...' + TEXT_RESET)
            break

        # Generate all characters
        if choice == '1':
            n = input('Enter the number of images to generate per class [Enter = \"200\"]: ')
            if n == '':
                n = 200
            n = int(n)

            main(num_chars=n)
            continue

        # Invalid input
        else:
            print(TEXT_YELLOW + '>> Invalid choice.' + TEXT_RESET)
            continue

    return

if __name__ == '__main__':
    driver()
