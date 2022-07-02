import random
import os
from tqdm import tqdm
from sys import argv
from PIL import Image, ImageDraw, ImageFont
from math import floor

# Define colors
TEXT_GREEN = '\033[92m'
TEXT_RESET = '\033[0m'

# Constants (image size: <1000, 219>)
max_number_width = 300
font_size = 220
initial_point = (120, 35)
middle_point = (380, 35)
final_point = (680, 35)
stroke_width = 2

# Constants (image size: <200, 44>)
max_number_width_small = 65
font_size_small = 40
initial_point_small = (25, 8)
middle_point_small = (77, 8)
final_point_small = (137, 8)
stroke_width_small = 0

# Paths
empty_plate_path = 'assets/plates/empty-plate.png'
empty_plate_path_small = 'assets/plates/empty-plate-small.png'
font_path = 'assets/plates/plates1999.ttf'
output_path = 'output/'

# Using small plate
max_number_width = max_number_width_small
font_size = font_size_small
initial_point = initial_point_small
middle_point = middle_point_small
final_point = final_point_small
empty_plate_path = empty_plate_path_small
stroke_width = stroke_width_small

# Function to generate a random number plate
def generate_plate_number() -> str:
    # List of possible characters
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plate = ""

    # Generate a random plate of 7 characters
    for _ in range(2):
        plate += letters[int(random.random() * len(letters))]
    for _ in range(3):
        plate += numbers[int(random.random() * len(numbers))]
    for _ in range(2):
        plate += letters[int(random.random() * len(letters))]
    
    return plate

# Function to create an image with the given plate
def generate_plate(plate: str) -> Image:
    # Open base image
    img = Image.open(empty_plate_path)
    # Create a draw object
    draw = ImageDraw.Draw(img)
    # Create a font object
    font = ImageFont.truetype(font_path, font_size)

    # Draw the plate (initial letters)
    draw.text(initial_point, plate[:2], fill=(0, 0, 0), font=font, stroke_width=stroke_width)
    
    # Justify center text (central numbers)
    spaces = max_number_width - draw.textlength(plate[2:5], font=font)
    if spaces > 3:
        spaces = floor(spaces /3)
        draw.text(middle_point, plate[2], fill=(0, 0, 0), font=font, stroke_width=stroke_width)

        off1 = (draw.textlength(plate[2], font=font) + spaces + middle_point[0], middle_point[1])
        draw.text(off1, plate[3], fill=(0, 0, 0), font=font, stroke_width=stroke_width)

        off2 = (draw.textlength(plate[2:4], font=font) + spaces + middle_point[0], middle_point[1])
        draw.text(off2, plate[4], fill=(0, 0, 0), font=font, stroke_width=stroke_width)
    else:
        draw.text(middle_point, plate[2:5], fill=(0, 0, 0), font=font, stroke_width=stroke_width)
    
    # Draw the plate (final letters)
    draw.text(final_point, plate[5:], fill=(0, 0, 0), font=font, stroke_width=stroke_width)

    return img

# Function to check if a plate has been already created
def check_plate_number(plate: str) -> bool:
    with open(output_path + 'generated.txt', 'r') as f:
        line = f.readline()
        while line != '':
            if plate in line:
                return True
            line = f.readline()
    return False

# Function to create and save a plate
def create_plate(plate:str = None, gray:bool = True) -> None:
    # Generate a random plate number if necessary
    if plate is None:
        plate = generate_plate_number()

    # Check if the plate has been already generated
    while (check_plate_number(plate)):
        plate = generate_plate_number()

    # Create the image
    img = generate_plate(plate)

    # Convert the image in grayscale if necessary
    if gray:
        img = img.convert('L')

    # Save and close the image
    img.save(output_path + plate + '.png')
    img.close()

    # Save the plate into a text file
    with open(output_path + 'generated.txt', 'a+') as f:
        f.write(plate + '\n')
    return

# Driver function
def main(nplates:int, gray:bool) -> None:
    # Create the output directory if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create the file "generated.txt" if not exists
    f = open(output_path + 'generated.txt', 'a+')
    f.close()

    print(TEXT_GREEN 
        + '>> Generating {} plates in {}'.format(nplates, 'GRAY' if gray == True else 'COLOR')
        + TEXT_RESET)
    
    # Generate 1000 random plates
    for _ in tqdm(range(nplates)):
        create_plate(gray = gray)
    return


if __name__ == '__main__':
    print('Usage: py LPG.py <number of plates>[def:1000] <color|gray>[def:gray]')
    if len(argv) == 3:
        main(int(argv[1]), argv[2] == 'gray')
    elif len(argv) == 2:
        main(int(argv[1]), True)
    else:
        main(1000, True)
