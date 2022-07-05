import os
import random
import random
import numpy as np
from tqdm import tqdm
from math import floor
from perlin_noise import PerlinNoise
from PIL import Image, ImageDraw, ImageFont

# Define colors
TEXT_RESET = '\033[0m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'

# Constants (image size: <1000, 219>)
image_width = 1000
image_height = 219
max_number_width = 300
font_size = 220
initial_point = (120, 35)
middle_point = (380, 35)
final_point = (680, 35)
stroke_width = 2

# Constants (image size: <200, 44>)
image_width_small = 200
image_height_small = 44
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
image_width = image_width_small
image_height = image_height_small
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
def generate_plate(plate:str) -> Image:
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
def check_plate_number(plate:str) -> bool:
    with open(output_path + 'generated.txt', 'r') as f:
        line = f.readline()
        while line != '':
            if plate in line:
                return True
            line = f.readline()
    return False

# Function to create and save a plate
def create_plate(plate:str=None, gray:bool=True) -> None:
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

# Function to create plates with random noise (gray only)
def create_noisy_plates(plate:str=None) -> None:
    noise1 = PerlinNoise(octaves=random.randint(1, 5), seed=random.randint(1, 10000))
    noise2 = PerlinNoise(octaves=random.randint(1, 10), seed=random.randint(1, 10000))
    noise3 = PerlinNoise(octaves=random.randint(1, 50), seed=random.randint(1, 10000))
    noise4 = PerlinNoise(octaves=random.randint(1, 100), seed=random.randint(1, 10000))

    xpix, ypix = image_width, image_height
    pic = []
    for i in range(ypix):
        row = []
        for j in range(xpix):
            noise_val = random.random() * noise1([i/xpix, j/ypix])
            noise_val += random.random() * noise2([i/xpix, j/ypix])
            noise_val += random.random() * noise3([i/xpix, j/ypix])
            noise_val += random.random() * noise4([i/xpix, j/ypix])

            row.append(noise_val)
        pic.append(row)

    noise_img = np.array(pic) * 255
    noise_img = noise_img.astype('int8')

    # Generate a random plate number if necessary
    if plate is None:
        plate = generate_plate_number()

    # Check if the plate has been already generated
    while (check_plate_number(plate)):
        plate = generate_plate_number()

    # Create the image
    img = generate_plate(plate)

    # Convert the image in grayscale
    img = img.convert('L')

    # Add the noise to the image
    noisy = np.asarray(img) + noise_img
    noisy = Image.fromarray(noisy)
    noisy = noisy.convert('L')

    # Save and close the image
    noisy.save(output_path + plate + '.png')
    img.close()

    # Save the plate into a text file
    with open(output_path + 'generated.txt', 'a+') as f:
        f.write(plate + '\n')
    return

# Driver function
def main(nplates:int, gray:bool, perc:int) -> None:
    # Create the output directory if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create the file "generated.txt" if not exists
    f = open(output_path + 'generated.txt', 'a+')
    f.close()

    
    # Generate "nplates" random plates
    noisy = int(nplates * perc / 100)
    if nplates - noisy:
        print(TEXT_GREEN 
            + '>> Generating {} plates in {} (CLEAR)'.format(nplates - noisy, 'GRAY' if gray == True else 'COLOR')
            + TEXT_RESET)
        for _ in tqdm(range(nplates - noisy)):
            create_plate(gray=gray)
    
    if noisy:
        print(TEXT_GREEN 
            + '>> Generating {} plates in GRAY (NOISY)'.format(noisy)
            + TEXT_RESET)
        for _ in tqdm(range(noisy)):
            create_noisy_plates()

    return

# Main function
def driver_main():
    choice = 1

    while choice != '0':
        # Get the user input
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Generate mixed normal/noisy images.')
        print('  2. Generate normal images only.')
        print('  3. Generate noisy images only.')
        print('  4. Generate coloured images.')
        print('  0. Exit.')
        choice = input(TEXT_YELLOW + 'Enter your choice: ' + TEXT_RESET)

        # Generate mixed normal/noisy images
        if choice == '1':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000

            perc = input('Enter the percentage of noisy images to generate [Enter = \"50%\"]: ')
            if perc == '':
                perc = 50

            main(nplates=int(nplates), gray=True, perc=int(perc))

        # Generate normal images only
        elif choice == '2':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000

            main(nplates=int(nplates), gray=True, perc=0)

        # Generate noisy images only
        elif choice == '3':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000

            main(nplates=int(nplates), gray=True, perc=100)

        # Generate coloured images
        elif choice == '4':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000

            main(nplates=int(nplates), gray=False, perc=0)

        # Exit
        elif choice == '0':
            print(TEXT_YELLOW + '>> Exiting.' + TEXT_RESET)
            break

        # Invalid input
        else:
            print(TEXT_YELLOW + '>> Invalid choice.' + TEXT_RESET)
            continue

    return

if __name__ == '__main__':
    driver_main()
