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

# Constants (image size: <200, 44>)
image_width = 200
image_height = 44
max_number_width = 65
font_size = 40
initial_point = (25, 8)
middle_point = (77, 8)
final_point = (137, 8)
stroke_width = 0

# Constants for moto plates (image size: <106, 83>)
moto_image_width = 106
moto_image_height = 83
moto_max_number_width = 90 ###############################
moto_font_size = 40
moto_initial_point = (25, 8) #############################
moto_middle_point = (8, 47) ##############################
moto_final_point = (0, 0) ###############################
moto_stroke_width = 0

# Paths
empty_plate_path = 'assets/plates/empty-plate-small.png'
moto_empty_plate_path = 'assets/plates/empty-plate-small-moto.png'
font_path = 'assets/plates/plates1999.ttf'
output_path = 'output/'

# Function to generate a random number plate
def generate_plate_number(moto:bool=False) -> str:
    # List of possible characters ('I', 'O', 'Q', 'U' are not allowed)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plate = ""

    # Generate a random plate of 7 characters
    for _ in range(2):
        plate += letters[int(random.random() * len(letters))]
    for _ in range(3):
        plate += numbers[int(random.random() * len(numbers))]
    for _ in range(2):
        if moto:
            plate += numbers[int(random.random() * len(numbers))]
        else:
            plate += letters[int(random.random() * len(letters))]
    
    return plate

# Function to create an image with the given plate
def generate_plate(plate:str, moto:bool=False) -> Image:
    if not moto:
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
            spaces = floor(spaces / 3)
            draw.text(middle_point, plate[2], fill=(0, 0, 0), font=font, stroke_width=stroke_width)

            off1 = (draw.textlength(plate[2], font=font) + spaces + middle_point[0], middle_point[1])
            draw.text(off1, plate[3], fill=(0, 0, 0), font=font, stroke_width=stroke_width)

            off2 = (draw.textlength(plate[2:4], font=font) + 2 * spaces + middle_point[0], middle_point[1])
            draw.text(off2, plate[4], fill=(0, 0, 0), font=font, stroke_width=stroke_width)
        else:
            draw.text(middle_point, plate[2:5], fill=(0, 0, 0), font=font, stroke_width=stroke_width)
        
        # Draw the plate (final letters)
        draw.text(final_point, plate[5:], fill=(0, 0, 0), font=font, stroke_width=stroke_width)
        return img
    else:
        # Open base image
        img = Image.open(moto_empty_plate_path)
        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Create a font object
        font = ImageFont.truetype(font_path, moto_font_size)

        # Draw the plate (initial letters)
        draw.text(moto_initial_point, plate[:2], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)
        
        # Justify center text (numbers)
        spaces = moto_max_number_width - draw.textlength(plate[2:], font=font)
        if spaces > 5:
            spaces = floor(spaces / 5)
            draw.text(moto_middle_point, plate[2], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)

            off1 = (draw.textlength(plate[2], font=font) + spaces + moto_middle_point[0], moto_middle_point[1])
            draw.text(off1, plate[3], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)

            off2 = (draw.textlength(plate[2:4], font=font) + 2 * spaces + moto_middle_point[0], moto_middle_point[1])
            draw.text(off2, plate[4], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)

            off3 = (draw.textlength(plate[2:5], font=font) + 3 * spaces + moto_middle_point[0], moto_middle_point[1])
            draw.text(off3, plate[5], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)

            off4 = (draw.textlength(plate[2:6], font=font) + 4 * spaces + moto_middle_point[0], moto_middle_point[1])
            draw.text(off4, plate[6], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)
        else:
            draw.text(moto_middle_point, plate[2:], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)
        
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
def create_plate(plate:str=None, gray:bool=True, moto:bool=False) -> None:
    # Generate a random plate number if necessary
    if plate is None:
        plate = generate_plate_number(moto)

    # Check if the plate has been already generated
    while check_plate_number(plate):
        plate = generate_plate_number(moto)

    # Create the image
    img = generate_plate(plate, moto)

    # Convert the image in grayscale if necessary
    if gray:
        img = img.convert('L')

    # Save and close the image
    img.save(output_path + plate + '{}.png'.format('m' if moto else ''))
    img.close()

    # Save the plate into a text file
    with open(output_path + 'generated.txt', 'a+') as f:
        f.write(plate + '\n')
    return

# Function to create plates with random noise (gray only)
def create_noisy_plates(plate:str=None, moto:bool=False) -> None:
    noise1 = PerlinNoise(octaves=random.randint(1, 5), seed=random.randint(1, 10000))
    noise2 = PerlinNoise(octaves=random.randint(1, 10), seed=random.randint(1, 10000))
    noise3 = PerlinNoise(octaves=random.randint(1, 50), seed=random.randint(1, 10000))
    noise4 = PerlinNoise(octaves=random.randint(1, 100), seed=random.randint(1, 10000))

    if moto:
        xpix, ypix = moto_image_width, moto_image_height
    else:
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
        plate = generate_plate_number(moto)

    # Check if the plate has been already generated
    while (check_plate_number(plate)):
        plate = generate_plate_number(moto)

    # Create the image
    img = generate_plate(plate, moto)

    # Convert the image in grayscale
    img = img.convert('L')

    # Add the noise to the image
    noisy = np.asarray(img) + noise_img
    noisy = Image.fromarray(noisy)
    noisy = noisy.convert('L')

    # Save and close the image
    noisy.save(output_path + plate + '{}.png'.format('m' if moto else ''))
    img.close()

    # Save the plate into a text file
    with open(output_path + 'generated.txt', 'a+') as f:
        f.write(plate + '\n')
    return

# Driver function
def main(nplates:int, gray:bool, perc:int, moto:bool=False) -> None:
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
            + '>> Generating {} plates in {} (CLEAR) ({})'.format(nplates - noisy,
            'GRAY' if gray == True else 'COLOR', 'MOTO' if moto else 'AUTO')
            + TEXT_RESET)
        for _ in tqdm(range(nplates - noisy)):
            create_plate(gray=gray, moto=moto)
    
    if noisy:
        print(TEXT_GREEN 
            + '>> Generating {} plates in GRAY (NOISY) ({})'.format(noisy, 'MOTO' if moto else 'AUTO')
            + TEXT_RESET)
        for _ in tqdm(range(noisy)):
            create_noisy_plates(moto=moto)

    return

# Main function
def driver_main():
    choice = 1

    while choice != '0':
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Generate car and moto plates.')
        print('  2. Generate car plates only.')
        print('  3. Generate moto plates only.')
        print('  0. Exit.')
        choice = input(TEXT_YELLOW + 'Enter your choice: ' + TEXT_RESET)

        # Generate car and moto plates
        if choice == '1':
            ratio = input('Enter the percentage of car plates [Enter = \"50%\"]: ')
            if ratio == '':
                ratio = 50
            ratio = int(ratio)

        # Generate car plates only
        elif choice == '2':
            ratio = 100

        # Generate moto plates only
        elif choice == '3':
            ratio = 0

        # Exit
        elif choice == '0':
            print(TEXT_YELLOW + '>> Exiting...' + TEXT_RESET)
            break

        # Invalid choice
        else:
            print(TEXT_YELLOW + '>> Invalid choice. Try again.' + TEXT_RESET)
            continue

        # Get the user input
        print(TEXT_YELLOW + '>> Select the type of the images. Type:' + TEXT_RESET)
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
            nplates = int(nplates)

            perc = input('Enter the percentage of noisy images to generate [Enter = \"50%\"]: ')
            if perc == '':
                perc = 50

            main(nplates=int(nplates * ratio/100), gray=True, perc=int(perc), moto=False)
            main(nplates=int(nplates * (100-ratio)/100), gray=True, perc=int(perc), moto=True)

        # Generate normal images only
        elif choice == '2':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000
            nplates = int(nplates)

            main(nplates=int(nplates * ratio/100), gray=True, perc=0, moto=False)
            main(nplates=int(nplates * (100-ratio)/100), gray=True, perc=0, moto=True)

        # Generate noisy images only
        elif choice == '3':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000
            nplates = int(nplates)

            main(nplates=int(nplates * ratio/100), gray=True, perc=100, moto=False)
            main(nplates=int(nplates * (100-ratio)/100), gray=True, perc=100, moto=True)

        # Generate coloured images
        elif choice == '4':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000
            nplates = int(nplates)

            main(nplates=int(nplates * ratio/100), gray=False, perc=0, moto=False)
            main(nplates=int(nplates * (100-ratio)/100), gray=False, perc=0, moto=True)

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
