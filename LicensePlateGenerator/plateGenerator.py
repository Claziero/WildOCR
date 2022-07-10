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

# Define plate types
PLATE_TYPES = ['car', 'moto']

# Dimension constants
# Car plates are 200x44 pixels
car_image_width, car_image_height = 200, 44
car_font_size = 40
car_stroke_width = 0

# Moto plates are 106x83 pixels
moto_image_width, moto_image_height = 106, 83
moto_font_size = 40
moto_stroke_width = 0

# Constants for car plates (image size: <200, 44>)
car_max_number_width = 65
car_initial_point = (25, 8)
car_middle_point = (77, 8)
car_final_point = (137, 8)

# Constants for moto plates (image size: <106, 83>)
moto_max_number_width = 90
moto_initial_point = (25, 8)
moto_middle_point = (8, 47)

# Paths
car_empty_plate_path = 'assets/plates/empty-plate-car.png'
moto_empty_plate_path = 'assets/plates/empty-plate-moto.png'
carabinieri_empty_plate_path = 'assets/plates/empty-plate-carabinieri.png'
aeronautica_empty_plate_path = 'assets/plates/empty-plate-aeronautica-mil.png'
esercito_empty_plate_path = 'assets/plates/empty-plate-esercito.png'
marina_empty_plate_path = 'assets/plates/empty-plate-marina-mil.png'
vigili_fuoco_empty_plate_path = 'assets/plates/empty-plate-vigili-fuoco.png'
car_special_empty_plate_path = 'assets/plates/empty-plate-special-car.png'

font_path = 'assets/plates/plates1999.ttf'
output_path = 'output/'

# Auxiliar function to get the -* suffix of plate names and generated files
def get_suffix(ptype:str) -> str:
    if ptype == 'car':
        return '-car'
    if ptype == 'moto':
        return '-moto'
    
    return ''

# Auxiliar function to get the format of the plate string
def get_plate_format(ptype:str) -> list[int]:
    if ptype == 'car':
        return [2, 3, 2]
    if ptype == 'moto':
        return [2, 5, 0]
    
    return [0, 0, 0]


# Function to generate a random number plate
def generate_plate_number(initial_letters:int, central_numbers:int, final_letters:int) -> str:
    # List of possible characters ('I', 'O', 'Q', 'U' are not allowed)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plate = ""

    # Generate a random plate of 7 characters
    for _ in range(int(initial_letters)):
        plate += letters[int(random.random() * len(letters))]
    for _ in range(int(central_numbers)):
        plate += numbers[int(random.random() * len(numbers))]
    for _ in range(int(final_letters)):
        plate += letters[int(random.random() * len(letters))]
    
    return plate

# Function to create an image with the given plate
def generate_plate(plate:str, ptype:str) -> Image:
    # Car plates
    if ptype == 'car':
        # Open base image
        img = Image.open(car_empty_plate_path)
        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Create a font object
        font = ImageFont.truetype(font_path, car_font_size)

        # Draw the plate (initial letters)
        draw.text(car_initial_point, plate[:2], fill=(0, 0, 0), font=font, stroke_width=car_stroke_width)
        
        # Justify center text (central numbers)
        spaces = car_max_number_width - draw.textlength(plate[2:5], font=font)
        if spaces > 3:
            spaces = floor(spaces / 3)
            draw.text(car_middle_point, plate[2], fill=(0, 0, 0), font=font, stroke_width=car_stroke_width)

            off1 = (draw.textlength(plate[2], font=font) + spaces + car_middle_point[0], car_middle_point[1])
            draw.text(off1, plate[3], fill=(0, 0, 0), font=font, stroke_width=car_stroke_width)

            off2 = (draw.textlength(plate[2:4], font=font) + 2 * spaces + car_middle_point[0], car_middle_point[1])
            draw.text(off2, plate[4], fill=(0, 0, 0), font=font, stroke_width=car_stroke_width)
        else:
            draw.text(car_middle_point, plate[2:5], fill=(0, 0, 0), font=font, stroke_width=car_stroke_width)
        
        # Draw the plate (final letters)
        draw.text(car_final_point, plate[5:], fill=(0, 0, 0), font=font, stroke_width=car_stroke_width)
        return img
    
    # Moto plates
    elif ptype == 'moto':
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

    # Incorrect plate type
    else:
        return None

# Function to check if a plate has been already created
def check_plate_number(plate:str, ptype:str) -> bool:
    with open(output_path + 'generated{}.txt'.format(get_suffix(ptype)), 'r') as f:
        line = f.readline()
        while line != '':
            if plate in line:
                return True
            line = f.readline()
    return False

# Function to create and save a plate
def create_plate(gray:bool=True, ptype:str='car') -> None:
    # Generate a random plate number
    seq = get_plate_format(ptype)
    plate = generate_plate_number(seq[0], seq[1], seq[2])

    # Check if the plate has been already generated
    while check_plate_number(plate, ptype=ptype):
        plate = generate_plate_number(seq[0], seq[1], seq[2])

    # Create the image
    img = generate_plate(plate, ptype)

    # Convert the image in grayscale if necessary
    if gray:
        img = img.convert('L')

    # Save and close the image
    img.save(output_path + plate + '{}.png'.format(get_suffix(ptype)))
    img.close()

    # Save the plate into a text file
    with open(output_path + 'generated{}.txt'.format(get_suffix(ptype)), 'a+') as f:
        f.write(plate + '\n')
    return

# Function to generate a random noise image
def generate_noise_image(width:int=1000, height:int=1000) -> np.ndarray:
    noise1 = PerlinNoise(octaves=random.randint(20, 100), seed=random.randint(1, 10000))
    noise2 = PerlinNoise(octaves=random.randint(50, 100), seed=random.randint(1, 10000))

    pic = []
    for i in range(height):
        row = []
        for j in range(width):
            noise_val = random.random() * noise1([i/width, j/height])
            noise_val += random.random() * noise2([i/width, j/height])

            row.append(noise_val)
        pic.append(row)

    return np.array(pic) * 255

# Function to create plates with random noise (gray only)
def create_noisy_plate(ptype:str='car', noise:np.ndarray=None) -> None:
    if ptype == 'car':
        xpix, ypix = car_image_width, car_image_height
    elif ptype == 'moto':
        xpix, ypix = moto_image_width, moto_image_height

    # Cut a random window from the noise image of dimensions (xpix, ypix)
    x = random.randint(0, 1000 - xpix)
    y = random.randint(0, 1000 - ypix)
    noise_img = noise[y:int(y+ypix), x:int(x+xpix)]
    noise_img = noise_img.astype('int8')

    # Generate a random plate number
    seq = get_plate_format(ptype)
    plate = generate_plate_number(seq[0], seq[1], seq[2])

    # Check if the plate has been already generated
    while (check_plate_number(plate, ptype)):
        plate = generate_plate_number(seq[0], seq[1], seq[2])

    # Create the image
    img = generate_plate(plate, ptype)

    # Convert the image in grayscale
    img = img.convert('L')

    # Add the noise to the image
    noisy = np.asarray(img) + noise_img
    noisy = Image.fromarray(noisy)
    noisy = noisy.convert('L')

    # Save and close the image
    noisy.save(output_path + plate + '{}.png'.format(get_suffix(ptype)))
    img.close()

    # Save the plate into a text file
    with open(output_path + 'generated{}.txt'.format(get_suffix(ptype)), 'a+') as f:
        f.write(plate + '\n')
    return

# Driver function
def main(nplates:int, gray:bool, perc:int, ptype:str, new_noise:int=1000) -> None:
    # Create the output directory if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create the files "generated.txt" and "generated-m.txt" if not existing
    for t in PLATE_TYPES:
        f = open(output_path + 'generated{}.txt'.format(get_suffix(t)), 'a+')
        f.close()

    # Generate "nplates" random plates
    noisy = int(nplates * perc / 100)
    if nplates - noisy:
        print(TEXT_GREEN 
            + '>> Generating {} plates in {} (CLEAR) ({})'.format(nplates - noisy,
            'GRAY' if gray == True else 'COLOR', ptype)
            + TEXT_RESET)
        for _ in tqdm(range(nplates - noisy)):
            create_plate(gray=gray, ptype=ptype)
    
    if noisy:
        print(TEXT_GREEN 
            + '>> Generating {} plates in GRAY (NOISY) ({})'.format(noisy, ptype)
            + TEXT_RESET)
        for i in tqdm(range(noisy)):
            # Every "new_noise" iterations regenerate the noise image
            if i % new_noise == 0:
                noise = generate_noise_image()
            create_noisy_plate(ptype=ptype, noise=noise)

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
            perc = int(perc)

            new_noise = input('Regenerate the noise image every [Enter = \"1000\"] plates: ')
            if new_noise == '':
                new_noise = 1000
            new_noise = int(new_noise)            

            main(nplates=int(nplates * ratio/100), gray=True, perc=perc, ptype='car')
            main(nplates=int(nplates * (100-ratio)/100), gray=True, perc=perc, ptype='moto')

        # Generate normal images only
        elif choice == '2':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000
            nplates = int(nplates)

            main(nplates=int(nplates * ratio/100), gray=True, perc=0, ptype='car')
            main(nplates=int(nplates * (100-ratio)/100), gray=True, perc=0, ptype='moto')

        # Generate noisy images only
        elif choice == '3':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000
            nplates = int(nplates)

            new_noise = input('Regenerate the noise image every [Enter = \"1000\"] plates: ')
            if new_noise == '':
                new_noise = 1000
            new_noise = int(new_noise)

            main(nplates=int(nplates * ratio/100), gray=True, perc=100, ptype='car')
            main(nplates=int(nplates * (100-ratio)/100), gray=True, perc=100, ptype='moto')

        # Generate coloured images
        elif choice == '4':
            nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
            if nplates == '':
                nplates = 1000
            nplates = int(nplates)

            main(nplates=int(nplates * ratio/100), gray=False, perc=0, ptype='car')
            main(nplates=int(nplates * (100-ratio)/100), gray=False, perc=0, ptype='moto')

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
