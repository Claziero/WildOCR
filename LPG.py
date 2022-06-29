import random
from sys import argv
from PIL import Image, ImageDraw, ImageFont
from math import floor

# Constants
max_number_width = 300
initial_point = (120, 35)
middle_point = (380, 35)
final_point = (680, 35)

# Paths
empty_plate_path = 'assets/plates/empty-plate.png'
font_path = 'assets/plates/plates1999.ttf'
output_path = 'output/'

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
    font = ImageFont.truetype(font_path, 220)

    # Draw the plate (initial letters)
    draw.text(initial_point, plate[:2], fill=(0, 0, 0), font=font, stroke_width=2)
    
    # Justify center text (central numbers)
    spaces = max_number_width - draw.textlength(plate[2:5], font=font)
    if spaces > 3:
        spaces = floor(spaces /3)
        draw.text(middle_point, plate[2], fill=(0, 0, 0), font=font, stroke_width=2)

        off1 = (draw.textlength(plate[2], font=font) + spaces + middle_point[0], middle_point[1])
        draw.text(off1, plate[3], fill=(0, 0, 0), font=font, stroke_width=2)

        off2 = (draw.textlength(plate[2:4], font=font) + spaces + middle_point[0], middle_point[1])
        draw.text(off2, plate[4], fill=(0, 0, 0), font=font, stroke_width=2)
    else:
        draw.text(middle_point, plate[2:5], fill=(0, 0, 0), font=font, stroke_width=2)
    
    # Draw the plate (final letters)
    draw.text(final_point, plate[5:], fill=(0, 0, 0), font=font, stroke_width=2)

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
def create_plate(plate:str = None) -> None:
    # Generate a random plate number if necessary
    if plate is None:
        plate = generate_plate_number()

    # Check if the plate has been already generated
    while (check_plate_number(plate)):
        plate = generate_plate_number()

    # Create the image
    img = generate_plate(plate)

    # Save and close the image
    img.save(output_path + plate + '.png')
    img.close()

    # Save the plate into a text file
    with open(output_path + 'generated.txt', 'a+') as f:
        f.write(plate + '\n')
    return

# Driver function
def main(nplates:int) -> None:
    # Create the file "generated.txt" if not exists
    f = open(output_path + 'generated.txt', 'a+')
    f.close()

    # Generate 1000 random plates
    for i in range(nplates):
        create_plate()
        if i % 100 == 0:
            print('Generated', i, 'plates out of', nplates, '(' + str(i/nplates*100) + '%)')
    
    print('Generated', nplates, 'plates out of', nplates, '(100%)')
    return


if __name__ == '__main__':
    if len(argv) == 2:
        nplates = int(argv[1])
    else:
        nplates = 1000

    main(nplates)
