import os
import cv2
import random
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from math import floor
from perlin_noise import PerlinNoise
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Define colors
TEXT_RESET = '\033[0m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'

# Define plate types
PLATE_TYPES = ['auto', 'moto', 'aeronautica', 'carabinieri', 'esercito', 'marina', 'vigili_fuoco', 'auto_sp']

# Dimension constants
# Car plates are 200x44 pixels
auto_image_width, auto_image_height = 200, 44
auto_font_size = 40
auto_stroke_width = 0

# Moto plates are 106x83 pixels
moto_image_width, moto_image_height = 106, 83
moto_font_size = 40
moto_stroke_width = 0

# Constants for car plates (image size: <200, 44>)
auto_max_number_width = 65
auto_initial_point = (25, 8)
auto_middle_point = (77, 8)
auto_final_point = (137, 8)

# Constants for moto plates (image size: <106, 83>)
moto_max_number_width = 90
moto_initial_point = (25, 8)
moto_middle_point = (8, 47)

# Constants for aeronautica/carabinieri/esercito/marina plates (image size: <200, 44>)
aeronautica_max_number_width = 65
aeronautica_middle_point = (70, 6)
aeronautica_final_point = (130, 6)
aeronautica_font_size = 46

#Constants for vigili del fuoco plates (image size: <200, 44>)
vigili_fuoco_max_number_width = 115
vigili_fuoco_middle_point = (77, 6)
vigili_fuoco_font_size = 46

# Constants for auto special plates (image size: <106, 83>)
auto_sp_max_number_width = 46
auto_sp_initial_point = (36, 8)
auto_sp_middle_point = (5, 46)
auto_sp_final_point = (65, 46)

# Paths
auto_empty_plate_path = 'assets/empty-plate-auto.png'
moto_empty_plate_path = 'assets/empty-plate-moto.png'
carabinieri_empty_plate_path = 'assets/empty-plate-carabinieri.png'
aeronautica_empty_plate_path = 'assets/empty-plate-aeronautica-mil.png'
esercito_empty_plate_path = 'assets/empty-plate-esercito.png'
marina_empty_plate_path = 'assets/empty-plate-marina-mil.png'
vigili_fuoco_empty_plate_path = 'assets/empty-plate-vigili-fuoco.png'
auto_sp_empty_plate_path = 'assets/empty-plate-special-auto.png'

font_path = 'assets/plates1999.ttf'
output_path = 'output/'
chars_path = 'chars/'

# Auxiliar function to get the -* suffix of plate names and generated files
def get_suffix(ptype:str) -> str:
    if ptype == 'auto':
        return '-auto'
    if ptype == 'moto':
        return '-moto'
    if ptype == 'aeronautica':
        return '-aero'
    if ptype == 'carabinieri':
        return '-cara'
    if ptype == 'esercito':
        return '-eser'
    if ptype == 'marina':
        return '-mari'
    if ptype == 'vigili_fuoco':
        return '-vigf'
    if ptype == 'auto_sp':
        return '-ausp'
    
    return ''

# Auxiliar function to get the format of the plate string
def get_plate_format(ptype:str) -> list[int]:
    if ptype == 'auto' or ptype == 'auto_sp':
        return [2, 3, 2]
    if ptype == 'moto':
        return [2, 5, 0]
    if ptype == 'aeronautica' or ptype == 'carabinieri'\
        or ptype == 'esercito' or ptype == 'marina':
        return [2, 3, 0]
    if ptype == 'vigili_fuoco':
        return [0, 5, 0]
    
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
def generate_plate(plate:str, ptype:str) -> Image.Image:
    # Car plates
    if ptype == 'auto':
        # Open base image
        img = Image.open(auto_empty_plate_path)
        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Create a font object
        font = ImageFont.truetype(font_path, auto_font_size)

        # Draw the plate (initial letters)
        draw.text(auto_initial_point, plate[:2], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)
        
        # Justify center text (central numbers)
        spaces = auto_max_number_width - draw.textlength(plate[2:5], font=font)
        if spaces > 3:
            spaces = floor(spaces / 3)
            draw.text(auto_middle_point, plate[2], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)

            off1 = (draw.textlength(plate[2], font=font) + spaces + auto_middle_point[0], auto_middle_point[1])
            draw.text(off1, plate[3], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)

            off2 = (draw.textlength(plate[2:4], font=font) + 2 * spaces + auto_middle_point[0], auto_middle_point[1])
            draw.text(off2, plate[4], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)
        else:
            draw.text(auto_middle_point, plate[2:5], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)
        
        # Draw the plate (final letters)
        draw.text(auto_final_point, plate[5:], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)
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
        
    # Aeronautica plates
    elif ptype == 'aeronautica' or ptype == 'carabinieri'\
        or ptype == 'esercito' or ptype == 'marina':
        # Open base image
        if ptype == 'aeronautica':
            img = Image.open(aeronautica_empty_plate_path)
        elif ptype == 'carabinieri':
            img = Image.open(carabinieri_empty_plate_path)
        elif ptype == 'esercito':
            img = Image.open(esercito_empty_plate_path)
        elif ptype == 'marina':
            img = Image.open(marina_empty_plate_path)

        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Create a font object
        font = ImageFont.truetype(font_path, aeronautica_font_size)

        # Draw the plate (central letters)
        draw.text(aeronautica_middle_point, plate[:2], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)
        
        # Justify center text (central numbers)
        spaces = aeronautica_max_number_width - draw.textlength(plate[2:5], font=font)
        if spaces > 3:
            spaces = floor(spaces / 3)
            draw.text(aeronautica_final_point, plate[2], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)

            off1 = (draw.textlength(plate[2], font=font) + spaces + aeronautica_final_point[0], aeronautica_middle_point[1])
            draw.text(off1, plate[3], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)

            off2 = (draw.textlength(plate[2:4], font=font) + 2 * spaces + aeronautica_final_point[0], aeronautica_middle_point[1])
            draw.text(off2, plate[4], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)
        else:
            draw.text(aeronautica_final_point, plate[2:5], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)
        return img

    # Vigili del fuoco plates
    elif ptype == 'vigili_fuoco':
        # Open base image
        img = Image.open(vigili_fuoco_empty_plate_path)
        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Create a font object
        font = ImageFont.truetype(font_path, vigili_fuoco_font_size)
        
        # Justify center text (central numbers)
        spaces = vigili_fuoco_max_number_width - draw.textlength(plate, font=font)
        if spaces > 5:
            spaces = floor(spaces / 5)
            draw.text(vigili_fuoco_middle_point, plate[0], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)

            off1 = (draw.textlength(plate[0], font=font) + spaces + vigili_fuoco_middle_point[0], vigili_fuoco_middle_point[1])
            draw.text(off1, plate[1], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)

            off2 = (draw.textlength(plate[0:2], font=font) + 2 * spaces + vigili_fuoco_middle_point[0], vigili_fuoco_middle_point[1])
            draw.text(off2, plate[2], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)

            off3 = (draw.textlength(plate[0:3], font=font) + 3 * spaces + vigili_fuoco_middle_point[0], vigili_fuoco_middle_point[1])
            draw.text(off3, plate[3], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)

            off4 = (draw.textlength(plate[0:4], font=font) + 4 * spaces + vigili_fuoco_middle_point[0], vigili_fuoco_middle_point[1])
            draw.text(off4, plate[4], fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)
        else:
            draw.text(vigili_fuoco_middle_point, plate, fill=(0, 0, 0), font=font, stroke_width=auto_stroke_width)
        return img

    # Auto speciale plates
    elif ptype == 'auto_sp':
        # Open base image
        img = Image.open(auto_sp_empty_plate_path)
        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Create a font object
        font = ImageFont.truetype(font_path, moto_font_size)

        # Draw the plate (initial letters)
        draw.text(auto_sp_initial_point, plate[:2], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)
        
        # Justify center text (central numbers)
        spaces = auto_sp_max_number_width - draw.textlength(plate[2:5], font=font)
        if spaces > 3:
            spaces = floor(spaces / 3)
            draw.text(auto_sp_middle_point, plate[2], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)

            off1 = (draw.textlength(plate[2], font=font) + spaces + auto_sp_middle_point[0], auto_sp_middle_point[1])
            draw.text(off1, plate[3], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)

            off2 = (draw.textlength(plate[2:4], font=font) + 2 * spaces + auto_sp_middle_point[0], auto_sp_middle_point[1])
            draw.text(off2, plate[4], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)
        else:
            draw.text(auto_sp_middle_point, plate[2:5], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)

        # Draw the plate (final letters)
        draw.text(auto_sp_final_point, plate[5:], fill=(0, 0, 0), font=font, stroke_width=moto_stroke_width)
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
def create_plate(gray:bool=True, ptype:str='auto', aff_t:bool=False) -> None:
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

    # Apply affine transformations if necessary
    if aff_t:
        img = affine_transform(img)
        img = Image.fromarray(img)

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
def create_noisy_plate(ptype:str='auto', noise:np.ndarray=None, aff_t:bool=False) -> None:
    # Moto/auto_sp plates have different shape
    if ptype == 'moto' or ptype == 'auto_sp':
        xpix, ypix = moto_image_width, moto_image_height
    # Every other plate is basically a car plate
    else:
        xpix, ypix = auto_image_width, auto_image_height

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

    # Apply affine transformations if necessary
    if aff_t:
        img = affine_transform(img)

    # Add the noise to the image
    noisy = np.asarray(img) + noise_img
    noisy = Image.fromarray(noisy)
    noisy = noisy.convert('L')

    # Save the image
    noisy.save(output_path + plate + '{}.png'.format(get_suffix(ptype)))

    # Save the plate into a text file
    with open(output_path + 'generated{}.txt'.format(get_suffix(ptype)), 'a+') as f:
        f.write(plate + '\n')
    return

# Function to apply affine transformations to images
def affine_transform(im:Image.Image) -> cv2.Mat:
    img = np.asarray(im)
    rows, cols = img.shape[0], img.shape[1]

    p1 = [10, 10]
    p2 = [10, rows - 10]
    p3 = [cols - 10, 10]
    pts1 = np.float32([p1, p2, p3])
    
    delta = 4
    rand = np.random.randint(-delta, delta, 6)
    p1r = [p1[0] + rand[0], p1[1] + rand[1]]
    p2r = [p2[0] + rand[2], p2[1] + rand[3]]
    p3r = [p3[0] + rand[4], p3[1] + rand[5]]
    pts2 = np.float32([p1r, p2r, p3r])
    
    M = cv2.getAffineTransform(pts1, pts2)
    color = np.random.randint(0, 255)
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=(color, color, color))
    return dst

# Function to apply perspective transformations to images
def perspective_transform(im:Image.Image) -> cv2.Mat:
    img = np.asarray(im)
    rows, cols = img.shape[0], img.shape[1]

    p1 = [10, 10]
    p2 = [10, rows - 10]
    p3 = [cols - 10, 10]
    p4 = [cols - 10, rows - 10]
    pts1 = np.float32([p1, p2, p3, p4])
    
    delta = 4
    rand = np.random.randint(-delta, delta, 8)
    p1r = [p1[0] + rand[0], p1[1] + rand[1]]
    p2r = [p2[0] + rand[2], p2[1] + rand[3]]
    p3r = [p3[0] + rand[4], p3[1] + rand[5]]
    p4r = [p4[0] + rand[6], p4[1] + rand[7]]
    pts2 = np.float32([p1r, p2r, p3r, p4r])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

# Function to remove shadows from images
def remove_shadows(im:cv2.Mat) -> cv2.Mat:
    # Dilate and blur the image
    dilated_img = cv2.dilate(im, np.ones((11, 11), np.uint8))
    # cv2.imshow('dilated', dilated_img)
    bg_img = cv2.medianBlur(dilated_img, 3)
    # cv2.imshow('bg', bg_img)

    # Subtract the blurred image from the original image
    diff_img = 255 - cv2.absdiff(im, bg_img)
    # cv2.imshow('diff', diff_img)

    # Normalize the image
    norm_img = diff_img.copy()
    cv2.normalize(
        src = diff_img,
        dst = norm_img,
        alpha = 0,
        beta = 255,
        norm_type = cv2.NORM_MINMAX,
        dtype = cv2.CV_8UC1
    )
    # cv2.imshow('norm', norm_img)

    _, thr_img = cv2.threshold(norm_img, 220, 0, cv2.THRESH_TRUNC)
    cv2.normalize(
        src = thr_img, 
        dst = thr_img,
        alpha = 0,
        beta = 255,
        norm_type = cv2.NORM_MINMAX,
        dtype = cv2.CV_8UC1
    )
    # cv2.imshow('thr', thr_img)
    
    return thr_img

# Function to remove shadows from images
def remove_shadows2(im:cv2.Mat) -> cv2.Mat:
    blur = cv2.medianBlur(im, 3)
    # cv2.imshow('blur', blur)

    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return th


# Function to extract single characters from the plate 
def extract_characters(plate:Image.Image, rm_shdw:bool = False) -> list[cv2.Mat]:
    # Add a white border to the image
    plate = ImageOps.expand(plate, border=2, fill='white')

    # Convert the image in cv2 format
    img = np.asarray(plate)
    # cv2.imshow('img', img)

    # Remove shadows from the image
    if rm_shdw:
        img = remove_shadows(img)
        # cv2.imshow('rm_shdw', img)

    # print(np.mean(img))

    # If the image is too dark, brighten it
    if np.mean(img) < 120:
        # print(TEXT_YELLOW + 'brighten' + TEXT_RESET)
        img = cv2.convertScaleAbs(img, alpha=1.7)
        # cv2.imshow('bright', img)

    # If the image is too bright, darken it
    elif np.mean(img) > 160:
        # print(TEXT_YELLOW + 'darken' + TEXT_RESET)
        img = cv2.convertScaleAbs(img, alpha=0.7)
        # cv2.imshow('dark', img)

    # Apply thresholding to the image
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Apply morphological transformations to the image
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

    # cv2.imshow('Processed', img)

    # Find the contours of the image
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Sort the contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a list to store the characters
    characters = []
    positions = []

    # For each contour, extract the character
    r = img.copy()
    for cnt in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        # Show the bounding rectangle
        # cv2.rectangle(r, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.imshow('Contours', r)

        # If the area is too small or too large, ignore it
        if w * h < 100 or w * h > 900:
            continue
        
        # Extract the character from the image
        char = img[y:y+h, x:x+w]

        # Exclude characters with less than 15% or more than 60% of black pixels
        s = np.sum(char) / (w * h * 255)
        if s > 0.85 or s < 0.4:
            continue

        # Resize the character to a fixed size
        char = cv2.resize(char, (20, 40))

        # Add a black border to the character
        # char_cp = np.zeros((44, 24))
        # char_cp[2:42, 2:22] = char
        # char_cp[0:2, :] = 0
        # char_cp[42:44, :] = 0
        # char_cp[:, 0:2] = 0
        # char_cp[:, 22:24] = 0

        # Add the character to the list
        positions.append((char, x, y, w, h))

    # Sort the characters by x position
    positions = sorted(positions, key=lambda x: x[1]) 

    # Add the characters to the list
    for pos in positions:
        characters.append(pos[0])

    # Plot found characters
    # matplotlib.use('TkAgg')
    # for i, char in enumerate(characters):
    #     plt.subplot(1, len(characters), i + 1)
    #     plt.imshow(char, cmap='gray')
    #     plt.axis('off')
    # plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return characters


# Driver function
def main(nplates:int, gray:bool, perc_noise:int, ptype:str, aff_t:int, new_noise:int=1000) -> None:
    # Create the output directory if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create the files "generated-*.txt" if not exist
    for t in PLATE_TYPES:
        f = open(output_path + 'generated{}.txt'.format(get_suffix(t)), 'a+')
        f.close()

    # Switch on ptype parameter
    if ptype == 'mixed':
        # Generate equal number of plates for each type
        n = nplates // len(PLATE_TYPES)
    else:
        n = nplates

    for t in range(len(PLATE_TYPES) if n != nplates else 1):
        # Generate "n" random plates
        noisy = int(n * perc_noise / 100)
        p = PLATE_TYPES[t] if n != nplates else ptype
        affines = int(n * aff_t / 100)
        affines_noisy = int(affines * perc_noise / 100)
        affines_normal = affines - affines_noisy

        if n - noisy:
            print(TEXT_GREEN 
                + '>> Generating {} plates in {} (CLEAR) ({})'.format(n - noisy,
                'GRAY' if gray == True else 'COLOR', p)
                + TEXT_RESET)
            for _ in tqdm(range(n - noisy)):
                create_plate(gray=gray, ptype=p, aff_t=affines_normal>0)
                affines_normal -= 1
        
        if noisy:
            print(TEXT_GREEN 
                + '>> Generating {} plates in GRAY (NOISY) ({})'.format(noisy, p)
                + TEXT_RESET)
            for i in tqdm(range(noisy)):
                # Every "new_noise" iterations regenerate the noise image
                if i % new_noise == 0:
                    noise = generate_noise_image()
                create_noisy_plate(ptype=p, noise=noise, aff_t=affines_noisy>0)
                affines_noisy -= 1

    return

# Main function
def driver_main():
    choice = 1

    while choice != '0':
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Generate mixed plates.')
        print('  2. Generate auto plates only.')
        print('  3. Generate moto plates only.')
        print('  4. Generate aeronautica plates only.')
        print('  5. Generate carabinieri plates only.')
        print('  6. Generate esercito plates only.')
        print('  7. Generate marina plates only.')
        print('  8. Generate vigili del fuoco plates only.')
        print('  9. Generate auto special plates only.')
        print(' 10. Extract characters from generated images.')
        print('  0. Exit.')
        choice = input(TEXT_YELLOW + 'Enter your choice: ' + TEXT_RESET)

        # Generate mixed plates
        if choice == '1':
            ptype = 'mixed'

        # Generate auto plates only
        elif choice == '2':
            ptype = 'auto'

        # Generate moto plates only
        elif choice == '3':
            ptype = 'moto'

        # Generate aeronautica plates only
        elif choice == '4':
            ptype = 'aeronautica'

        # Generate carabinieri plates only
        elif choice == '5':
            ptype = 'carabinieri'

        # Generate esercito plates only
        elif choice == '6':
            ptype = 'esercito'

        # Generate marina plates only
        elif choice == '7':
            ptype = 'marina'

        # Generate vigili del fuoco plates only
        elif choice == '8':
            ptype = 'vigili_fuoco'

        # Generate auto special plates only
        elif choice == '9':
            ptype = 'auto_sp'

        # Extract characters from generated images
        elif choice == '10':
            if not os.path.exists(chars_path):
                os.makedirs(chars_path)

            print('Saving cropped characters to {}'.format(chars_path))
            n = 0
            for el in os.listdir(output_path):
                if el.endswith('.png'):
                    img = Image.open(output_path + el)
                    chars = extract_characters(img)

                    # Save characters to file
                    for i, char in enumerate(chars):
                        cv2.imwrite(chars_path + '{}-{}.png'.format(el[i], n), char)
                        n += 1

            continue

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

        # Exit
        if choice == '0':
            print(TEXT_YELLOW + '>> Exiting.' + TEXT_RESET)
            break

        nplates = input('Enter the number of plates to generate [Enter = \"1000\"]: ')
        if nplates == '':
            nplates = 1000
        nplates = int(nplates)

        aff_t = input('Enter the percentage of plates with affine transformations [Enter = \"75\"]: ')
        if aff_t == '':
            aff_t = 75
        aff_t = int(aff_t)

        # Generate mixed normal/noisy images
        if choice == '1':
            perc = input('Enter the percentage of noisy images to generate [Enter = \"50%\"]: ')
            if perc == '':
                perc = 50
            perc = int(perc)

            new_noise = input('Regenerate the noise image every [Enter = \"1000\"] plates: ')
            if new_noise == '':
                new_noise = 1000
            new_noise = int(new_noise)            

            main(nplates=nplates, gray=True, perc_noise=perc, ptype=ptype, new_noise=new_noise, aff_t=aff_t)

        # Generate normal images only
        elif choice == '2':
            main(nplates=nplates, gray=True, perc_noise=0, ptype=ptype, aff_t=aff_t)

        # Generate noisy images only
        elif choice == '3':
            new_noise = input('Regenerate the noise image every [Enter = \"1000\"] plates: ')
            if new_noise == '':
                new_noise = 1000
            new_noise = int(new_noise)

            main(nplates=nplates, gray=True, perc_noise=100, ptype=ptype, new_noise=new_noise, aff_t=aff_t)

        # Generate coloured images
        elif choice == '4':
            main(nplates=nplates, gray=False, perc_noise=0, ptype=ptype, aff_t=aff_t)

        # Invalid input
        else:
            print(TEXT_YELLOW + '>> Invalid choice.' + TEXT_RESET)
            continue

    return

if __name__ == '__main__':
    driver_main()
