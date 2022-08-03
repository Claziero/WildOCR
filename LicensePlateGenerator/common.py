import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

# Define colors
TEXT_RESET = '\033[0m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'


# Function to remove shadows from images
def remove_shadows(im:cv2.Mat, show:bool=False, save:bool=False) -> cv2.Mat:
    # Dilate the image
    dilated_img = cv2.dilate(im, np.ones((11, 11), np.uint8))
    if show: cv2.imshow('dilated', dilated_img)
    if save: cv2.imwrite('dilated2.png', dilated_img)

    # Blur the image
    bg_img = cv2.medianBlur(dilated_img, 3)
    if show: cv2.imshow('bg', bg_img)
    if save: cv2.imwrite('bg.png', bg_img)

    # Subtract the blurred image from the original image
    diff_img = 255 - cv2.absdiff(im, bg_img)
    if show: cv2.imshow('diff', diff_img)
    if save: cv2.imwrite('diff.png', diff_img)

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
    if show: cv2.imshow('norm', norm_img)
    if save: cv2.imwrite('norm.png', norm_img)

    # Apply thresholding to the image
    _, thr_img = cv2.threshold(norm_img, 220, 0, cv2.THRESH_TRUNC)
    if show: cv2.imshow('thr_img', thr_img)
    if save: cv2.imwrite('thr_img.png', thr_img)

    # Normalize the image again
    cv2.normalize(
        src = thr_img, 
        dst = thr_img,
        alpha = 0,
        beta = 255,
        norm_type = cv2.NORM_MINMAX,
        dtype = cv2.CV_8UC1
    )
    if show: cv2.imshow('thr2', thr_img)
    if save: cv2.imwrite('thr2.png', thr_img)
    
    return thr_img

# Function to apply transformations to images before character extraction
def apply_trfs(plate:Image.Image, rm_shdw:bool = False, show:bool=False, save:bool=False) -> cv2.Mat:
    # Convert the image in cv2 format
    img = np.asarray(plate)
    if show: cv2.imshow('img', img)
    if save: cv2.imwrite('gray.png', img)

    # Remove shadows from the image
    if rm_shdw:
        img = remove_shadows(img, show, save)
        if show: cv2.imshow('rm_shdw', img)
    
    # If the image is too dark, brighten it
    if np.mean(img) < 120:
        img = cv2.convertScaleAbs(img, alpha=1.7)
        if show: 
            cv2.imshow('bright', img)
            print(TEXT_YELLOW + 'brighten' + TEXT_RESET)

    # If the image is too bright, darken it
    elif np.mean(img) > 160:
        img = cv2.convertScaleAbs(img, alpha=0.7)
        if show: 
            cv2.imshow('dark', img)
            print(TEXT_YELLOW + 'darken' + TEXT_RESET)

    # Apply thresholding to the image
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if show: cv2.imshow('thr', img)
    if save: cv2.imwrite('threshold.png', img)

    # Apply morphological transformations to the image
    
    # Alternative for erode + dilate
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # if show: cv2.imshow('open', img)
    # if save: cv2.imwrite('open.png', img)
    
    # Erode the image
    img = 255 - img
    img = cv2.erode(img, np.ones((2, 2), np.uint8))
    img = 255 - img
    if show: cv2.imshow('erode', img)
    if save: cv2.imwrite('erode.png', img)

    # Dilate the image
    img = 255 - img
    img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    img = 255 - img
    if show: cv2.imshow('dilate', img)
    if save: cv2.imwrite('dilate.png', img)
    
    # Apply Closing to the image
    img = 255 - img
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    img = 255 - img
    if show: cv2.imshow('close', img)
    if save: cv2.imwrite('close.png', img)
    
    if show: cv2.waitKey(0)
    return img

# Function to extract single characters from the plate 
def extract_characters(plate:Image.Image, rm_shdw:bool=False, show:bool=False, save:bool=False) -> list[cv2.Mat]:
    # Add a white border to the image
    plate = ImageOps.expand(plate, border=2, fill='white')

    # Apply transformations to the image
    img = apply_trfs(plate, rm_shdw, show, save)

    # Find the contours of the image
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Sort the contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a list to store the characters
    characters = []
    positions = []

    # For each contour, extract the character
    for cnt in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

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

        # Add the character to the list
        if show:
            # Add a black border to the character
            char_cp = np.zeros((44, 24))
            char_cp[2:42, 2:22] = char
            char_cp[0:2, :] = 0
            char_cp[42:44, :] = 0
            char_cp[:, 0:2] = 0
            char_cp[:, 22:24] = 0
            positions.append((char_cp, x, y, w, h))
        else:
            positions.append((char, x, y, w, h))

    # Sort the characters by x position
    positions = sorted(positions, key=lambda x: x[1]) 

    # Add the characters to the list
    for pos in positions:
        characters.append(pos[0])

    # Plot found characters
    if show:
        matplotlib.use('TkAgg')
        for i, char in enumerate(characters):
            plt.subplot(1, len(characters), i + 1)
            plt.imshow(char, cmap='gray')
            plt.axis('off')
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return characters
