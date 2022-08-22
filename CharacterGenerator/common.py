import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
def apply_trfs(img:cv2.Mat, rm_shdw:bool = False, warp:bool=True, show:bool=False, save:bool=False) -> tuple[cv2.Mat, bool]:
    if show: cv2.imshow('img', img)
    if save: cv2.imwrite('gray.png', img)

    # Remove shadows from the image
    if rm_shdw:
        img = remove_shadows(img, show, save)
        if show: cv2.imshow('rm_shdw', img)

    # Rectify the image
    if warp:
        img, warped = rectify_plate(img, show, save)
        if show: cv2.imshow('rect', img)
        if save: cv2.imwrite('rect.png', img)
    else:
        warped = False
    
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
    return img, warped

# Function to order points for perspective transformation
def order_points(pts):
    # Find centre of object
    center = np.mean(pts)

    # Move coordinate system to centre of object
    shifted = pts - center

    # Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]

# Function to rectify the plate image using homographic transformations
def rectify_plate(plate:cv2.Mat, show:bool=False, save:bool=False) -> tuple[cv2.Mat, bool]:
    warped = False
    
    # Blur the image
    img = cv2.GaussianBlur(plate, (5, 5), 1)
    if show: cv2.imshow('blur', img)
    if save: cv2.imwrite('blur.png', img)

    # Apply thresholding to the image
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if show: cv2.imshow('thr', img)
    if save: cv2.imwrite('threshold.png', img)

    # Find contours
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Sort the contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the largest contour
    max_area = 0
    points = []
    index = None
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # print(area)
        # The contour area has to be at least 1000 pixel
        if area > 1000:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            if area > max_area and len(approx) == 4:
                max_area = area
                points = approx
                index = i

    # Get the largest contour
    if index is not None:
        cnt = contours[index]
        
        # Draw the contour
        copy = plate.copy()
        cv2.drawContours(copy, [cnt], -1, (0, 255, 0), 4)
        if show: cv2.imshow('contour', copy)
        if save: cv2.imwrite('contour.png', copy)

        # Get the points of the contour
        src = np.squeeze(points).astype(np.float32)
        # Reorder the points
        src = order_points(src)

        # Get the points of the destination image
        dst = np.array([
            [0, 0], 
            [plate.shape[1], 0], 
            [plate.shape[1], plate.shape[0]], 
            [0, plate.shape[0]]
            ], dtype=np.float32)
        # Reorder the points
        dst = order_points(dst)

        # Get the transformation matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # Apply the transformation matrix
        plate = cv2.warpPerspective(plate, M, (plate.shape[1], plate.shape[0]))

        # Add a white border to the image
        plate = cv2.copyMakeBorder(plate, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        warped = True

        # Show the result
        if show: cv2.imshow('rect', plate)
        if save: cv2.imwrite('rect.png', plate)

    cv2.waitKey(0)
    return plate, warped

# Function to extract single characters from the plate 
def extract_characters_plate(plate:cv2.Mat, rm_shdw:bool=False, show:bool=False, save:bool=False) -> list[cv2.Mat]:
    # Apply transformations to the image
    img, warped = apply_trfs(plate, rm_shdw, True, show, save)

    # Add a white border to the image
    img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Find the contours of the image
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Sort the contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a list to store the characters
    characters = []
    positions = []
    positions_cp = []

    # For each contour, extract the character
    for cnt in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

        # If the area is too small or too large, ignore it
        if warped and (w * h < 300 or w * h > 2500):
            continue
        elif not warped and (w * h < 100 or w * h > 900):
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
            positions_cp.append((char_cp, x, y, w, h))

        positions.append((char, x, y, w, h))

    # Sort the characters by x position
    positions = sorted(positions, key=lambda x: x[1]) 
    if show:
        char_cp = []
        positions_cp = sorted(positions_cp, key=lambda x: x[1])
        for pos in positions_cp:
            char_cp.append(pos[0])

    # Add the characters to the list
    for pos in positions:
        characters.append(pos[0])

    # Plot found characters
    if show:
        matplotlib.use('TkAgg')
        for i, char in enumerate(char_cp):
            plt.subplot(1, len(char_cp), i + 1)
            plt.imshow(char, cmap='gray')
            plt.axis('off')
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return characters

# Function to extract single characters from a text
def extract_characters_text(img:cv2.Mat, rm_shdw:bool=False, show:bool=False, save:bool=False) -> list[list[cv2.Mat]]:
    # Apply transformations to the image
    img = apply_trfs(img, rm_shdw, False, show, save)[0]

    # If the image has black text on white background, invert the colors
    if np.sum(img) > (img.shape[0] * img.shape[1] * 255) / 2:
        img = 255 - img
    
    # --------------------------------LINE DETECTION--------------------------------
    # Find lines of text in the image
    # Dilate the image
    kernel = np.ones((1, 20), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=1)
    if show: cv2.imshow('dilate', dilate)
    if save: cv2.imwrite('dilate.png', dilate)

    # Find the contours of the image
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Create a list to store the lines
    lines = []
    im2 = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # The aspect ratio must not be vertical
        if w / h < 0.5:
            continue

        # Adapt the area limits using the area of the image
        if w * h < img.shape[0] * img.shape[1] / 10:
            continue

        # Extract the line from the image
        a = y - 5 if y > 5 else 0
        b = y + h + 5 if y + h < im2.shape[0] - 5 else im2.shape[0] - 1
        c = x - 5 if x > 5 else 0
        d = x + w + 5 if x + w < im2.shape[1] - 5 else im2.shape[1] - 1
        
        line = img[a:b, c:d]
        lines.append((line, x, y, w, h))
    
    # Show the contours found
    if show: cv2.imshow('contours', im2)
    if save: cv2.imwrite('contours.png', im2)

    # Order found contours by y and then x position
    lines = sorted(lines, key=lambda x: (x[2], x[1]))
    if show:
        for i, line in enumerate(lines):
            cv2.imshow('line' + str(i), line[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if save: 
        for i, line in enumerate(lines):
            cv2.imwrite('line' + str(i) + '.png', line[0])
    
    
    # --------------------------------WORD DETECTION--------------------------------
    result_words = []

    # For each line, find words inside them
    for line in lines:
        # Dilate the image
        kernel = np.ones((1, 9), np.uint8)
        dilate = cv2.dilate(line[0], kernel, iterations=1)
        if show: cv2.imshow('dilate', dilate)
        if save: cv2.imwrite('dilate.png', dilate)

        # Find the contours of the image
        contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # Create a list to store the words
        words = []
        im2 = line[0].copy()

        # For each contour, extract the word
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # The aspect ratio must not be vertical
            if w / h < 0.5:
                continue

            # Adapt the area limits using the area of the image
            if w * h < line[3] * line[4] / 10:
                continue

            # Extract the word from the image
            a = y - 5 if y > 5 else 0
            b = y + h + 5 if y + h < im2.shape[0] - 5 else im2.shape[0] - 1
            c = x - 5 if x > 5 else 0
            d = x + w + 5 if x + w < im2.shape[1] - 5 else im2.shape[1] - 1

            word = line[0][a:b, c:d]
            words.append((word, x, y, w, h))
        
        # Show the contours found
        if show: cv2.imshow('contours', im2)
        if save: cv2.imwrite('contours.png', im2)

        # Order found contours by x position
        words = sorted(words, key=lambda x: x[1])
        if show:
            for i, word in enumerate(words):
                cv2.imshow('word' + str(i), word[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if save: 
            for i, word in enumerate(words):
                cv2.imwrite('word' + str(i) + '.png', word[0])

        result_words.extend(words)
    

    # --------------------------------CHARACTER DETECTION--------------------------------
    # Create a list to store the characters
    characters = []
    positions = []
    positions_cp = []
    result_chars = []

    # For each word, extract the characters
    for word in result_words:
        # Find the contours of the image
        contours = cv2.findContours(word[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # Sort the contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Create a list to store the characters
        characters = []
        positions = []
        positions_cp = []

        # For each contour, extract the character
        for cnt in contours:
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            x -= 1 if x > 0 else 0
            y -= 1 if y > 0 else 0
            w += 2
            h += 2

            # If the area is too small or too large, ignore it
            # Adapt the area limits using the area of the image
            # if w * h > word[0].shape[0] * word[0].shape[1] / 2:
            #     continue
            if w * h < word[0].shape[0] * word[0].shape[1] / 25:
                continue
            
            # Extract the character from the image
            char = word[0][y:y+h, x:x+w]

            # If the character is white on black background, invert it
            if np.sum(char) > (w * h * 255) / 2.5:
                char = 255 - char

            # If the character is like "I" or "i", it has a special w/h ratio
            if w / h < 0.3:
                # Resize the image to fit into 28 pixel of height
                char = cv2.resize(char, (int(28 * w / h), 28))

                # Add white padding at left and right to reach 28 pixels of width
                char = cv2.copyMakeBorder(char, 0, 0, int((28 - char.shape[1]) / 2), int((28 - char.shape[1]) / 2), cv2.BORDER_CONSTANT, value=(255, 255, 255))

            # Resize the character to a fixed size
            char = cv2.resize(char, (28, 28))

            # char = cv2.resize(char, (24, 24))
            # Add a white border to the character
            # char_cp = np.zeros((28, 28))
            # char_cp[2:26, 2:26] = char
            # char_cp[0:2, :] = 255
            # char_cp[26:28, :] = 255
            # char_cp[:, 0:2] = 255
            # char_cp[:, 26:28] = 255
            # char = char_cp

            # Add the character to the list
            if show:
                # Add a black border to the character
                char_cp = np.zeros((32, 32))
                char_cp[2:30, 2:30] = char
                char_cp[0:2, :] = 0
                char_cp[30:32, :] = 0
                char_cp[:, 0:2] = 0
                char_cp[:, 30:32] = 0
                positions_cp.append((char_cp, x, y, w, h))

            positions.append((char, x, y, w, h))

        # Sort the characters by x position
        positions = sorted(positions, key=lambda x: x[1])
        if show:
            char_cp = []
            positions_cp = sorted(positions_cp, key=lambda x: x[1])
            for pos in positions_cp:
                char_cp.append(pos[0])

        # Add the characters to the list
        for pos in positions:
            characters.append(pos[0])

        # Plot found characters
        if show:
            matplotlib.use('TkAgg')
            for i, char in enumerate(char_cp):
                plt.subplot(1, len(char_cp), i + 1)
                plt.imshow(char, cmap='gray')
                plt.axis('off')
            plt.show()

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Append this line of characters to the result list
        result_chars.append((characters))

    return result_chars
