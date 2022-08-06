import os
import cv2
import sys
import numpy as np

sys.path.insert(0, './OCR')
sys.path.insert(0, './LicensePlateGenerator')

from PIL import Image
from OCR.driver import Driver
from Detector.detect import Detector
from LicensePlateGenerator.common import extract_characters

# Define colors
TEXT_RESET = '\033[0m'
TEXT_RED = '\033[91m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'
TEXT_BLUE = '\033[94m'

# Dimension constants
# Car plates are 200x44 pixels
auto_image_width, auto_image_height = 200, 44
auto_min_ar, auto_max_ar = 1.5, 6 # Correct AR = 4.54

# Moto plates are 106x83 pixels
moto_image_width, moto_image_height = 106, 83
moto_min_ar, moto_max_ar = 1, 1.5 # Correct AR = 1.27

# Define paths
data_path = 'data/'
input_path = data_path + 'input/'
output_path = data_path + 'output/'
video_path = data_path + 'video/'

# Function to convert an image to the correct format for CNN
def process_image(img:cv2.Mat, log:bool=True) -> np.ndarray:
    # Check the image dimensions
    img_ar = img.shape[1] / img.shape[0]
    # print('Image dimensions: ' + str(img.shape[1]) + 'x' + str(img.shape[0]))
    # print('Image aspect ratio: ' + str(img_ar))

    # If it's a car plate, resize it to the correct dimensions
    if img_ar >= auto_min_ar and img_ar <= auto_max_ar:
        # Resize the image to the correct dimensions
        img = cv2.resize(img, (auto_image_width, auto_image_height))

    # If it's a motorcycle plate, resize it to the correct dimensions
    elif img_ar >= moto_min_ar and img_ar <= moto_max_ar:
        # Resize the image to the correct dimensions
        img = cv2.resize(img, (moto_image_width, moto_image_height))

    # If there's an error
    else:
        if log: print(TEXT_RED + '>> Image dimensions are not correct.' + TEXT_RESET)
        return None

    # Convert the image to a numpy array
    img = np.array(img, dtype=np.uint8)

    return img

# Function to write the OCR string to the image
def write_ocr(img:cv2.Mat, coords:list[int], ocr_string:str) -> cv2.Mat:
    write_point = (coords[1] - 0, coords[0] - 10)

    result = cv2.putText(img,
        text = ocr_string,
        org = write_point,
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = (0, 255, 0),
        thickness = 2,
        lineType = cv2.LINE_AA)

    result = cv2.rectangle(img,
        pt1 = (coords[1], coords[0]),
        pt2 = (coords[3], coords[2]),
        color = (0, 255, 0),
        thickness = 3)
    
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)
    return result

# Function to read a plate detection image
def plate_detect(cnn_driver:Driver, img:cv2.Mat, coords:list[int], save:str, log:bool=True) -> cv2.Mat:
    # Crop the plate
    crop = img[coords[0]:coords[2], coords[1]:coords[3]]
    if save: cv2.imwrite('plate.png', crop)

    # Convert the image to the correct format for CNN
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = np.array(crop, dtype=np.uint8)
    crop = Image.fromarray(crop)
    
    # Extract single characters from the image
    chars = extract_characters(crop, show=True, save=True)

    # If there are less than 7 characters, retry the scanning using remove_shadows function
    if len(chars) < 7:
        # Extract single characters from the image
        chars = extract_characters(crop, True, show=True, save=True)

        # If there are less than 7 characters recognized, the plate is not valid
        if len(chars) < 7:
            if log:
                print(TEXT_RED + '>> Recognised only {} characters out of 7.'.format(len(chars)) + TEXT_RESET)
            return img

    # Predict all characters
    ocr = ''
    confidence = []
    for char in chars:
        ch, cd = cnn_driver.forward(char)
        ocr += ch
        confidence.append(cd)
    if log:
        print(TEXT_BLUE + '>> Recognised plate number w/ processing: ' + ocr + TEXT_RESET)

    # If the plate is predicted
    if ocr:
        # If there are more than 7 characters recognized, check the first 2 characters
        # If the first 2 characters are numbers, remove them
        index = 0
        for _ in range(2):
            if len(ocr) > 7 and ocr[index].isdecimal():
                ocr = ocr[1:]
                confidence = np.delete(confidence, index)
            else:
                index = 1

        # Check if there are at least 3 numbers in sequence
        index = 2
        for _ in range(3):
            if len(ocr) > 7:
                if ocr[index].isdecimal():
                    # No letters found
                    if ocr[index + 1].isdecimal() and ocr[index + 2].isdecimal():
                        break
                    # Remove letters in position index + 1
                    elif not ocr[index + 1].isdecimal() and ocr[index + 2].isdecimal():
                        ocr = ocr[:index + 1] + ocr[index + 2:]
                        confidence = np.delete(confidence, index + 1)
                        continue
                    # Remove letters in position index + 2
                    elif ocr[index + 1].isdecimal() and not ocr[index + 2].isdecimal():
                        ocr = ocr[:index + 2] + ocr[index + 3:]
                        confidence = np.delete(confidence, index + 2)
                        continue
                # Remove letter in position index
                elif not ocr[index].isdecimal() and ocr[index + 1].isdecimal() and ocr[index + 2].isdecimal():
                    ocr = ocr[:index] + ocr[index + 1:]
                    confidence = np.delete(confidence, index)
                    continue                                

        # If there are more than 7 characters recognized, remove those with the lowest confidence
        while len(ocr) > 7:
            min = np.argmin(confidence)
            confidence = np.delete(confidence, min)
            ocr = ocr[:min] + ocr[min+1:]

        # Print the text
        if log:
            print(TEXT_BLUE + '>> Recognised plate number: ' + ocr + TEXT_RESET)
        res = write_ocr(img, coords, ocr)

        # Save the image
        if save is not False:
            cv2.imwrite(save, res)
        return res

    elif log:
        print(TEXT_RED + '>> Plate not recognised.' + TEXT_RESET)
    
    return img

# Function to scan an image
def scan_image(cnn_driver:Driver, pd:Detector, img:cv2.Mat, save:str, log:bool=True) -> cv2.Mat:
    # Get all detections
    img_array = np.asarray(img)
    detections = pd.detect_and_crop(img_array)

    # If there's no detections, return the original image
    num_detections = detections['num_detections']
    if num_detections == 0:
        if log: print(TEXT_RED + '>> No detections found.' + TEXT_RESET)
        return img

    # Retrive data from detections
    detection_boxes = detections['detection_boxes']
    detection_scores = detections['detection_scores']
    detection_classes = detections['detection_classes']
    
    # Loop through all detections
    for i in range(num_detections):
        # Get the bounding box
        coords = detection_boxes[i]
        # Get the confidence score
        score = detection_scores[i]
        # Get the class
        class_id = detection_classes[i]

        # If the class_id is 1 (plate), extract the plate text
        if class_id == 1:
            # Get the coordinates of the plate
            coords[0] = coords[0] * img.shape[0] - 5
            coords[1] = coords[1] * img.shape[1] - 5
            coords[2] = coords[2] * img.shape[0] + 5
            coords[3] = coords[3] * img.shape[1] + 5
            coords = coords.astype(int)

            # Process the plate
            img = plate_detect(cnn_driver, img, coords, save, log)

        # Else it's a text area
        else:
            # Scan the text
            pass


        
    return img

# Function to scan a video file
def scan_video(cnn_driver:Driver, pd:Detector, video_file:str, save:str) -> None:
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # If the video file is opened
    if cap.isOpened():
        # Get the video frame number
        frame_number_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save, fourcc, 24.0, (width, height))

        # Initialize the frame number
        frame_number = 0

        # While the video is being read
        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()
            # cv2.imshow('Cam', frame)

            # Force stop condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # If the frame is read
            if ret:
                # Scan the frame as an image
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                res = scan_image(cnn_driver, pd, frame, False, False)

                # Write the image to the video file
                res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                out.write(res)

                # Increment the frame number
                frame_number += 1
                if frame_number % 100 == 0:
                    print(TEXT_GREEN 
                        + 'Frame number: {}/{}'.format(frame_number, frame_number_tot) 
                        + TEXT_RESET)

            # If the video is finished
            else:
                # Break the loop
                break

        # Release the video file
        cap.release()
        out.release()
    else:
        print(TEXT_RED + '>> Error opening video stream or file.' + TEXT_RESET)

    return

# Driver function
def driver() -> None:
    # Create the input and output directories if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # Create a NN for OCR functionality
    cnn_driver = Driver()

    # Create a NN for plate detection functionality
    detector = Detector('Detector/')
    
    choice = 1
    nn_loaded = False
    while choice != '0':
        # Get the user input
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Load pretrained models of OCR NN and Detector NN.')
        print('  2. Scan an image.')
        print('  3. Scan a directory.')
        print('  4. Scan a video.')
        print('  0. Exit.')
        choice = input(TEXT_YELLOW + 'Enter your choice: ' + TEXT_RESET)   

        # Exit
        if choice == '0':
            print(TEXT_YELLOW + '>> Exiting...' + TEXT_RESET)
            break

        # Load pretrained models of OCR NN and Detector NN
        if choice == '1':
            # Load the OCR NN
            load = input('Enter the path to the pretrained model for OCR NN [Enter = \"OCR/model.pkl\"]: ')
            if load == '':
                load = 'OCR/model.pkl'
            cnn_driver.load_model(load)

            # Load the Detector NN
            detector.load_from_checkpoint()
            nn_loaded = True
            continue

        # Scan an image
        elif choice == '2':
            if not nn_loaded:
                print(TEXT_RED + '>> NNs not loaded.' + TEXT_RESET)

                # Load the OCR NN
                load = input('Enter the path to the pretrained model for OCR NN [Enter = \"OCR/model.pkl\"]: ')
                if load == '':
                    load = 'OCR/model.pkl'
                cnn_driver.load_model(load)

                # Load the Detector NN
                detector.load_from_checkpoint()
                nn_loaded = True

            # Get the image path
            print('Taking input images from \"' + input_path + '\" folder.')
            img_path = input('Enter the name of the image [Enter = \"image.jpg\"]: ')
            if img_path == '':
                img_path = 'image.jpg'

            print('Saving output images to \"' + output_path + '\" folder.')
            save = input('Enter the name of the image to save [Enter = \"{}\" | \"n\" = None]: '.format(img_path))
            if save == '':
                save = img_path
            elif save == 'n':
                save = False

            # Load the image
            img = cv2.imread(input_path + img_path)

            if save != False: save_name = os.path.join(output_path, save)
            else: save_name = False
            scan_image(cnn_driver, detector, img, save_name)
            continue

        # Scan a directory
        elif choice == '3':
            if not nn_loaded:
                print(TEXT_RED + '>> NNs not loaded.' + TEXT_RESET)

                # Load the OCR NN
                load = input('Enter the path to the pretrained model for OCR NN [Enter = \"OCR/model.pkl\"]: ')
                if load == '':
                    load = 'OCR/model.pkl'
                cnn_driver.load_model(load)

                # Load the Detector NN
                detector.load_from_checkpoint()
                nn_loaded = True
                
            # Get the directory path
            print('Taking input images from \"' + input_path + '\" folder.')
            dir_path = input('Enter the name of the directory [Enter = \".\"]: ')
            if dir_path == '':
                dir_path = '.'

            print('Saving output images to \"' + output_path + '\" folder.')
            save = input('Enter the name of the directory to save images in [Enter = \"{}\" | \"n\" = None]: '.format(dir_path))
            if save == '':
                save = dir_path
            elif save == 'n':
                save = False

            # Scan the directory
            for im in os.listdir(os.path.join(input_path, dir_path)):
                img_name = os.path.join(input_path, dir_path, im)
                print('Scanning image \"' + img_name + '\" ...')

                # Load the image
                img = cv2.imread(img_name)

                if save != False: save_name = os.path.join(output_path, save, im)
                else: save_name = False
                scan_image(cnn_driver, detector, img, save_name)

            continue

        # Scan a video
        elif choice == '4':
            if not nn_loaded:
                print(TEXT_RED + '>> NNs not loaded.' + TEXT_RESET)

                # Load the OCR NN
                load = input('Enter the path to the pretrained model for OCR NN [Enter = \"OCR/model.pkl\"]: ')
                if load == '':
                    load = 'OCR/model.pkl'
                cnn_driver.load_model(load)

                # Load the Detector NN
                detector.load_from_checkpoint()
                nn_loaded = True

            # Get the video path
            print('Taking input videos from \"' + video_path + '\" folder.')
            video = input('Enter the name of the video [Enter = \"video.mp4\"]: ')
            if video == '':
                video = 'video.mp4'
            video = os.path.join(video_path, video)
            

            print('Saving output video to \"' + video_path + '\" folder.')
            save = input('Enter the name of the video to save [Enter = \"{}\" | \"n\" = None]: '.format(video[:-4] + '_output.mp4'))
            if save == '':
                save = video[:-4] + '_output.mp4'
            elif save == 'n':
                save = False

            # Scan the video
            print('Scanning video \"' + video + '\" ...')
            scan_video(cnn_driver, detector, video, save)
            continue
            
        # If there's an error
        else:
            print(TEXT_YELLOW + '>> Invalid choice.' + TEXT_RESET)
            continue

    return

if __name__ == '__main__':
    driver()
