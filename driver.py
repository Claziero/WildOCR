import os
import cv2
import sys
import numpy as np

sys.path.insert(0, './OCR')
sys.path.insert(0, './CharacterGenerator')

from OCR.driver import Driver
from Detector.detect import Detector
from CharacterGenerator.common import extract_characters_plate, extract_characters_text

# Define colors
TEXT_RESET = '\033[0m'
TEXT_RED = '\033[91m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'
TEXT_BLUE = '\033[94m'

# Define paths
data_path = 'data/'
input_path = data_path + 'input/'
output_path = data_path + 'output/'
video_path = data_path + 'video/'

# Function to write the OCR string to the image
def write_ocr(img:cv2.Mat, coords:list[int], ocr_string:str, area_type:str='plate') -> cv2.Mat:
    write_point = (coords[1], coords[0] - 30)

    if area_type == 'plate':
        font_color = (0, 255, 255)
    elif area_type == 'text':
        font_color = (0, 255, 0)

    # Draw the bounding box
    result = cv2.rectangle(img,
        pt1 = (coords[1], coords[0]),
        pt2 = (coords[3], coords[2]),
        color = font_color,
        thickness = 3)
    
    # Draw a rectangle to write on
    text_size = cv2.getTextSize(ocr_string, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_width, text_height = text_size
    result = cv2.rectangle(result,
        pt1 = write_point,
        pt2 = (write_point[0] + text_width, write_point[1] + text_height + 10),
        color = font_color,
        thickness = cv2.FILLED)

    # Write the text
    result = cv2.putText(result,
        text = ocr_string,
        org = (write_point[0], write_point[1] + 25),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = (0, 0, 0),
        thickness = 2,
        lineType = cv2.LINE_AA)

    return result

# Function to read a plate detection image
def plate_detect(plate_cnn_driver:Driver, img:cv2.Mat, coords:list[int], save:str, log:bool=True) -> tuple[str, str]:
    # Crop the plate
    crop = img[coords[0]:coords[2], coords[1]:coords[3]]
    if save: cv2.imwrite('plate.png', crop)

    # Convert the image to grayscale
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = np.array(crop, dtype=np.uint8)
    
    # Extract single characters from the image
    chars = extract_characters_plate(crop, show=True, save=True)

    # If there are less than 7 characters, retry the scanning using remove_shadows function
    if len(chars) < 7:
        # Extract single characters from the image
        chars = extract_characters_plate(crop, True, show=True, save=True)

        # If there are less than 7 characters recognized, the plate is not valid
        if len(chars) < 7:
            if log:
                print(TEXT_RED + '>> Recognised only {} characters out of 7.'.format(len(chars)) + TEXT_RESET)
            return None, None

    # Predict all characters
    ocr = ''
    confidence = []
    for char in chars:
        ch, cd = plate_cnn_driver.forward(char)
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

    elif log:
        print(TEXT_RED + '>> Plate not recognised.' + TEXT_RESET)
        return None, None
    
    return ocr, 'plate'

# Function to read a text detection image
def text_detect(text_cnn_driver:Driver, img:cv2.Mat, coords:list[int], save:str, log:bool=True) -> tuple[str, str]:
    # Crop the text
    crop = img[coords[0]:coords[2], coords[1]:coords[3]]
    if save: cv2.imwrite('text.png', crop)

    # Convert the image to the correct format for CNN
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = np.array(crop, dtype=np.uint8)
    
    # Extract single characters from the image
    line_chars = extract_characters_text(crop, show=True, save=True)

    # For each line, predict the characters
    ocr = ''
    for line in line_chars:
        for char in line:
            ch = text_cnn_driver.forward(char)
            ocr += ch
        ocr += ' '

    # Print the text
    if log:
        print(TEXT_BLUE + '>> Recognised text: ' + ocr + TEXT_RESET)
    
    return ocr, 'text'

# Function to scan an image
def scan_image(plate_cnn_driver:Driver, text_cnn_driver:Driver, pd:Detector, img:cv2.Mat, save:str, log:bool=True) -> cv2.Mat:
    # Get all detections
    img_array = np.asarray(img)
    detections = pd.detect(img_array)

    # If there's no detections, return the original image
    num_detections = detections['num_detections']
    if num_detections == 0:
        if log: print(TEXT_RED + '>> No detections found.' + TEXT_RESET)
        return img

    # Retrive data from detections
    detection_boxes = detections['detection_boxes']
    detection_classes = detections['detection_classes']
    
    # Loop through all detections
    recognitions = []
    for i in range(num_detections):
        # Get the bounding box
        coords = detection_boxes[i]
        # Get the class
        class_id = detection_classes[i]

        # Get the coordinates of the area
        coords[0] = coords[0] * img.shape[0] - 5
        coords[1] = coords[1] * img.shape[1] - 5
        coords[2] = coords[2] * img.shape[0] + 5
        coords[3] = coords[3] * img.shape[1] + 5
        coords = coords.astype(int)

        # If the class_id is 1 (plate), extract the plate text
        if class_id == 1:
            # Process the plate
            ocr, typ = plate_detect(plate_cnn_driver, img, coords, save, log)
            if ocr is not None:
                recognitions.append((ocr, typ, coords))

        # Else it's a text area
        else:
            # Scan the text
            ocr, typ = text_detect(text_cnn_driver, img, coords, save, log)
            recognitions.append((ocr, typ, coords))

    # Draw the detections
    for ocr, typ, coords in recognitions:
        img = write_ocr(img, coords, ocr, typ)

    # Save the image
    if save is not False:
        cv2.imwrite(save, img)

    return img

# Function to scan a video file
def scan_video(plate_cnn_driver:Driver, text_cnn_driver:Driver, pd:Detector, video_file:str, save:str) -> None:
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
                res = scan_image(plate_cnn_driver, text_cnn_driver, pd, frame, False, False)

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

    # Create a NN for plate OCR functionality
    plate_cnn_driver = Driver('plate')
    # Create a NN for text OCR functionality
    text_cnn_driver = Driver('text')

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
            # Load the PLATE OCR NN
            load = input('Enter the path to the pretrained model for PLATE OCR NN [Enter = \"OCR/model_plate.pkl\"]: ')
            if load == '':
                load = 'OCR/model_plate.pkl'
            plate_cnn_driver.load_model(load)

            # Load the TEXT OCR NN
            load = input('Enter the path to the pretrained model for TEXT OCR NN [Enter = \"OCR/model_text.pkl\"]: ')
            if load == '':
                load = 'OCR/model_text.pkl'
            text_cnn_driver.load_model(load)

            # Load the Detector NN
            detector.load_from_checkpoint()
            nn_loaded = True
            continue

        # Scan an image
        elif choice == '2':
            if not nn_loaded:
                print(TEXT_RED + '>> NNs not loaded.' + TEXT_RESET)

                # Load the PLATE OCR NN
                load = input('Enter the path to the pretrained model for PLATE OCR NN [Enter = \"OCR/model_plate.pkl\"]: ')
                if load == '':
                    load = 'OCR/model_plate.pkl'
                plate_cnn_driver.load_model(load)

                # Load the TEXT OCR NN
                load = input('Enter the path to the pretrained model for TEXT OCR NN [Enter = \"OCR/model_text.pkl\"]: ')
                if load == '':
                    load = 'OCR/model_text.pkl'
                text_cnn_driver.load_model(load)

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
            scan_image(plate_cnn_driver, text_cnn_driver, detector, img, save_name)
            continue

        # Scan a directory
        elif choice == '3':
            if not nn_loaded:
                print(TEXT_RED + '>> NNs not loaded.' + TEXT_RESET)

                # Load the PLATE OCR NN
                load = input('Enter the path to the pretrained model for PLATE OCR NN [Enter = \"OCR/model_plate.pkl\"]: ')
                if load == '':
                    load = 'OCR/model_plate.pkl'
                plate_cnn_driver.load_model(load)

                # Load the TEXT OCR NN
                load = input('Enter the path to the pretrained model for TEXT OCR NN [Enter = \"OCR/model_text.pkl\"]: ')
                if load == '':
                    load = 'OCR/model_text.pkl'
                text_cnn_driver.load_model(load)

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
                scan_image(plate_cnn_driver, text_cnn_driver, detector, img, save_name)

            continue

        # Scan a video
        elif choice == '4':
            if not nn_loaded:
                print(TEXT_RED + '>> NNs not loaded.' + TEXT_RESET)

                # Load the PLATE OCR NN
                load = input('Enter the path to the pretrained model for PLATE OCR NN [Enter = \"OCR/model_plate.pkl\"]: ')
                if load == '':
                    load = 'OCR/model_plate.pkl'
                plate_cnn_driver.load_model(load)

                # Load the TEXT OCR NN
                load = input('Enter the path to the pretrained model for TEXT OCR NN [Enter = \"OCR/model_text.pkl\"]: ')
                if load == '':
                    load = 'OCR/model_text.pkl'
                text_cnn_driver.load_model(load)

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
            scan_video(plate_cnn_driver, text_cnn_driver, detector, video, save)
            continue
            
        # If there's an error
        else:
            print(TEXT_YELLOW + '>> Invalid choice.' + TEXT_RESET)
            continue

    return

if __name__ == '__main__':
    driver()
