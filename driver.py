import os
import cv2
import sys
import numpy as np

sys.path.insert(0, './OCR')
from OCR.driver import Driver
from PlateDetector.detect import PlateDetect

# Define colors
TEXT_RESET = '\033[0m'
TEXT_RED = '\033[91m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'
TEXT_BLUE = '\033[94m'

# Dimension constants
# Car plates are 200x44 pixels
auto_image_width, auto_image_height = 200, 44
auto_min_ar, auto_max_ar = 2, 6 # Correct AR = 4.54

# Moto plates are 106x83 pixels
moto_image_width, moto_image_height = 106, 83
moto_min_ar, moto_max_ar = 1, 1.5 # Correct AR = 1.27

# Define paths
data_path = 'data/'
input_path = data_path + 'input/'
output_path = data_path + 'output/'

# Function to convert an image to the correct format for CNN
def process_image(img:cv2.Mat) -> np.ndarray:
    # Check the image dimensions
    img_ar = img.shape[1] / img.shape[0]
    print('Image dimensions: ' + str(img.shape[1]) + 'x' + str(img.shape[0]))
    print('Image aspect ratio: ' + str(img_ar))

    # If it's a car plate, resize it to the correct dimensions
    if img_ar >= auto_min_ar and img_ar <= auto_max_ar:
        # Resize the image to the correct dimensions
        img = cv2.resize(img, (auto_image_width, auto_image_height))

    # If it's a motorcycle plate, resize it to the correct dimensions
    elif img_ar >= moto_min_ar and img_ar <= moto_max_ar:
        # Resize the image to the correct dimensions
        img = cv2.resize(img, (moto_image_width, moto_image_height))##########################

    # If there's an error
    else:
        print(TEXT_RED + '>> Image dimensions are not correct.' + TEXT_RESET)
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
    plate_detect = PlateDetect('PlateDetector/')
    
    choice = 1
    nn_loaded = False
    while choice != '0':
        # Get the user input
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Load pretrained models of OCR NN and Detector NN.')
        print('  2. Scan an image.')
        print('  3. Scan a directory.')
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
            plate_detect.load_from_checkpoint()
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
                plate_detect.load_from_checkpoint()
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
            img_array = np.asarray(img)

            # Detect the plate
            crop, coords = plate_detect.detect_and_crop(img_array)
            # print(coords)

            # If the plate is detected
            if crop is not None:
                # Process the image
                # cv2.imshow('Image', crop)
                # cv2.waitKey(0)
                crop = process_image(crop)

                # If the image is processed
                if crop is not None:
                    # Predict the plate
                    text, ptype = cnn_driver.forward(crop)

                    # If the plate is predicted
                    if text is not None:
                        # Print the text
                        print(TEXT_BLUE + '>> Recognised plate number: ' + text + TEXT_RESET)
                        print(TEXT_BLUE + '>> Recognised plate type: ' + ptype + TEXT_RESET)

                        res = write_ocr(img, coords, text)

                        # Save the image
                        if save is not False:
                            cv2.imwrite(output_path + save, res)
                    else:
                        print(TEXT_RED + '>> Plate not recognised.' + TEXT_RESET)
            else:
                print(TEXT_RED + '>> Plate not detected.' + TEXT_RESET)

            # cv2.destroyAllWindows()
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
                plate_detect.load_from_checkpoint()
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
                print(im)
                # Load the image
                img = cv2.imread(os.path.join(input_path, dir_path, im))
                img_array = np.asarray(img)

                # Detect the plate
                crop, coords = plate_detect.detect_and_crop(img_array)

                # If the plate is detected
                if crop is not None:
                    # Process the image
                    crop = process_image(crop)

                    # If the image is processed
                    if crop is not None:
                        # Predict the plate
                        text, ptype = cnn_driver.forward(crop)

                        # If the plate is predicted
                        if text is not None:
                            # Print the text
                            print(TEXT_BLUE + '>> Recognised plate number: ' + text + TEXT_RESET)
                            print(TEXT_BLUE + '>> Recognised plate type: ' + ptype + TEXT_RESET)

                            res = write_ocr(img, coords, text)

                            # Save the image
                            if save is not False:
                                cv2.imwrite(os.path.join(output_path, save, im), res)
                        else:
                            print(TEXT_RED + '>> Plate not recognised.' + TEXT_RESET)
                else:
                    print(TEXT_RED + '>> Plate not detected.' + TEXT_RESET)

            continue
            
        # If there's an error
        else:
            print(TEXT_YELLOW + '>> Invalid choice.' + TEXT_RESET)
            continue

    return

if __name__ == '__main__':
    driver()
