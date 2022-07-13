import os
import cv2
import sys
import numpy as np

sys.path.insert(0, '../OCR')
from driver import Driver
from detect import PlateDetect

# Define colors
TEXT_RESET = '\033[0m'
TEXT_RED = '\033[91m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'
TEXT_BLUE = '\033[94m'

# Dimension constants
# Car plates are 200x44 pixels
auto_image_width, auto_image_height = 200, 44
auto_min_ar, auto_max_ar = 4, 5.5 # Correct AR = 4.54

# Moto plates are 106x83 pixels
moto_image_width, moto_image_height = 106, 83
moto_min_ar, moto_max_ar = 1, 1.5 # Correct AR = 1.27

# Define paths
images_path = 'images/'
output_path = 'output/'

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
        img = cv2.resize(img, (moto_image_width, moto_image_height))

    # If there's an error
    else:
        print(TEXT_RED + '>> Image dimensions are not correct.' + TEXT_RESET)
        return None

    # Convert the image to a numpy array
    img = np.array(img, dtype=np.uint8)

    return img


# Driver function
def driver() -> None:
    # Create the output directory if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create a NN for OCR functionality
    cnn_driver = Driver()
    
    choice = 1
    model_loaded = False
    while choice != '0':
        # Get the user input
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Load a CNN pretrained model.')
        print('  2. Run detection on an image.')
        print('  0. Exit.')
        choice = input(TEXT_YELLOW + 'Enter your choice: ' + TEXT_RESET)

        # Exit
        if choice == '0':
            print(TEXT_YELLOW + 'Exiting...' + TEXT_RESET)
            break

        # Load a CNN pretrained model
        elif choice == '1':
            load = input('Enter the path to the pretrained model [Enter = \"../OCR/model.pkl\"]: ')
            if load == '':
                load = '../OCR/model.pkl'
            cnn_driver.load_model(load)

        # Run detection on an image
        elif choice == '2':
            print('Input images will be taken from \"' + images_path + '\" folder.')
            in_name = input('Enter the name of the image [Enter = \"test.jpg\"]: ')
            if in_name == '':
                in_name = 'test.jpg'
            
            print('Output images will be saved to \"' + output_path + '\" folder.')
            out_name = input('Enter the name of the output image [Enter = \"test_out.jpg\"]: ')
            if out_name == '':
                out_name = 'test_out.jpg'

            # Run the detection
            pd = PlateDetect()
            img = pd.detect(images_path + in_name)

            # Modify the image so that it can be passed to the CNN for OCR
            img = process_image(img)
            
            # Analyze the image with the CNN
            if img is not None:
                ocr = cnn_driver.forward(img)
                pd.write_ocr(ocr)
                pd.save_image(output_path + out_name)
                print(TEXT_GREEN + '>> Detection and OCR completed successfully.' + TEXT_RESET)

    return

if __name__ == '__main__':
    driver()
