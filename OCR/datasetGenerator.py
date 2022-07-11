"""
Generate a new dataset file starting from "License Plate Generator" output images
The dataset are generated in .csv format

Default names for datasets:
    dataset.csv         (FULL DATASET)
    dataset_train.csv   (TRAIN DATASET)
    dataset_valid.csv   (VALID DATASET)
    dataset_test.csv    (TEST DATASET)

For every image in the path, the dataset is generated with the following format:
    - The image is converted to a list of ints (gray pixel values)
        - CAR images have shape (200, 44) = 8800 pixels
        - MOTORCYCLE images have shape (106, 83) = 8798 pixels
          => generating the dataset, last 2 ints of a MOTO image will be 0
    - The image name is added to the dataset
    - The image name is converted to a list of ints
        - (22 letters + 10 numbers) x 7 positions (max)
          + 8 for discriminating plate types = 232 ints
        - discriminating bits (assume 1 or 0) are disposed in the following order:
            - CAR plate
            - MOTORCYCLE plate
            - AERONAUTICA MILITARE plate
            - CARABINIERI plate
            - ESERCITO plate
            - MARINA MILITARE plate
            - VIGILI DEL FUOCO plate
            - AUTO SPECIALE plate
        - letters 'I', 'O', 'Q', 'U' are not allowed
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Define colors
TEXT_RESET = '\033[0m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'

# Function to calculate the gap of the characters
def calculate_gap(c:str) -> int:
    # Numbers
    if ord(c) <= 57:
        return -39

    # Letters
    gap = 0
    if ord(c) > ord('I'):
        gap += 1
    if ord(c) > ord('O'):
        gap += 1
    if ord(c) > ord('Q'):
        gap += 1
    if ord(c) > ord('U'):
        gap += 1
    return gap

# Function to convert a number plate string in 232 ints
def convert_to_ints(string:str) -> list[int]:
    # Convert the "string" to a list of ints
    ints = []

    size = len(string) - 5
    for i in range(size):
        char = [0] * 32
        char[ord(string[i]) - 65 - calculate_gap(string[i])] = 1
        ints.extend(char)
    
    # Add remaining ints if necessary
    if size < 7:
        ints.extend([0] * (32 * (7 - size)))

    # Add last ints for discriminating plates
    ptype = [0] * 8
    if string.endswith('-auto'):
        ptype[0] = 1
    elif string.endswith('-moto'):
        ptype[1] = 1
    elif string.endswith('-aero'):
        ptype[2] = 1
    elif string.endswith('-cara'):
        ptype[3] = 1
    elif string.endswith('-eser'):
        ptype[4] = 1
    elif string.endswith('-mari'):
        ptype[5] = 1
    elif string.endswith('-vigf'):
        ptype[6] = 1
    elif string.endswith('-ausp'):
        ptype[7] = 1
    
    ints.extend(ptype)
    return ints

# Function to generate the dataset in .csv format
def generate_dataset_csv(path:str, filename:str) -> None:
    # Create a new output file "filename"
    dataset = open(filename, 'w+')

    # Get all the images in the path
    images = os.listdir(path)
    # For each image in the path
    for elem in tqdm(images):
        if elem.endswith('.png'):
            img = Image.open(path + elem)
            img = np.array(img)
            if elem.endswith('-moto.png') or elem.endswith('-ausp.png'):
                img = preprocess_moto(img)
            img = img.flatten()
            img = img.tolist()
            img = str(img)[1:-1].replace(' ', '').strip()
            dataset.write(img + ',' 
                + elem[:-4] + ','
                + str(convert_to_ints(elem[:-4]))[1:-1] + '\n')
    
    # Close the dataset file
    dataset.close()
    return

# Function to split the dataset
def split_dataset(path:str, train:str, test:str, valid:str, perc_train:int=80, perc_valid:int=10) -> None:
    dataset = pd.read_csv(path, header=None)
    train_d = open(train, 'w+')
    valid_d = open(valid, 'w+')
    test_d = open(test, 'w+')

    # Split the dataset in train and test
    train_dataset = dataset.sample(frac=perc_train/100, random_state=42)
    perc_valid = perc_valid / (100 - perc_train)
    valid_dataset = dataset.drop(train_dataset.index).sample(frac=perc_valid, random_state=42)
    test_dataset = dataset.drop(train_dataset.index).drop(valid_dataset.index).sample(frac=1, random_state=42)

    # Write the datasets
    train_dataset.to_csv(train_d, index=False, header=False, line_terminator='\n')
    valid_dataset.to_csv(valid_d, index=False, header=False, line_terminator='\n')
    test_dataset.to_csv(test_d, index=False, header=False, line_terminator='\n')

    train_d.close()
    valid_d.close()
    test_d.close()
    return

# Function to preprocess the MOTO plate before being written in the dataset
def preprocess_moto(X:np.ndarray) -> np.ndarray:
    """
        The MOTO image is (originally) of size <w:106, h:83>.
        The MOTO image is resized to <w:200, h:44> to be used in the network.
        The image will be cut in half horizontally to get 2 images 
        of size <106, 42> and <106, 41>, then both images will be scaled to <100, 44>
        and the two images will be concatenated to get the final image of size <200, 44>.
    """

    # Open the image from the array
    img = Image.fromarray(X.reshape(83, 106))

    # Cut the image
    img_1 = img.crop((0, 0, 106, 42))
    img_2 = img.crop((0, 42, 106, 83))

    # Resize the images
    img_1 = img_1.resize((100, 44), Image.ANTIALIAS)
    img_2 = img_2.resize((100, 44), Image.ANTIALIAS)

    # Concatenate the images
    img_1 = np.array(img_1)
    img_2 = np.array(img_2)
    img = np.concatenate((img_1, img_2), axis=1)
    
    # Return the image array
    return img
    

# Main function
def driver_main():
    choice = 1

    while choice != '0':
        # Get the user input
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Generate all datasets (full + train + validation + test).')
        print('  2. Generate full dataset only.')
        print('  3. Split full dataset into train, validation and test datasets.')
        print('  0. Exit.')
        choice = input(TEXT_YELLOW + 'Enter your choice: ' + TEXT_RESET)

        # Generate all datasets
        if choice == '1':
            images_path = input('Enter the file path containing the images [Enter = \"../LicensePlateGenerator/output/\"]: ')
            if images_path == '':
                images_path = '../LicensePlateGenerator/output/'

            filename = input('Enter the filename of the dataset [Enter = \"dataset.csv\"]: ')
            if filename == '':
                filename = 'dataset.csv'

            filename_train = input('Enter the filename of the train dataset [Enter = \"dataset_train.csv\"]: ')
            if filename_train == '':
                filename_train = 'dataset_train.csv'

            filename_test = input('Enter the filename of the test dataset [Enter = \"dataset_test.csv\"]: ')
            if filename_test == '':
                filename_test = 'dataset_test.csv'

            filename_valid = input('Enter the filename of the validation dataset [Enter = \"dataset_valid.csv\"]: ')
            if filename_valid == '':
                filename_valid = 'dataset_valid.csv'

            perc_train = input('Enter the percentage of the dataset to use for the train dataset [Enter = \"80%\"]: ')
            if perc_train == '':
                perc_train = 80
            perc_train = int(perc_train)

            perc_valid = input('Enter the percentage of the dataset to use for the validation dataset [Enter = \"10%\"]: ')
            if perc_valid == '':
                perc_valid = 10
            perc_valid = int(perc_valid)

            print(TEXT_GREEN 
                + '>> Generating dataset from {} into {} ...'.format(images_path, filename)
                + TEXT_RESET)      
            generate_dataset_csv(images_path, filename)
            
            print(TEXT_GREEN 
                + '>> Splitting dataset from {} to {} ({}%), {} ({}%), {} ({}%) ...'.format(filename,
                filename_train, perc_train, filename_valid, 100 - perc_train - perc_valid, filename_test, 100 - perc_train - perc_valid)
                + TEXT_RESET)
            split_dataset(filename, filename_train, filename_test, filename_valid, perc_train, perc_valid)
            print(TEXT_GREEN + '>> Done.' + TEXT_RESET)

        # Generate full dataset only
        elif choice == '2':
            images_path = input('Enter the file path containing the images [Enter = \"../LicensePlateGenerator/output/\"]: ')
            if images_path == '':
                images_path = '../LicensePlateGenerator/output/'

            filename = input('Enter the filename of the dataset [Enter = \"dataset.csv\"]: ')
            if filename == '':
                filename = 'dataset.csv'

            print(TEXT_GREEN 
                + '>> Generating dataset from {} into {} ...'.format(images_path, filename)
                + TEXT_RESET)      
            generate_dataset_csv(images_path, filename)
            print(TEXT_GREEN + '>> Done.' + TEXT_RESET)

        # Split full dataset into train and test datasets
        elif choice == '3':
            filename = input('Enter the filename of the dataset [Enter = \"dataset.csv\"]: ')
            if filename == '':
                filename = 'dataset.csv'

            filename_train = input('Enter the filename of the train dataset [Enter = \"dataset_train.csv\"]: ')
            if filename_train == '':
                filename_train = 'dataset_train.csv'

            filename_test = input('Enter the filename of the test dataset [Enter = \"dataset_test.csv\"]: ')
            if filename_test == '':
                filename_test = 'dataset_test.csv'

            filename_valid = input('Enter the filename of the validation dataset [Enter = \"dataset_valid.csv\"]: ')
            if filename_valid == '':
                filename_valid = 'dataset_valid.csv'

            perc_train = input('Enter the percentage of the dataset to use for the train dataset [Enter = \"80%\"]: ')
            if perc_train == '':
                perc_train = 80
            perc_train = int(perc_train)

            perc_valid = input('Enter the percentage of the dataset to use for the validation dataset [Enter = \"10%\"]: ')
            if perc_valid == '':
                perc_valid = 10
            perc_valid = int(perc_valid)

            print(TEXT_GREEN 
                + '>> Splitting dataset from {} to {} ({}%), {} ({}%), {} ({}%) ...'.format(filename,
                filename_train, perc_train, filename_valid, 100 - perc_train - perc_valid, filename_test, 100 - perc_train - perc_valid)
                + TEXT_RESET)
            split_dataset(filename, filename_train, filename_test, filename_valid, perc_train, perc_valid)
            print(TEXT_GREEN + '>> Done.' + TEXT_RESET)

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
