"""
Generate a new dataset file starting from "License Plate Generator" output images
The dataset is generated in .csv format

For every image in the path, the dataset is generated with the following format:
    - The image is converted to a list of ints (gray pixel values)
        - CAR images have shape (200, 44) = 8800 pixels
        - MOTORCYCLE images have shape (106, 83) = 8798 pixels
          => generating the dataset, last 2 ints of a MOTO image will be 0
    - The image name is added to the dataset
    - The image name is converted to a list of ints
        - (22 letters + 10 numbers) x 7 positions 
          + 1 for discriminating plate types (0 = CAR, 1 = MOTO) = 225 ints
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

# Function to convert a number plate string in 225 ints
def convert_to_ints(string:str) -> list[int]:
    # Convert the "string" to a list of ints
    ints = []

    for i in range(7):
        char = [0] * 32
        char[ord(string[i]) - 65 - calculate_gap(string[i])] = 1
        ints.extend(char)

    # Add the last int for discriminating plate types
    if len(string) == 8 and string[7] == 'm': # MOTO
        ints.append(1)
    else:
        ints.append(0)
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
            if elem.endswith('m.png'):
                img = np.append(img, [0, 0])
            img = img.flatten()
            img = img.tolist()
            img = str(img)[1:-1].replace(' ', '').strip()
            dataset.write(img + ',' 
                + elem[:-4] + ','
                + str(convert_to_ints(elem[:-3]))[1:-1] + '\n')
    
    # Close the dataset file
    dataset.close()
    return

# Function to split the dataset
def split_dataset(path:str, train:str, test:str, perc:int=80) -> None:
    dataset = pd.read_csv(path, header=None)
    train_d = open(train, 'w+')
    test_d = open(test, 'w+')

    # Split the dataset in train and test
    train_dataset = dataset.sample(frac=perc/100, random_state=42)
    test_dataset = dataset.drop(train_dataset.index).sample(frac=1, random_state=42)

    # Write the datasets
    train_dataset.to_csv(train_d, index=False, header=False, line_terminator='\n')
    test_dataset.to_csv(test_d, index=False, header=False, line_terminator='\n')

    train_d.close()
    test_d.close()
    return
    

# Main function
def driver_main():
    choice = 1

    while choice != '0':
        # Get the user input
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Generate all datasets (full + train + test).')
        print('  2. Generate full dataset only.')
        print('  3. Split full dataset into train and test datasets.')
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

            perc = input('Enter the percentage of the dataset to use for the train dataset [Enter = \"80%\"]: ')
            if perc == '':
                perc = 80
            perc = int(perc)

            print(TEXT_GREEN 
                + '>> Generating dataset from {} into {} ...'.format(images_path, filename)
                + TEXT_RESET)      
            generate_dataset_csv(images_path, filename)
            
            print(TEXT_GREEN 
                + '>> Splitting dataset from {} to {} ({}%), {} ({}%) ...'.format(filename,
                filename_train, perc, filename_test, 100 - perc)
                + TEXT_RESET)
            split_dataset(filename, filename_train, filename_test, perc)
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

            perc = input('Enter the percentage of the dataset to use for the train dataset [Enter = \"80%\"]: ')
            if perc == '':
                perc = 80
            perc = int(perc)

            print(TEXT_GREEN 
                + '>> Splitting dataset from {} to {} ({}%), {} ({}%) ...'.format(filename,
                filename_train, perc, filename_test, 100 - perc)
                + TEXT_RESET)
            split_dataset(filename, filename_train, filename_test, perc)
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
