# Generate a new dataset file starting from "License Plate Generator" output images
import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define colors
TEXT_GREEN = '\033[92m'
TEXT_RESET = '\033[0m'

# Input path containing images
images_path = '../LicensePlateGenerator/output/'

# Function to convert a number plate string in 134 ints
def convert_to_ints(string:str) -> list[int]:
    # Convert the string to a list of ints
    ints = []

    letter = [0] * 26
    letter[ord(string[0]) - 65] = 1
    ints.extend(letter)
    letter = [0] * 26
    letter[ord(string[1]) - 65] = 1
    ints.extend(letter)
    number = [0] * 10
    number[ord(string[2]) - 48] = 1
    ints.extend(number)
    number = [0] * 10
    number[ord(string[3]) - 48] = 1
    ints.extend(number)
    number = [0] * 10
    number[ord(string[4]) - 48] = 1
    ints.extend(number)
    letter = [0] * 26
    letter[ord(string[5]) - 65] = 1
    ints.extend(letter)
    letter = [0] * 26
    letter[ord(string[6]) - 65] = 1
    ints.extend(letter)

    return ints

# Function to generate the dataset in .npy format
def generate_dataset_binary(path:str, output_path:str) -> None:
    # Get all the images in the path
    images = os.listdir(path)
    # Create a new folder for the dataset
    os.makedirs(output_path)
    # For each image in the path
    for image in tqdm(images):
        if image.endswith('.png'):
            # Open the image
            img = Image.open(path + image)
            # Convert the image to numpy array
            img = np.array(img)
            # Save the image in the new folder
            np.save(output_path + "/" + image, img)
    return

# Function to generate the dataset in .csv format
def generate_dataset_csv(path:str) -> None:
    # Create a new output file "dataset.csv"
    dataset = open('dataset.csv', 'w+')

    # Get all the images in the path
    images = os.listdir(path)
    # For each image in the path
    for elem in tqdm(images):
        if elem.endswith('.png'):
            img = Image.open(path + elem)
            img = np.array(img)
            img = img.flatten()
            img = img.tolist()
            dataset.write(str(img)[1:-1] + ',' 
                + elem[:-4] + ','
                + str(convert_to_ints(elem[:-4]))[1:-1] + '\n')
    
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
    

if __name__ == '__main__':
    print(TEXT_GREEN 
        + '>> Generating dataset from {} ...'.format(images_path)
        + TEXT_RESET)
    generate_dataset_csv(images_path)
    
    print(TEXT_GREEN 
        + '>> Splitting dataset from {} to {} ({}%), {} ({}%) ...'.format('dataset.csv',
        'dataset_train.csv', 80, 'dataset_test.csv', 20)
        + TEXT_RESET)
    split_dataset('dataset.csv', 'dataset_train.csv', 'dataset_test.csv')
    print(TEXT_GREEN + '>> Done.' + TEXT_RESET)
