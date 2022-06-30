# Generate a new dataset file starting from "License Plate Generator" output images
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Input path containing images
images_path = '../LicensePlateGenerator/output/'

# Function to generate the dataset in .npy format
def generate_dataset_binary(path, output_path) -> None:
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

# Function to generate the dataset in .txt format
def generate_dataset_txt(path) -> None:
    # Create a new output file "dataset.txt"
    dataset = open('dataset.txt', 'w+')

    # Get all the images in the path
    images = os.listdir(path)
    # For each image in the path
    for elem in tqdm(images):
        if elem.endswith('.png'):
            img = Image.open(path + elem)
            img = np.array(img)
            img = img.flatten()
            img = img.tolist()
            dataset.write('{' + str(img) + ',' + elem[:-4] + '}\n')
    
    # Close the dataset file
    dataset.close()
    return

# Function to generate the dataset in .csv format
def generate_dataset_csv(path) -> None:
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
            dataset.write(str(img)[1:-1] + ',' + elem[:-4] + '\n')
    
    # Close the dataset file
    dataset.close()
    return


if __name__ == '__main__':
    generate_dataset_csv(images_path)
    # generate_dataset_txt(images_path)
    # generate_dataset_binary(images_path, 'dataset/')
