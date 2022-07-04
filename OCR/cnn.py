"""
  Implementation of a Convolution Neural Network
  for Italian Number Plates recognition.

  > Input image is a grayscale image of a number plate
    of size <w:200, h:44, c:1>

  > Output is a vector of length 134, where each element is a probability of the
    corresponding digit to be a certain letter/number.

  > The network consists of a series of convolutional layers, followed by a
    series of fully connected layer.
"""

import io
from math import ceil, sqrt
from sklearn.model_selection import train_test_split
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Define colors
TEXT_RESET = '\033[0m'
TEXT_RED = '\033[91m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'
TEXT_BLUE = '\033[94m'

if torch.cuda.is_available():
    print(TEXT_GREEN 
        + '>> CUDA is available ({}).'.format(torch.cuda.get_device_name())
        + TEXT_RESET)
    gpu = torch.device("cuda:0")
    cpu = torch.device("cpu")
else:
    gpu = cpu = torch.device("cpu")

# Class for the Convolution Neural Network
class ConvNet(nn.Module):
    # Constructor
    def __init__(self) -> None:
        super(ConvNet, self).__init__()

        # Initial image size: <w:200, h:44, c:1>
        # Convolutional layer 1: <w:200, h:44, c:1> -> <w:99, h:21, c:6>
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Convolutional layer 2: <w:99, h:21, c:6> -> <w:48, h:9, c:16>
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Fully connected layer: <w:48, h:9, c:16> -> (26 * 4 + 10 * 3) = 134 neurons
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16 * 9 * 48, 16 * 9 * 24),
            torch.nn.ReLU(),
            torch.nn.Linear(16 * 9 * 24, 16 * 9 * 12),
            torch.nn.ReLU(),
            torch.nn.Linear(16 * 9 * 12, 72 * 12),
            torch.nn.ReLU(),
            torch.nn.Linear(72 * 12, 26 * 4 + 10 * 3)
        )
        return

    # Forward function: generate a prediction given the input "x"
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, 16 * 9 * 48)
        out = self.fc(out)
        return out

    # Function to train the network
    def train_net(self, X_train:torch.Tensor, Y_train:torch.Tensor, epochs:int, learning_rate:float) -> None:
        # Define Loss function and optimizer
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

        # Train the network
        for epoch in range(epochs):
            for i, data in enumerate(X_train.values):
                # Convert the data to torch tensor
                data = torch.from_numpy(data.reshape(1, 1, 44, 200)).float().to(gpu)
                # Forward pass
                output = self.forward(data).to(gpu)
                # Calculate the loss
                loss = criterion(output[0], torch.from_numpy(Y_train.values[i]).float().to(gpu))
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, epochs, i+1, len(X_train), loss.item()))
                
        print('Finished Training')
        return

    # Function to test the network
    def test_net(self, X_test:torch.Tensor, Y_test:torch.Tensor, save_preds:str=None) -> None:
        # Test the network
        self.eval()
        with torch.no_grad():
            correct = 0
            for i, data in enumerate(X_test.values):
                # Convert the data to torch tensor
                data = torch.from_numpy(data.reshape(1, 1, 44, 200)).float().to(gpu)
                # Forward pass
                output = self.forward(data).to(cpu)
                # Check the results
                correct += self.check_results(output[0], Y_test.values[i])

                # Save the predictions if required
                if save_preds is not None:
                    self.save_predictions(X_test = X_test.values[i],
                        Y_pred = output[0],
                        Y_test = Y_test.values[i],
                        filename = save_preds)

            print(TEXT_BLUE 
                + 'Accuracy: {:.2f}%'.format(100 * correct / (7 * len(X_test)))
                + TEXT_RESET)
            print(TEXT_BLUE 
                + 'Total: {}, Correct: {}'.format(7 * len(X_test), correct)
                + TEXT_RESET)
        return

    # Function to save the network
    def save(self, filename:str) -> None:
        torch.save(self.state_dict(), filename)
        print(TEXT_GREEN + 'Model saved to {}'.format(filename) + TEXT_RESET)
        return

    # Function to check the results of the network
    def check_results(self, net_output:torch.Tensor, Y_test:np.ndarray) -> int:
        correct = 0

        # Check the first 26 positions (first letter)
        out_index = np.argmax(net_output[0:26])
        sol_index = np.argmax(Y_test[0:26])
        if out_index == sol_index:
            correct += 1

        # Check the second 26 positions (second letter)
        out_index = np.argmax(net_output[26:52])
        sol_index = np.argmax(Y_test[26:52])
        if out_index == sol_index:
            correct += 1
        
        # Check the third 10 positions (first number)
        out_index = np.argmax(net_output[52:62])
        sol_index = np.argmax(Y_test[52:62])
        if out_index == sol_index:
            correct += 1

        # Check the fourth 10 positions (second number)
        out_index = np.argmax(net_output[62:72])
        sol_index = np.argmax(Y_test[62:72])
        if out_index == sol_index:
            correct += 1

        # Check the fifth 10 positions (third number)
        out_index = np.argmax(net_output[72:82])
        sol_index = np.argmax(Y_test[72:82])
        if out_index == sol_index:
            correct += 1

        # Check the sixth 26 positions (third letter)
        out_index = np.argmax(net_output[82:108])
        sol_index = np.argmax(Y_test[82:108])
        if out_index == sol_index:
            correct += 1

        # Check the seventh 26 positions (fourth letter)
        out_index = np.argmax(net_output[108:134])
        sol_index = np.argmax(Y_test[108:134])
        if out_index == sol_index:
            correct += 1
        
        return correct

    # Function to convert the network output to a string
    def output_to_string(self, net_output:torch.Tensor) -> str:
        # Convert the output to a string
        out_string = ''
        out_index = np.argmax(net_output[0:26])
        out_string += chr(out_index + 65)
        out_index = np.argmax(net_output[26:52])
        out_string += chr(out_index + 65)
        out_index = np.argmax(net_output[52:62])
        out_string += chr(out_index + 48)
        out_index = np.argmax(net_output[62:72])
        out_string += chr(out_index + 48)
        out_index = np.argmax(net_output[72:82])
        out_string += chr(out_index + 48)
        out_index = np.argmax(net_output[82:108])
        out_string += chr(out_index + 65)
        out_index = np.argmax(net_output[108:134])
        out_string += chr(out_index + 65)
        return out_string

    # Function to save the predictions in string format
    def save_predictions(self, X_test:np.ndarray, Y_pred:torch.Tensor, Y_test:np.ndarray, filename:str) -> None:
        f = open(filename, 'a+')
        
        # Convert the output to a string
        out_string = self.output_to_string(Y_pred)
        test_string = self.output_to_string(Y_test)
        X_test = str(np.array(X_test).flatten().tolist())[1:-1]

        # Write the output to the file
        f.write(X_test + ',' + out_string + ',' + test_string + '\n')
        f.close()
        return

    # Function to show "num" images with their predictions
    def show_predictions(self, num:int, preds:str='preds.csv') -> None:
        # Read the predictions file and get the first "num" images
        with open(preds, 'r') as f:
            lines = ''
            for _ in range(num):
                lines += f.readline()

        # Convert the lines to a dataframe
        lines = io.StringIO(lines)
        lines = pd.read_csv(lines, header=None)
        lines = lines.iloc[:num]

        # Get the images and predictions
        img = lines.iloc[:, :-2]
        Y_pred = lines.iloc[:, -2]

        # Plot
        rows = ceil(sqrt(num))
        fig = plt.figure(figsize=(rows, rows / 2), constrained_layout=True)
        for i in range(num):
            # Get the image, prediction and test string
            img_array = np.array(img.iloc[i]).reshape(44, 200)
            pred_array = Y_pred.iloc[i]
            
            # Plot the image
            fig.add_subplot(rows, rows, i + 1)
            plt.imshow(img_array, cmap='gray')
            plt.axis('off')
            plt.title(pred_array)

        plt.show()
        return


# Class to drive the program
class Driver:
    def __init__(self) -> None:
        self.X_train:pd.DataFrame = None
        self.Y_train:pd.DataFrame = None
        self.X_test:pd.DataFrame = None
        self.Y_test:pd.DataFrame = None
        self.net:ConvNet = ConvNet().to(gpu)

        self.save_model:bool = False
        self.save_model_path:str = None
        self.save_preds:bool = False
        self.save_preds_path:str = None
        self.model_loaded:bool = False
        return

    # Function to load the dataset (full)
    def load_dataset(self, filename:str) -> None:
        # Load the dataset
        print(TEXT_GREEN + '>> Loading dataset ...' + TEXT_RESET)
        data = pd.read_csv(filename, header=None)

        # Split and normalize the dataset
        X = data.iloc[:, :8800] / 255
        Y = data.iloc[:, 8801:]
        
        # Split the dataset into training and testing
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print(TEXT_GREEN + '>> Dataset loaded.' + TEXT_RESET)
        return

    # Function to load the train dataset
    def load_train(self, filename:str) -> None:
        # Load the dataset
        print(TEXT_GREEN + '>> Loading train dataset ...' + TEXT_RESET)
        data = pd.read_csv(filename, header=None)

        # Split and normalize the dataset
        self.X_train = data.iloc[:, :8800] / 255
        self.Y_train = data.iloc[:, 8801:]
        
        print(TEXT_GREEN + '>> Train dataset loaded.' + TEXT_RESET)
        return

    # Function to load the test dataset
    def load_test(self, filename:str) -> None:
        # Load the dataset
        print(TEXT_GREEN + '>> Loading test dataset ...' + TEXT_RESET)
        data = pd.read_csv(filename, header=None)

        # Split and normalize the dataset
        self.X_test = data.iloc[:, :8800] / 255
        self.Y_test = data.iloc[:, 8801:]
        
        print(TEXT_GREEN + '>> Test dataset loaded.' + TEXT_RESET)
        return

    # Function to load the model
    def load_model(self, filename:str) -> None:
        # Load the model
        print(TEXT_GREEN + '>> Loading model...' + TEXT_RESET)

        if os.path.exists(filename):
            self.net.load_state_dict(torch.load(filename))
            self.model_loaded = True
            print(TEXT_GREEN + '>> Model loaded successfully ({}).'.format(filename) + TEXT_RESET)
        else:
            print(TEXT_RED + '>> Model file does not exist ({}).'.format(filename) + TEXT_RESET)
            
        return

    # Function to train the network
    def train(self, epochs:int, learning_rate:float) -> None:
        # Train the network
        print(TEXT_GREEN 
            + '>> Training for {} epochs with learning rate = {} ...'.format(epochs, learning_rate)
            + TEXT_RESET)

        start_time = time.time()
        self.net.train_net(self.X_train, self.Y_train, epochs, learning_rate)
        self.model_loaded = True
        end_time = time.time()

        print(TEXT_GREEN 
            + '>> Training finished in {} seconds.'.format(end_time - start_time) 
            + TEXT_RESET)

        # Save the trained network model if required
        if self.save_model:
            print(TEXT_GREEN + '>> Saving model to {} ...'.format(self.save_model_path) + TEXT_RESET)
            self.net.save(self.save_model_path)
        return

    # Function to test the network
    def test(self) -> None:
        # Test the network
        print(TEXT_GREEN 
            + '>> Testing the network on {} samples ...'.format(len(self.X_test))
            + TEXT_RESET)

        start_time = time.time()
        self.net.test_net(self.X_test, self.Y_test, self.save_preds_path)
        end_time = time.time()

        print(TEXT_GREEN 
            + '>> Testing finished in {} seconds.'.format(end_time - start_time) 
            + TEXT_RESET)
        return

    # Function to show the predictions with the images associated with them
    def show_preds(self, filename:str, num:int=64) -> None:
        # Show the predictions
        self.net.show_predictions(num, filename)
        return


# Main function
def driver_main():
    d = Driver()
    choice = 0

    while choice != '4':
        # Get the user input
        print(TEXT_YELLOW + '>> Driver helper. Select the function to run. Type:' + TEXT_RESET)
        print('  1. Load dataset.')
        print('  2. Train the network.')
        print('  3. Test the network.')
        print('  4. Load a network pretrained model.')
        print('  5. Show some images with their predictions.')
        print('  0. Exit.')
        choice = input(TEXT_YELLOW + 'Enter your choice: ' + TEXT_RESET)

        # Load the dataset
        if choice == '1':
            print(TEXT_YELLOW + '>> Select the option. Type:' + TEXT_RESET)
            print('  1. Load entire dataset (train + test).')
            print('  2. Load train dataset only.')
            print('  3. Load test dataset only.')
            print('  0. Back.')
            choice = input(TEXT_YELLOW + 'Enter your choice: ' + TEXT_RESET)

            # Back
            if choice == '0':
                continue

            # Load entire dataset
            elif choice == '1':
                filename = input('Enter the filename [Enter = \"dataset.csv\"]: ')
                if filename == '':
                    filename = 'dataset.csv'
                d.load_dataset(filename)

            # Load train dataset only
            elif choice == '2':
                filename = input('Enter the filename [Enter = \"dataset_train.csv\"]: ')
                if filename == '':
                    filename = 'dataset_train.csv'
                d.load_train(filename)

            # Load test dataset only
            elif choice == '3':
                filename = input('Enter the filename [Enter = \"dataset_test.csv\"]: ')
                if filename == '':
                    filename = 'dataset_test.csv'
                d.load_test(filename)

            # Invalid choice
            else:
                print(TEXT_RED + '>> Invalid choice.' + TEXT_RESET)
                continue

        # Train the network
        elif choice == '2':
            if d.X_train is None:
                print(TEXT_RED + '>> Training dataset not loaded.' + TEXT_RESET)
                dataset_path = input('Enter the path to the training dataset [Enter = \"dataset_train.csv\"]: ')
                if dataset_path == '':
                    dataset_path = 'dataset_train.csv'
                d.load_train(dataset_path)

            save = input('Enter the path to save the trained model [Enter = \"model.pkl\" | \"n\" = None]: ')
            if save == '':
                d.save_model_path = 'model.pkl'
                d.save_model = True
            elif save == 'n':
                d.save_model = False
            else:
                d.save_model_path = save
                d.save_model = True

            epochs = input('Enter the number of epochs to train [Enter = \"3\"]: ')
            if epochs == '':
                epochs = 3
            epochs = int(epochs)

            learning_rate = input('Enter the learning rate [Enter = \"0.001\"]: ')
            if learning_rate == '':
                learning_rate = 0.001
            learning_rate = float(learning_rate)

            d.train(epochs, learning_rate)

        # Test the network
        elif choice == '3':
            if d.X_test is None:
                print(TEXT_RED + '>> Test dataset not loaded.' + TEXT_RESET)
                dataset_path = input('Enter the path to the test dataset [Enter = \"dataset_test.csv\"]: ')
                if dataset_path == '':
                    dataset_path = 'dataset_test.csv'
                d.load_test(dataset_path)

            if not d.model_loaded:
                print(TEXT_RED + '>> Model not loaded.' + TEXT_RESET)
                load = input('Enter the path to the trained model [Enter = \"model.pkl\"]: ')
                if load == '':
                    load = 'model.pkl'
                d.load_model(load)

            preds = input('Enter the path to save the predictions [Enter = \"preds.csv\" | \"n\" = None]: ')
            if preds == '':
                preds = 'preds.csv'
                f = open(preds, 'w+')
                f.close()
                d.save_preds_path = preds
                d.save_preds = True
            elif preds == 'n':
                d.save_preds = False
            else:
                f = open(preds, 'w+')
                f.close()
                d.save_preds_path = preds
                d.save_preds = True

            d.test()

        # Load a network pretrained model
        elif choice == '4':
            load = input('Enter the path to the pretrained model [Enter = \"model.pkl\"]: ')
            if load == '':
                load = 'model.pkl'
            d.load_model(load)

        # Show some images with their predictions
        elif choice == '5':
            preds = input('Enter the path to the predictions file [Enter = \"preds.csv\"]: ')
            if preds == '':
                preds = 'preds.csv'

            num = input('Enter the number of images to show [Enter = \"64\"]: ')
            if num == '':
                num = 64
            num = int(num)

            d.show_preds(preds, num)

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
