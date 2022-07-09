"""
  Implementation of a Convolution Neural Network
  for Italian Number Plates recognition.

  > Input image is a grayscale image of a number plate
    of size <w:200, h:44, c:1> (CAR images) or <w:106, h:83, c:1> (MOTORCYCLE images)

  > Output is a vector of length 225, where each element is a probability of the
    corresponding digit to be a certain letter/number.
    Every character is an array of 32 ints identifiing the specific character
    There are 7 characters in total, hence 32 * 7 = 224
    The last element discriminates the plate type (0 = CAR, 1 = MOTORCYCLE)

  > The network consists of a series of convolutional layers, followed by a
    series of fully connected layer.
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from io import StringIO
from math import ceil, sqrt
import matplotlib.pyplot as plt

# Define colors
TEXT_RESET = '\033[0m'
TEXT_GREEN = '\033[92m'
TEXT_BLUE = '\033[94m'

# Class for the Convolution Neural Network
class ConvNet(nn.Module):
    # Constructor
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.gpu:torch.device = None
        self.cpu:torch.device = None

        # Define the criterion for loss computation
        self.criterion = nn.CrossEntropyLoss()

        # Array containing the loss function values
        self.train_loss_array = []
        self.valid_loss_array = []
        self.valid_accuracy_array = []

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

        # Fully connected layer: <w:48, h:9, c:16> -> (32 * 7) + 1 = 225 neurons
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16 * 9 * 48, 16 * 9 * 24),
            torch.nn.ReLU(),
            torch.nn.Linear(16 * 9 * 24, 16 * 9 * 12),
            torch.nn.ReLU(),
            torch.nn.Linear(16 * 9 * 12, 72 * 12),
            torch.nn.ReLU(),
            torch.nn.Linear(72 * 12, 32 * 7 + 1)
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
    def train_net(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, epochs:int, learning_rate:float, X_valid:pd.DataFrame, Y_valid:pd.DataFrame) -> None:
        # Define optimizer
        self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

        # Train the network
        for epoch in range(epochs):
            for i, data in enumerate(X_train.values):
                # Convert the data to torch tensor
                data = torch.from_numpy(data.reshape(1, 1, 44, 200)).float().to(self.gpu)
                # Forward pass
                output = self.forward(data).to(self.gpu)
                # Calculate the loss
                loss = self.criterion(output[0], torch.from_numpy(Y_train.values[i]).float().to(self.gpu))
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 1000 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, epochs, i+1, len(X_train), loss.item()))
                    # Save the train loss function values every 1000 iterations
                    self.train_loss_array.append(loss.item())
                
            # For each epoch validate the network
            self.validate_net(X_valid, Y_valid)

        print('Finished Training')
        self.show_loss()
        return

    # Function to validate the network
    def validate_net(self, X_valid:pd.DataFrame, Y_valid:pd.DataFrame) -> None:
        # Validate the network
        self.eval()
        with torch.no_grad():
            correct = loss = 0
            for i, data in enumerate(X_valid.values):
                # Convert the data to torch tensor
                data = torch.from_numpy(data.reshape(1, 1, 44, 200)).float().to(self.gpu)
                # Forward pass
                output = self.forward(data).to(self.gpu)
                # Calculate the loss
                l = self.criterion(output[0], torch.from_numpy(Y_valid.values[i]).float().to(self.gpu))
                loss += l.item()
                # Calculate the accuracy
                output = output.to(self.cpu)
                correct += self.check_results(output[0], Y_valid.values[i])

            # Calculate the average loss and accuracy
            loss /= len(X_valid)
            self.valid_loss_array.append(loss)

            accuracy = 100 * correct / (8 * len(X_valid))
            self.valid_accuracy_array.append(accuracy)

        print(TEXT_BLUE + 'Accuracy of the network on the validation set: {:.2f}%'.format(accuracy) + TEXT_RESET)
        return

    # Function to test the network
    def test_net(self, X_test:pd.DataFrame, Y_test:pd.DataFrame, save_preds:str=None) -> None:
        # Test the network
        self.eval()
        with torch.no_grad():
            correct = 0
            for i, data in enumerate(X_test.values):
                # Convert the data to torch tensor
                data = torch.from_numpy(data.reshape(1, 1, 44, 200)).float().to(self.gpu)
                # Forward pass
                output = self.forward(data).to(self.cpu)
                # Check the results
                correct += self.check_results(output[0], Y_test.values[i])

                # Save the predictions if required
                if save_preds is not None:
                    self.save_predictions(X_test = X_test.values[i],
                        Y_pred = output[0],
                        Y_test = Y_test.values[i],
                        filename = save_preds)

            print(TEXT_BLUE 
                + 'Accuracy: {:.2f}%'.format(100 * correct / (8 * len(X_test)))
                + TEXT_RESET)
            print(TEXT_BLUE 
                + 'Total: {}, Correct: {}'.format(8 * len(X_test), correct)
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

        # For every character in the net output
        for i in range(7):
            # Check the 32 positions
            out_index = np.argmax(net_output[32 * i:32 * (i + 1)])
            sol_index = np.argmax(Y_test[32 * i:32 * (i + 1)])
            if out_index == sol_index:
                correct += 1

        # Check the last bit discriminating the plates
        if round(float(net_output[-1])) == Y_test[-1]:
            correct += 1

        return correct

    # Function to calculate the gap of the characters
    def calculate_gap(self, index:int) -> int:
        # Numbers
        if index >= 22:
            return -39

        # Letters
        gap = 0
        if index > ord('I') - 66:
            gap += 1
        if index > ord('O') - 67:
            gap += 1
        if index > ord('Q') - 68:
            gap += 1
        if index > ord('U') - 69:
            gap += 1
        return gap

    # Function to convert the network output to a string
    def output_to_string(self, net_output:torch.Tensor) -> str:
        # Convert the output to a string
        out_string = ''

        for i in range(7):
            out_index = np.argmax(net_output[32 * i:32 * (i + 1)])
            out_string += chr(out_index + 65 + self.calculate_gap(out_index))

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
        lines = StringIO(lines)
        lines = pd.read_csv(lines, header=None)

        # Get the images and predictions
        img = lines.iloc[:, :-2]
        Y_pred = lines.iloc[:, -2]
        Y_test = lines.iloc[:, -1]

        # Plot
        rows = ceil(sqrt(num))
        fig = plt.figure(figsize=(rows, rows / 2), constrained_layout=True)
        for i in range(num):
            # Get the image, prediction and test string
            # If the expected value of the string ends with 'm', than is a MOTO plate
            if Y_test.iloc[i][-1] == 'm':
                img_array = self.reverse_moto(img.iloc[i])
            else:
                img_array = np.array(img.iloc[i]).reshape(44, 200)
            
            prediction = Y_pred.iloc[i]
            
            # Plot the image
            fig.add_subplot(rows, rows, i + 1)
            plt.imshow(img_array, cmap='gray')
            plt.axis('off')
            plt.title(prediction)

        plt.show()
        return

    # Function to show the loss function
    def show_loss(self) -> None:
        # Plot the train loss function
        plot = plt.figure(figsize=(10, 5), constrained_layout=True)
        plot.add_subplot(1, 3, 1)
        plt.plot(self.train_loss_array)
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        # Plot the validation loss function
        plot.add_subplot(1, 3, 2)
        plt.plot(self.valid_loss_array)
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Plot the validation accuracy function
        plot.add_subplot(1, 3, 3)
        plt.plot(self.valid_accuracy_array)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.savefig('graphs.png')
        plt.show()
        return

    # Function to preprocess the MOTO plate before entering the network
    def preprocess_moto(self, X:pd.Series) -> np.ndarray:
        """
            The MOTO image is (originally) of size <w:106, h:83>.
            The MOTO image is resized to <w:200, h:44> to be used in the network.
            The image will be cut in half horizontally to get 2 images 
            of size <106, 42> and <106, 41>, then both images will be scaled to <100, 44>
            and the two images will be concatenated to get the final image of size <200, 44>.
        """

        # Open the image from the array
        img = Image.fromarray(X[:-2].values.reshape(83, 106))

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
        
        # Return the image
        return img

    # Function to reverse the MOTO plate preprocessing (to be shown as the original image)
    def reverse_moto(self, X:np.ndarray) -> np.ndarray:
        """
            The MOTO image is (now) of size <w:200, h:44>.
            The MOTO image is resized to <w:106, h:83> to be shown as the original image.
            The image will be cut in half horizontally to get 2 images 
            of size <100, 44> each, then the images will be scaled to <106, 42> and <106, 41>
            and the two images will be concatenated vertically to get the final image of size <106, 83>.
        """

        # Open the image from the array
        img = Image.fromarray(X)

        # Cut the image
        img_1 = img.crop((0, 0, 100, 44))
        img_2 = img.crop((100, 0, 200, 44))

        # Resize the images
        img_1 = img_1.resize((106, 42), Image.ANTIALIAS)
        img_2 = img_2.resize((106, 41), Image.ANTIALIAS)

        # Concatenate the images
        img_1 = np.array(img_1)
        img_2 = np.array(img_2)
        img = np.concatenate((img_1, img_2), axis=0)
        
        # Return the image
        return img
