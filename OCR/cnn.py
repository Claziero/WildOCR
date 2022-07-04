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

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
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
                data = torch.from_numpy(data.reshape(1, 1, 44, 200)).float().to(self.gpu)
                # Forward pass
                output = self.forward(data).to(self.gpu)
                # Calculate the loss
                loss = criterion(output[0], torch.from_numpy(Y_train.values[i]).float().to(self.gpu))
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
        lines = StringIO(lines)
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
