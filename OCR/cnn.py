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

from sklearn.model_selection import train_test_split
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Define colors
TEXT_GREEN = '\033[92m'
TEXT_BLUE = '\033[94m'
TEXT_RED = '\033[91m'
TEXT_RESET = '\033[0m'

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
    def test_net(self, X_test:torch.Tensor, Y_test:torch.Tensor) -> None:
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

            print(TEXT_BLUE 
                + 'Accuracy: {:.2f}%'.format(100 * correct / (7 * len(X_test)))
                + TEXT_RESET)
            print(TEXT_BLUE 
                + 'Total: {}, Correct: {}'.format(7 * len(X_test), correct)
                + TEXT_RESET)
            
        print(TEXT_GREEN + '>>  Finished Testing' + TEXT_RESET)
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



# Driver function
def driver(dataset_path:str, load:str=None, save:str='model.plk', epochs:int=4, learning_rate:float=0.001) -> None:
    # Load the dataset
    print(TEXT_GREEN + '>> Loading dataset...' + TEXT_RESET)
    dataset = pd.read_csv(dataset_path, header=None)

    # Split and normalize the dataset
    X = dataset.iloc[:, :8800] / 255
    Y = dataset.iloc[:, 8801:]

    # Split the dataset into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(TEXT_GREEN + '>> Dataset loaded.' + TEXT_RESET)
    
    # Create a network object
    net = ConvNet()

    # Load the trained network model if required
    train = False
    if load is not None and os.path.exists(load):
        net.load_state_dict(torch.load(load))
        print(TEXT_GREEN 
            + '>> Model loaded successfully ({}).'.format(load)
            + TEXT_RESET)
    elif load is not None:
        print(TEXT_RED + '>> No model found. Training ...' + TEXT_RESET)
        train = True

    # Move the network to the gpu (if available)
    net.to(gpu)

    # Train the network if required
    if load is None or train:
        print(TEXT_GREEN 
            + '>> Training for {} epochs with learning rate = {} ...'.format(epochs, learning_rate)
            + TEXT_RESET)
        start_time = time.time()
        net.train_net(X_train, Y_train, epochs=epochs, learning_rate=learning_rate)
        end_time = time.time()
        print(TEXT_GREEN 
            + '>> Training finished in {} seconds.'.format(end_time - start_time) 
            + TEXT_RESET)

    # Save the trained network model if required
    if train and save is not None:
        print(TEXT_GREEN + '>> Saving model to {} ...'.format(save) + TEXT_RESET)
        net.save(save)

    # Test the network
    net.test_net(X_test, Y_test)
    return

if __name__ == '__main__':
    driver('dataset.csv', load='model.plk', save='model.plk') # To use pre-trained model
    # driver('dataset.csv', save='model.plk', epochs=10, learning_rate=0.001) # To train a new model
