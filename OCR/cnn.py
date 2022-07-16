"""
  Implementation of a Convolution Neural Network
  for Italian Number Plates recognition.

  > Input image is a grayscale image of a number plate
    of size <w:200, h:44, c:1> (CAR images) or <w:106, h:83, c:1> (MOTORCYCLE images)

  > Output is a vector of length 232, where each element is a probability of the
    corresponding digit to be a certain letter/number/class.
    Every character is an array of 32 ints identifiing the specific character
    There are 7 characters in total, hence 32 * 7 = 224
    The last 8 elements discriminate the plate type.
    Discriminating bits (assume 1 or 0) are disposed in the following order:
        - CAR plate
        - MOTORCYCLE plate
        - AERONAUTICA MILITARE plate
        - CARABINIERI plate
        - ESERCITO plate
        - MARINA MILITARE plate
        - VIGILI DEL FUOCO plate
        - AUTO SPECIALE plate

  > The network consists of a series of convolutional layers, followed by a
    series of fully connected layer.
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from math import ceil, sqrt
from datasetGenerator import reverse_moto

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

        # Network parameters
        self.last_bits_count = 8
        self.l1_out_ch = 7
        self.l2_out_ch = 15
        self.fc1_in_dim = self.l2_out_ch * 9 * 48
        self.fc1_out_dim = self.fc1_in_dim // 2
        self.fc2_out_dim = self.fc1_out_dim // 2
        self.fc3_out_dim = self.fc2_out_dim // 2
        self.fc4_out_dim = 32 * 7 + self.last_bits_count

        # Initial image size: <w:200, h:44, c:1>
        # Convolutional layer 1: <w:200, h:44, c:1> -> <w:99, h:21>
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.l1_out_ch, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Convolutional layer 2: <w:99, h:21> -> <w:48, h:9>
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.l1_out_ch, self.l2_out_ch, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Fully connected layer: output size = (32 * 7) + 8 = 232 neurons
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc1_in_dim, self.fc1_out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc1_out_dim, self.fc2_out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc2_out_dim, self.fc3_out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc3_out_dim, self.fc4_out_dim)
        )
        return

    # Forward function: generate a prediction given the input "x"
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, self.fc1_in_dim)
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
            sol_index = np.argmax(Y_test[32 * i:32 * (i + 1)])

            # If last 2 characters are null, don't count them as a failure
            # because the network doesn't need to predict them
            if i > 4 and sol_index == 0 and Y_test[32 * i] == 0:
                correct += 1
            else:
                out_index = np.argmax(net_output[32 * i:32 * (i + 1)])
                if out_index == sol_index:
                    correct += 1

        # Check the last bits discriminating the plates
        out_ptype = np.argmax(net_output[-self.last_bits_count:])
        sol_ptype = np.argmax(Y_test[-self.last_bits_count:])
        if out_ptype == sol_ptype:
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
    def output_to_string(self, net_output:torch.Tensor) -> tuple[str, str]:
        # Convert the output to a string
        out_string = ''

        # Calculate plate type from prediction
        ptype = np.argmax(net_output[-8:])
        for i in range(7):
            # If plate type has only 5 characters, break the loop
            if i > 4 and ptype != 0 and ptype != 1 and ptype != 7:
                break

            out_index = np.argmax(net_output[32 * i:32 * (i + 1)])
            out_string += chr(out_index + 65 + self.calculate_gap(out_index))

        return out_string, self.plate_type_to_string(ptype)

    # Function to convert the plate type into a string
    def plate_type_to_string(self, ptype:int) -> str:
        if ptype == 0:
            return 'auto'
        if ptype == 1:
            return 'moto'
        if ptype == 2:
            return 'aeronautica'
        if ptype == 3:
            return 'carabinieri'
        if ptype == 4:
            return 'esercito'
        if ptype == 5:
            return 'marina'
        if ptype == 6:
            return 'vigfuoco'
        if ptype == 7:
            return 'autosp'
        
        return None
        
    # Function to save the predictions in string format
    def save_predictions(self, X_test:np.ndarray, Y_pred:torch.Tensor, Y_test:np.ndarray, filename:str) -> None:
        f = open(filename, 'a+')
        
        # Convert the output to a string
        out_string, out_ptype = self.output_to_string(Y_pred)
        test_string, sol_ptype = self.output_to_string(Y_test)
        X_test = str(np.array(X_test).flatten().tolist())[1:-1]

        # Write the output to the file
        f.write(X_test + ',' + out_string + ',' + out_ptype 
            + ',' + test_string + ',' + sol_ptype +'\n')
        f.close()
        return

    # Function to show "num" images with their predictions
    def show_predictions(self, num:int, preds:str='preds.csv') -> None:
        # Read the predictions file and get "num" random images
        lines = pd.read_csv(preds, header=None)
        lines = lines.sample(n=num)

        # Get the images and predictions
        img = lines.iloc[:, :-4]
        Y_pred = lines.iloc[:, -4]
        ptype_pred = lines.iloc[:, -3]
        Y_test = lines.iloc[:, -2]
        ptype_test = lines.iloc[:, -1]

        # Plot
        rows = ceil(sqrt(num))
        fig = plt.figure(figsize=(rows*2, rows), constrained_layout=True)
        for i in range(num):
            # Get the image, prediction and test string
            # If the predicted plate type is 'moto' or 'autosp, then it has a different shape
            if ptype_pred.iloc[i] == 'moto' or ptype_pred.iloc[i] == 'autosp':
                img_array = reverse_moto(np.array(img.iloc[i]))
            else:
                img_array = np.array(img.iloc[i]).reshape(44, 200)
            
            prediction = Y_pred.iloc[i]
            ptype = ptype_pred.iloc[i]

            if ptype == ptype_test.iloc[i] and prediction == Y_test.iloc[i]:
                color = 'green'
            else:
                color = 'red'
            
            # Plot the image
            fig.add_subplot(rows, rows, i + 1)
            plt.imshow(img_array, cmap='gray')
            plt.axis('off')
            plt.title(prediction + '\n' + ptype, color=color)

        plt.show()
        return

    # Function to show the loss function
    def show_loss(self) -> None:
        # Plot the train loss function
        plot = plt.figure(figsize=(15, 5))
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

        plot.subplots_adjust(wspace=0.5)
        plt.savefig('graphs.png')
        return
