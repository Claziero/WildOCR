"""
  Implementation of a Convolution Neural Network
  for Italian Number Plates recognition.

  > Input image is a grayscale image of a character of size <w:20, h:40, c:1>

  > Output is a vector of length 32, where each element is a probability of the
    corresponding digit to be a certain letter/number.

  > The network consists of a series of convolutional layers, followed by a
    series of fully connected layer.
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from math import ceil, sqrt

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
        self.l1_out_ch = 2
        self.l2_out_ch = 4
        self.fc1_in_dim = self.l2_out_ch * 3 * 8
        self.fc1_out_dim = self.fc1_in_dim // 2
        self.fc2_out_dim = 32

        # Initial image size: <w:20, h:40, c:1>
        # Convolutional layer 1: <w:20, h:40> -> <w:9, h:19>
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.l1_out_ch, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Convolutional layer 2: <w:9, h:19> -> <w:3, h:8>
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.l1_out_ch, self.l2_out_ch, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Fully connected layer: output size = 32 neurons
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc1_in_dim, self.fc1_out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc1_out_dim, self.fc2_out_dim)
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
                data = torch.from_numpy(data.reshape(1, 1, 40, 20)).float().to(self.gpu)
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
            acc = self.validate_net(X_valid, Y_valid)

            # Stop the training if accuracy reaches 99%
            if acc >= 99.5:
                # Save the model as checkpoint
                self.save('ckpt-{}.pkl'.format(epoch+1))
                break

            # Save the model as checkpoint
            self.save('ckpt-{}.pkl'.format(epoch+1))

        print('Finished Training')
        self.show_loss()
        return

    # Function to validate the network
    def validate_net(self, X_valid:pd.DataFrame, Y_valid:pd.DataFrame) -> float:
        # Validate the network
        self.eval()
        with torch.no_grad():
            correct = loss = 0
            for i, data in enumerate(X_valid.values):
                # Convert the data to torch tensor
                data = torch.from_numpy(data.reshape(1, 1, 40, 20)).float().to(self.gpu)
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

            accuracy = 100 * correct / len(X_valid)
            self.valid_accuracy_array.append(accuracy)

        print(TEXT_BLUE + 'Accuracy of the network on the validation set: {:.2f}%'.format(accuracy) + TEXT_RESET)
        return accuracy

    # Function to test the network
    def test_net(self, X_test:pd.DataFrame, Y_test:pd.DataFrame, save_preds:str=None) -> float:
        # Test the network
        self.eval()
        with torch.no_grad():
            correct = 0
            for i, data in enumerate(X_test.values):
                # Convert the data to torch tensor
                data = torch.from_numpy(data.reshape(1, 1, 40, 20)).float().to(self.gpu)
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

            accuracy = 100 * correct / len(X_test)
            print(TEXT_BLUE 
                + 'Accuracy: {:.2f}%'.format(accuracy)
                + TEXT_RESET)
            print(TEXT_BLUE 
                + 'Total: {}, Correct: {}'.format(len(X_test), correct)
                + TEXT_RESET)
        return accuracy

    # Function to save the network
    def save(self, filename:str) -> None:
        torch.save(self.state_dict(), filename)
        print(TEXT_GREEN + 'Model saved to {}'.format(filename) + TEXT_RESET)
        return

    # Function to check the results of the network
    def check_results(self, net_output:torch.Tensor, Y_test:np.ndarray) -> int:
        # Check the last bits discriminating the plates
        out = np.argmax(net_output)
        sol = np.argmax(Y_test)
        if out == sol: return 1
        else: return 0

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
    def output_to_string(self, net_output:torch.Tensor) -> tuple[str, float]:
        pred = np.argmax(net_output)
        return chr(pred + 65 + self.calculate_gap(pred)), net_output[pred]

    # Function to save the predictions in string format
    def save_predictions(self, X_test:np.ndarray, Y_pred:torch.Tensor, Y_test:np.ndarray, filename:str) -> None:
        f = open(filename, 'a+')
        
        # Convert the output to a string
        out_string = self.output_to_string(Y_pred)[0]
        test_string = self.output_to_string(Y_test)[0]
        X_test = str(np.array(X_test).flatten().tolist())[1:-1].replace(' ', '').strip()

        # Write the output to the file
        f.write(X_test + ',' + out_string + ',' + test_string + '\n')
        f.close()
        return

    # Function to show "num" images with their predictions
    def show_predictions(self, num:int, preds:str='preds.csv') -> None:
        # Read the predictions file and get "num" random images
        lines = pd.read_csv(preds, header=None)
        lines = lines.sample(n=num)

        # Get the images and predictions
        img = lines.iloc[:, :-2]
        Y_pred = lines.iloc[:, -2]
        Y_test = lines.iloc[:, -1]

        # Plot
        rows = ceil(sqrt(num))
        fig = plt.figure(figsize=(rows*2, rows), constrained_layout=True)
        for i in range(num):
            # Get the image, prediction and test string            
            img_array = np.array(img.iloc[i]).reshape(40, 20)
            
            prediction = Y_pred.iloc[i]

            if prediction == Y_test.iloc[i]: color = 'green'
            else: color = 'red'
            
            # Plot the image
            fig.add_subplot(rows, rows, i + 1)
            plt.imshow(img_array, cmap='gray')
            plt.axis('off')
            plt.title(prediction, color=color)

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
