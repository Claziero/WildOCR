import os
import time
import torch
import pandas as pd
from cnn import ConvNet
from sklearn.model_selection import train_test_split

# Define colors
TEXT_RESET = '\033[0m'
TEXT_RED = '\033[91m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'
TEXT_BLUE = '\033[94m'

# Class to drive the program
class Driver:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            print(TEXT_GREEN 
                + '>> CUDA is available ({}).'.format(torch.cuda.get_device_name())
                + TEXT_RESET)
            gpu = torch.device("cuda:0")
            cpu = torch.device("cpu")
        else:
            print(TEXT_RED + '>> CUDA is not available.' + TEXT_RESET)
            gpu = cpu = torch.device("cpu")

        self.X_train:pd.DataFrame = None
        self.Y_train:pd.DataFrame = None
        self.X_test:pd.DataFrame = None
        self.Y_test:pd.DataFrame = None

        self.net:ConvNet = ConvNet().to(gpu)
        self.net.gpu = gpu
        self.net.cpu = cpu

        self.save_model:bool = False
        self.save_model_path:str = None
        self.save_preds:bool = False
        self.save_preds_path:str = None
        self.model_loaded:bool = False
        return

    # Function to load the dataset (full) (not-used)
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
        elapsed = end_time - start_time

        print(TEXT_GREEN 
            + '>> Training finished in {:.0f} mins {:.2f} secs.'.format(elapsed / 60, elapsed % 60) 
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
        elapsed = end_time - start_time

        print(TEXT_GREEN 
            + '>> Testing finished in {:.0f} mins {:.2f} secs.'.format(elapsed / 60, elapsed % 60) 
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
    choice = 1

    while choice != '0':
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
                choice = 1
                continue

            # Load entire dataset
            elif choice == '1':
                train = input('Enter the filename for training dataset [Enter = \"dataset_train.csv\"]: ')
                if train == '':
                    train = 'dataset_train.csv'
                d.load_train(train)

                test = input('Enter the filename for test dataset [Enter = \"dataset_test.csv\"]: ')
                if test == '':
                    test = 'dataset_test.csv'
                d.load_test(test)

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
