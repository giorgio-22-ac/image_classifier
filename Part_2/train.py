import argparse, os, sys, functions_train


parser = argparse.ArgumentParser(description='Define the data directory and model parameters.')

parser.add_argument('Path',
                    metavar='path',
                    type=str,
                    help='Manual typing of the data directory chosen')

parser.add_argument('--save_dir',
                    type=str,
                    help='Save the checkpoint',
                    default = 'save_models',
                    required=False)

parser.add_argument('--arch',
                    type=str,
                    help='Define the architecture of the model',
                    default = 'vgg19',
                    required=False)

parser.add_argument('--learning_rate',
                    help='Define the learning rate of the model',
                    default = '0.0002',
                    required=False)

parser.add_argument('--hidden_units',
                    help='Define the hidden units of the model',
                    default = '1024',
                    required=False)

parser.add_argument('--epochs',
                    help='Define the epochs of the model',
                    default = '3',
                    required=False)

parser.add_argument('--gpu',
                    help='Define if use the GPU to train the model',
                    default = 'True',
                    required=False)

# Execute the parse_args() method
args = parser.parse_args()

data_dir = args.Path
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

if not os.path.isdir(data_dir):
    print('The path of the data specified does not exist.')
    sys.exit()

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
else:
    print('The directory to save the model already exists.')
    

trainloader, validloader, testloader, train_data = functions_train.load_data(data_dir)    
    
functions_train.model_train_and_test(save_dir,arch,learning_rate,hidden_units,epochs,gpu,trainloader,validloader,testloader,train_data)

