# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
# Imports argparse python module
import argparse
# Main program function defined below
def main():

    args = arg_parser() 

    train_dir = args.dir + '/train'
    valid_dir = args.dir + '/valid'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(args.image_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(args.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    output_size =  len(train_datasets.classes)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets,batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    
    input_sizes = {'vgg13': 25088, 'alexnet': 9216, 'resnet18': 512, 'densenet161': 2208}
    
    model,input_size,classifier = model_choice (input_sizes,args.arch,output_size,args.hidden_layers)
    train( trainloader,validloader,model,args.gpu,args.arch,args.epoch,args.learning_rate)
    save_checkpoint(model,args.save_dir,train_datasets,classifier,args.hidden_layers,input_size,output_size,args.arch,args.epoch,args.learning_rate)

def arg_parser():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser(description="training model program ")
    
    parser = argparse.ArgumentParser()
    
    # Argument 1: that's a path to a folder
    parser.add_argument('--dir', type = str, default = 'flowers', action='store', dest="dir", help = 'path to the data directory') 
    parser.add_argument('--save_dir', type = str, default = 'checkpoint1.pth', help = 'path to the save checkpoint directory') 
    parser.add_argument('--arch', type = str, default = 'vgg13', help = 'type of architecture: vgg13 or alexnet or resnet18 or densenet 161 --make sure to specify features num.')
    parser.add_argument('--learning_rate', type = float, default = 0.001,help = 'learning rate')
    parser.add_argument('--epoch', type = int, default = 2, help = 'epochs number')
    parser.add_argument('--image_size', type = int, default = 224, help = ' image size to forward to model ex:224 for 224*224')
    #parser.add_argument('--input_size', type = int, default = 25088, help = 'features number')
    #parser.add_argument('--output_size', type = int, default = 102, help = 'classes number')
    parser.add_argument('--hidden_layers', nargs='+',type = int,default = [(4096)],help = "hidden layers' sizes list default:one layer 4096 units")
    parser.add_argument('--gpu', action = 'store_true',default = True, help = " use gpu ")
   
    # Assigns variable in_args to parse_args()
    args = parser.parse_args()    
    return args
def check_gpu(gpu):
   # If gpu_arg is false then simply return the cpu device
    if not gpu:
        return torch.device("cpu")
    else:
    # If gpu_arg then make sure to check for CUDA before assigning it
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    
      
def model_choice (input_sizes,arch,output_size,hidden_layers):
    if (arch == 'vgg13'):
        model = models.vgg13(pretrained = 'true')
        input_size = input_sizes[arch] 
        classifier = model_classifier(model,input_size,output_size,hidden_layers)
        model.classifier = classifier 
  
    elif(arch == 'alexnet'):
        model = models.alexnet(True)
        input_size = input_sizes[arch]
        classifier = model_classifier(model,input_size,output_size,hidden_layers)
        model.classifier = classifier 

    elif(arch == 'resnet18'):
        model = models.resnet18(True)
        input_size = input_sizes[arch]
        classifier = model_classifier(model,input_size,output_size,hidden_layers)
        model.fc = classifier
        
    elif(arch == 'densenet161'):
        model = models.densenet161(True)
        input_size = input_sizes[arch]
        classifier = model_classifier(model,input_size,output_size,hidden_layers)
        model.classifier = classifier 

    else :
        model = models.vgg13(pretrained = 'true')
        input_size = input_sizes['vgg13']
        classifier = model_classifier(model,input_size,output_size,hidden_layers)
        model.classifier = classifier 
    return model,input_size,classifier       
        
def model_classifier(model,input_size,output_size,hidden_layers):
   # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    classifier = Network(input_size ,output_size ,[r for r in hidden_layers ], drop_p=0.5)
  
    return classifier

    

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
       
        '''
        def input_size(self):
            return input_size
        def output_size(self):
            return output_size
        def hidden_layers(self):
            return hidden_layers
        '''
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

 




def train( trainloader,validloader,model,gpu,arch,epochs_no = 3,lr=0.001):

    steps = 0
    running_loss = 0
    print_every = 10
    device = check_gpu(gpu)
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    if (arch == 'resnet18'):
        optimizer = optim.Adam(model.fc.parameters(), lr)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.optimizer = optimizer
    model.to(device)
    
    for e in range(epochs_no):
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):

            steps += 1
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion,device)

                print("Epoch: {}/{}.. ".format(e+1, epochs_no),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "validation loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "validation accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()
                
                
def validation(model, validloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(validloader):
        # Move input and label tensors to the GPU

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model.forward(inputs)
      
        test_loss += criterion(outputs, labels).item()

    
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def save_checkpoint(model,save_dir,train_datasets,classifier,hidden_layers,input_size,output_size,arch ='vgg13',epochs_no=3,lr=0.001):
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers':hidden_layers,
                  'arch': arch,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'learning_rate' : lr,
                  #'optimizer_state_dict':model.optimizer.state_dict(), useless except for training again "takes 1 gb use of space" 
                  'epochs_no': epochs_no,
                  'classifier': classifier}
    torch.save(checkpoint, save_dir)
    print(model.class_to_idx)
if __name__ == "__main__":
    main()