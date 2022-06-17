# Imports here
import matplotlib.pyplot as plt
import argparse
import json
from PIL import Image
import torch
import numpy as np
import seaborn as sns
from train import check_gpu
from torchvision import models
from train import Network

def main():

    args = arg_parser()

    model,model.class_to_idx = load_checkpoint(args.save_dir)
    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    device = check_gpu(args.gpu)

    predict (args.image_path,model,cat_to_name,device,args.topk)
    
   
def arg_parser():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser(description="prediction of a certain image")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', action = 'store_true',default = False, help = " use gpu ")
    parser.add_argument('--cat_to_name', type = str, default = 'cat_to_name.json', help = 'path to the label-to-name json file directory') 
    parser.add_argument('--save_dir', type = str, default = 'checkpoint1.pth', help = 'path to the checkpoint directory')
    parser.add_argument('--image_path', type = str, default ='flowers/test/14/image_06052.jpg', help = 'path to the image to be predicted') 
    parser.add_argument('--topk', type = int, default = 5, help = 'number of top predictions ') 

    args = parser.parse_args() 
    return args
    
    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if (checkpoint['arch'] =='vgg13'):
        model = models.vgg13(True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif (checkpoint['arch'] =='alexnet'):
        model = models.alexnet(True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']
    elif (checkpoint['arch'] =='resnet18'):
        model = models.resnet18(True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['classifier']
    elif (checkpoint['arch'] =='densenet161'):
        model = models.densenet161(True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']
    else:
        model = models.vgg13(pretrained = 'true')
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

        

    model.load_state_dict(checkpoint['state_dict'])
    idx = checkpoint['class_to_idx']

    return model,idx


def process_image(image,device):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    
    test_image = Image.open(image)
    # Get original dimensions
    height, width = test_image.size
    
    new_height =height
    new_width = width
    
    if (new_height > new_width):
        new_width = 256
        new_height = (new_width*new_height)
        resize = [new_height,new_width]
    else:
        new_height = 256
        new_width = (new_width*new_height)
        resize = [new_height,new_width]
    test_image.thumbnail(resize)
    #print(test_image.size)  
    center = width/4, height/4
    left, top, right, bottom = center[0]-112, center[1]-112, center[0]+112, center[1]+112
    test_image = test_image.crop((left, top, right, bottom))
    
    np_image = np.array(test_image)/255 # Divided by 255 because the model expected floats 0-1

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    img = np_image.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().to(device)
    #print(img.shape)

    return img



def predict(image_path, model, cat_to_name,device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    img = process_image(image_path,device)
    img.unsqueeze_(0) 
    model=model.to(device)
    with torch.no_grad():
        output = model.forward(img)
        ps = torch.exp(output) # linear scale probs
    # Find the top 5 results probabilities and folder indecies
        top_probs, top_idx = ps.topk(topk)
    
    #from tensor object to numpy array
    top_probs = top_probs[0]
    top_idx =top_idx[0]
    top_probs = np.array(top_probs)
    top_idx = np.array(top_idx)
    #'''
    
    #print('top probs : ',top_probs, " idx: ",top_idx)
    class_to_idx = model.class_to_idx

    idx_to_class = {idx : cl for cl , idx in class_to_idx.items() }# inverses the dictionary key and value
    top_labels = [idx_to_class[idx] for idx in top_idx]

    flowers = [cat_to_name[label] for label in top_labels]

    predictions = {order+1 : [top_probs[order],top_labels[order],flowers[order] ] for order in range(topk)}
    print (predictions)

if __name__ == '__main__': main()
