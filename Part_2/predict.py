# Imports here
import torch
import numpy as np
import argparse, os, sys, functions_predict, json
from pathlib import Path

parser = argparse.ArgumentParser(description='Define the flower directory and prediction parameters.')

parser.add_argument('Path',
                    metavar='path',
                    type=str,
                    help='Manual typing of the flower directory chosen')

parser.add_argument('Checkpoint',
                    type=str,
                    help='Load the checkpoint',
                    default = 'save_models/checkpoint.pth')

parser.add_argument('--top_k',
                    type=str,
                    help='Define the top K classes',
                    default = '3',
                    required=False)

parser.add_argument('--category_names',
                    help='Map of category to real names',
                    default = 'cat_to_name.json',
                    required=False)

parser.add_argument('--gpu',
                    help='Define if use the GPU to train the model',
                    default = 'True',
                    required=False)

# Execute the parse_args() method
args = parser.parse_args()

image_path = args.Path
checkpoint = args.Checkpoint
topk = args.top_k
category_names = args.category_names
gpu_act = args.gpu

if not os.path.isfile(image_path):
    print('The data specified does not exist.')
    sys.exit()
                       
if not os.path.isfile(checkpoint):
    print('The checkpoint specified does not exist.')
    sys.exit()
  
model = functions_predict.load_checkpoint(checkpoint)

device = torch.device("cuda" if gpu_act == 'True' else "cpu")
model.to(device);
model.eval()

image = functions_predict.process_image(image_path)
image = torch.from_numpy(np.array([image])).float()

if gpu_act == 'True':
    image = image.cuda()

logps = model.forward(image)
ps = torch.exp(logps).data

probs = torch.topk(ps, int(topk))[0].tolist()[0] # probabilities
index = torch.topk(ps, int(topk))[1].tolist()[0] # index
    
ind = []
for i in range(len(model.class_to_idx.items())):
    ind.append(list(model.class_to_idx.items())[i][0])

classes = []
for i in range(int(topk)):
    classes.append(ind[index[i]])

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
image_path = os.path.basename(os.path.dirname(Path(image_path)))
original_category = cat_to_name[str(image_path)]
print('The original category of the flower is',original_category, '.')

categories = []

for i in range(len(classes)):
    categories.append(cat_to_name[classes[i]])

print(categories)
print(probs)


