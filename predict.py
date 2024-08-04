import torch
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
from torch import nn
from torch import optim
#other useful imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path",help="img path")
    parser.add_argument("check",help="checkpoint",default="./home/workspace/checkpoint2.pth")
    
    args = parser.parse_args() 
    
    img_path = args.image_path
    check_path = args.check

#load model
model = models.vgg16(pretrained=True)
for p in model.parameters():
    p.require_grade = False
        
classifier = nn.Sequential( 
    nn.Linear(25088,5000),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(5000,1000),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(1000,102),
    nn.ReLU(),
    nn.LogSoftmax(dim=1))

model.classifier = classifier

state_dict = torch.load(check_path)
model.load_state_dict(state_dict)

def process_image(image):
    #resize
    if image.width > image.height:
        ratio = float(image.width) / float(image.height)
        newheight = ratio * 256
        image = image.resize((256, int(float(newheight))))
        
    else:
        ratio = float(image.height) / float(image.width)
        newwidth = ratio * 256
        image = image.resize((int(float(newwidth)), 256))
    
    #center crop
    left = (image.width - 224)/2
    top = (image.height - 224)/2
    right = (image.width +224)/2
    bottom = (image.height + 224)/2
    
    image = image.crop((left,top,right,bottom))

    #convert image to tensor and normalize it 
    image = F.to_tensor(image)
    image = F.normalize(image,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    return image
def predict(image_path, model, topk=5):
    # read image and process it
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze(0)
    # prepare model for evaluation
    model.cpu()
    
    model.eval()

    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, 5)
        top_prob = top_prob.exp() #calculate exponestial
    
    top_prob = top_prob.numpy()[0]
    top_labels = top_labels.numpy()[0]
    idx_to_class = {model.class_to_idx[key]: key for key in model.class_to_idx}

    #map labels 
    classes = []
    for l in top_labels:
      classes.append(l)
    
    return top_prob,classes

#image prediction + sanity checking
image = Image.open(img_path)
top_prob, top_classes = predict(image, model)

image = plt.imread(image)

label = top_classes[0]

plt.figure(figsize=(6,6))

cat_labels = []
for class_idx in top_classes:
    cat_labels.append(cat_to_name[str(class_idx)])
    
ax = plt.subplot(3,3,1)
plt.title(f'{cat_to_name[str(label)]}')
plt.imshow(image)




ax = plt.subplot(3,3,4)
ax.set_yticks(np.arange(5))
ax.set_yticklabels(cat_labels)
ax.set_xlabel('Probability')
ax.invert_yaxis()
ax.barh(np.arange(5), top_prob, xerr=0, align='center', color='blue')

plt.show()