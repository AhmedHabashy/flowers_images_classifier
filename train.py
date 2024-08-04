#torch imports 
import torch
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
from torch import nn
from torch import optim
#other useful imports
import argparse

#take data_directory as argument 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",help="data directory")
    parser.add_argument('--save_dir', dest="save_dir", default="./checkpoint.pth")
    parser.add_argument("--arch",help="architecture type (vgg16 or vgg13)",default="vgg16")
    parser.add_argument("--epochs",help="number of epochs",type=int,default=5)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", default=120)
    parser.add_argument('--learning_rate',default=0.001)
    parser.add_argument('--gpu', default="gpu")
    
    
    args = parser.parse_args()  
    
    flowers_directory = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    epoch = args.epochs
    lr = args.learning_rate
    hu = args.hidden_units
    gpu_state = args.gpu
    
    print("data directory = ",flowers_directory) # /home/workspace/ImageClassifier/flowers
    print("choosen arch = ",arch)

#directories    
train_dir = flowers_directory + '/train'
valid_dir = flowers_directory + '/valid'
test_dir = flowers_directory + '/test'

#applying transformation to the data 
# TODO: Define your transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomGrayscale(),
                                      transforms.RandomCrop((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

valid_test_tranform = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.RandomCrop((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_set = datasets.ImageFolder(root=train_dir,transform=train_transform)
valid_set = datasets.ImageFolder(root=valid_dir,transform=valid_test_tranform)
test_set = datasets.ImageFolder(root=test_dir,transform=valid_test_tranform)
# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32  ,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32  ,shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=64  ,shuffle=True)



def train(model,epoch):
    epochs = epoch
    model.cuda()
    for e in range(epochs):
        run_loss = 0
        for images,labels in train_loader:
            #reset grad history
            opt.zero_grad()
            #calculate prediction
            output = model.forward(images.cuda())
    #             output = model.forward(images)
            #calculate loss
            loss = loss_function(output,labels.cuda())
    #             loss = loss_function(output,labels)
            #calculate grads
            loss.backward()
            #propagate loss
            opt.step()
            #save loss
            run_loss += loss.item()
        model.eval()
        v_loss = 0
        v_accuracy=0
        for ii, (inputs2,labels2) in enumerate(valid_loader):
            opt.zero_grad()
            inputs2, labels2 = inputs2.cuda() , labels2.cuda()
            with torch.no_grad():    
                outputs = model.forward(inputs2)
                v_loss = loss_function(outputs,labels2)
                ps = torch.exp(outputs).data
                equality = (labels2.data == ps.max(1)[1])
                v_accuracy += equality.type_as(torch.FloatTensor()).mean()

                v_loss = v_loss / len(valid_loader)
                v_accuracy = v_accuracy /len(valid_loader)
        
        else:
            model.train()
            print(f'epochs:{e+1} train_loss:{run_loss/len(train_loader)} valid_loss:{v_loss} valid_accuracy:{v_accuracy}')
    return model
    
  

if arch == "vgg16":
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
    loss_function = nn.NLLLoss()
    opt = optim.Adam(model.classifier.parameters(),lr=lr) 
    model = train(model,epoch)
    #save model
    torch.save(model.state_dict(),save_dir)
    model.class_to_idx = train_set.class_to_idx
    state_dict = torch.load(save_dir)
    model.load_state_dict(state_dict)
 
elif arch == "vgg13":
    model = models.vgg13(pretrained=True)
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
    loss_function = nn.NLLLoss()
    opt = optim.Adam(model.classifier.parameters(),lr=lr) 
    model = train(model,epoch)
    
    #save model
    torch.save(model.state_dict(),save_dir)
    model.class_to_idx = train_set.class_to_idx
    state_dict = torch.load(save_dir)
    model.load_state_dict(state_dict)

#network testing
correct_class= 0
total_class= 0
model.cuda()
with torch.no_grad():
    for images,labels in test_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total_class += labels.size(0)
        correct_class += (predicted == labels).sum().item()
print(f"Accuracy : {correct_class*100/total_class}%")