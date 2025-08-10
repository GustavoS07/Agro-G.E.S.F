from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_transforms():
    data_transforms = {
        'train': transforms.Compose ([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            
        ]),
        
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            
        ]),
    }
    return data_transforms

def get_dataloaders(data_dir,batch_size=32,num_workes=4):
    data_transforms = get_data_transforms()
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                  for x in ['train','val']}
    
    dataloaders = {x:DataLoader(image_datasets[x],batch_size=32,shuffle=True,num_workers=4)
                    for x in ['train','val']}
    
    datasets_size = {x:len(image_datasets[x]) for x in ['train','val']}

    class_names = image_datasets['train'].classes

    return dataloaders,datasets_size,class_names