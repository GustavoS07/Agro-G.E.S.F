from data.data import get_dataloaders
from models.cnn import CNN
from training.train import train_model
from utils.utils import save_checkpoint,calculate_accuracy
import torch
import torch.nn as nn
import torch.optim as optim
import os
data_dir = os.path.join(os.path.dirname(__file__),'..','data')
data_dir = os.path.abspath(data_dir)
def main():
    print("Aqui")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data'
    batch_size = 64
    num_epochs = 30
    
    dataloaders,dataset_sizes,class_names = get_dataloaders(data_dir, batch_size=batch_size)
    print(f'Classes:{class_names}')
    print(f'Dataset sizes:{dataset_sizes}')
    
    model = CNN(num_classes=len(class_names))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    
    model,history = train_model(model,dataloaders,dataset_sizes,criterion,optimizer,device,num_epochs=num_epochs)
    
    
if __name__ == "__main__":
    main()