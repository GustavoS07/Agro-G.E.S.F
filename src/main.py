from data.data import get_dataloaders
from models.cnn import CNN
from training.train import train_model
import torch
import torch.nn as nn
import torch.optim as optim
import os
data_dir = os.path.join(os.path.dirname(__file__),'..','data')
data_dir = os.path.abspath(data_dir)

def main():
    ## Parte por identificar o dispositivo sendo usado
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Definindo o tamanho do batch
    batch_size = 64
    print(f'Tamanho do batch: {batch_size}')
    #Definindo quantas épocas o modelo vai treinar
    num_epochs = 30
    print(f'Número de épocas: {num_epochs}')

    ##Carregando da função data.py os dataloaders, o tamanho dos datasets e o nome das classes
    try:
        dataloaders,dataset_sizes,class_names = get_dataloaders(data_dir, batch_size=batch_size)
        print(f'Classes:{class_names}')
        print(f'Dataset sizes:{dataset_sizes}')
    except Exception as e:
        print(f'Erro ao carregar os dataloaders: {e}')
        return
    
    ## Definindo o modelo com a estrutura da CNN dentor de cnn.py
    model = CNN(num_classes=len(class_names))
    print(f'Modelo: {model} com {len(class_names)} classes')
    ## Definindo a função correpsondente a cálculo de loss
    criterion = nn.CrossEntropyLoss()
    
    ## Definindo o optimizador
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    print("\n"+"-"*20)
    
    try:
        model,history = train_model(
            model,
            dataloaders,
            dataset_sizes,
            criterion,optimizer,
            device,
            num_epochs=num_epochs,
            monitor_plots=True,
            checkpoint_dir='checkpoint'
        )
        
        final_model_path = 'modelo_final.pth'
        torch.save(model.state_dict(),final_model_path)
        print(f'Modelo salvo em: {final_model_path}')
    except Exception as e:
        print(f'Erro ao treinar o modelo: {e}')
        return

    print("\n"+"-"*20)
    
if __name__ == "__main__":
    main()