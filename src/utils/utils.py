## Código responsável por funções auxiliares
## Como:
##  SALVAR MODELO
## MONTITORAR MÉTRICAS

import torch

def save_checkpoint(model,optimizer,epoch,path='checkpoint.pth'):
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        
    },path)
    
def calculate_accuracy(outputs,labels):
    _, preds = torch.max(outputs,1)
    correct = torch.sum(preds == labels).item()
    return correct /labels.size(0)

