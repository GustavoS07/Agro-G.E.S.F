## Código responsável por funções auxiliares
## Como:
##  SALVAR MODELO
## MONTITORAR MÉTRICAS

import torch
import time
import copy
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

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

## Função para monitorar o treinamento
class TrainingMonitor():
    
    def __init__(self,save_plots,plot_dir='training_plots'):
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.start_time = None
        
        #Criando o diretório para salvar os gráficos
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
    def plot_metrics(self, history, epoch):
        if not self.save_plots:
            return
        
        # Verificar se há dados suficientes para plotar
        if (len(history.get('train_loss', [])) == 0 or 
            len(history.get('val_loss', [])) == 0):
            print("Dados insuficientes para plotar. Pulando...")
            return
        
        # Verificar se train e val têm o mesmo tamanho
        train_len = len(history['train_loss'])
        val_len = len(history['val_loss'])
        
        if train_len != val_len:
            print(f"Aviso: Tamanhos diferentes - Train: {train_len}, Val: {val_len}")
            # Usar o menor tamanho para evitar erro
            min_len = min(train_len, val_len)
            if min_len == 0:
                return
        else:
            min_len = train_len
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plotando o gráfico de perda
        epochs_range = range(1, min_len + 1)
        ax1.plot(epochs_range, history['train_loss'][:min_len], 'bo-', 
                label='Training Loss', linewidth=2)
        ax1.plot(epochs_range, history['val_loss'][:min_len], 'ro-', 
                label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plotando o gráfico de acurácia
        ax2.plot(epochs_range, history['train_acc'][:min_len], 'bo-', 
                label='Training Accuracy', linewidth=2)
        ax2.plot(epochs_range, history['val_acc'][:min_len], 'ro-', 
                label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/training_progress_epoch_{epoch+1}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    #Função para formatar o tempo
    def format_time(self,seconds):
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    

    def log_epoch_summary(self, epoch, num_epochs, history, best_acc, epoch_time):
        print(f"\n{'='*50}")
        print(f"Resumo da epoca {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        print(f"Tempo de treinamento: {self.format_time(epoch_time)}")
        
        # Verificar se existem dados antes de acessar
        if len(history['train_loss']) > 0:
            print(f"Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"Train Acurácia: {history['train_acc'][-1]:.4f}")
        
        # Só mostrar validação se existir
        if len(history.get('val_loss', [])) > 0:
            print(f"Val Loss: {history['val_loss'][-1]:.4f}")
            print(f"Val Acurácia: {history['val_acc'][-1]:.4f}")
        
        print(f"Melhor acuracia: {best_acc:.4f}")
        
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            remaining_epochs = num_epochs - (epoch + 1)
            if epoch >= 0:
                avg_time_per_epoch = total_time / (epoch + 1)
                eta = avg_time_per_epoch * remaining_epochs
                print(f"Tempo total: {self.format_time(total_time)} | ETA: {self.format_time(eta)}")
        
        print(f"{'='*50}\n")
