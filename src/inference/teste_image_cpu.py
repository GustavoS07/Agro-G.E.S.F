#!/usr/bin/env python3
"""
Versão com arquitetura EXATA baseada na análise do state_dict
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import traceback


class ExactCNN(nn.Module):
    """
    Arquitetura exata baseada no state_dict analisado:
    
    Features:
    - features.0: Conv2d(3, 32, 3)  
    - features.1: BatchNorm2d(32)
    - features.4: Conv2d(32, 64, 3)
    - features.5: BatchNorm2d(64) 
    - features.8: Conv2d(64, 128, 3)
    - features.9: BatchNorm2d(128)
    - features.12: Conv2d(128, 256, 3)
    - features.13: BatchNorm2d(256)
    
    Classifier:
    - classifier.1: Linear(50176, 512)
    - classifier.4: Linear(512, 6)
    """
    
    def __init__(self, num_classes=6):
        super(ExactCNN, self).__init__()
        
        # Features exatas conforme o state_dict
        self.features = nn.Sequential(
            # Bloco 1 (índices 0-3)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),     # features.0
            nn.BatchNorm2d(32),                             # features.1
            nn.ReLU(inplace=True),                          # features.2
            nn.MaxPool2d(2, 2),                             # features.3
            
            # Bloco 2 (índices 4-7)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # features.4
            nn.BatchNorm2d(64),                             # features.5
            nn.ReLU(inplace=True),                          # features.6
            nn.MaxPool2d(2, 2),                             # features.7
            
            # Bloco 3 (índices 8-11)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # features.8
            nn.BatchNorm2d(128),                            # features.9
            nn.ReLU(inplace=True),                          # features.10
            nn.MaxPool2d(2, 2),                             # features.11
            
            # Bloco 4 (índices 12-15)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # features.12
            nn.BatchNorm2d(256),                            # features.13
            nn.ReLU(inplace=True),                          # features.14
            nn.MaxPool2d(2, 2),                             # features.15
        )
        
        # Classifier exato conforme o state_dict
        # Input size para Linear: 50176 = 256 * 14 * 14 (assumindo input 224x224)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                                # classifier.0
            nn.Linear(50176, 512),                          # classifier.1
            nn.ReLU(inplace=True),                          # classifier.2
            nn.Dropout(0.5),                                # classifier.3
            nn.Linear(512, num_classes)                     # classifier.4
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PlantDiseasePredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.classes = [
            'Diabrotica_spreciosa',
            'Fungo', 
            'Lagarta',
            'Pinta_Preta',
            'Requeima',
            'Saudavel'
        ]
        
        # Transform para preprocessamento das imagens
        # IMPORTANTE: Usar mesmo tamanho que foi usado no treinamento
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Assumindo 224x224 baseado na dimensão Linear
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Inicializando preditor com device: {self.device}")
        print(f"Classes: {self.classes}")
        
        # Carregar modelo
        self.model = self._load_exact_model(model_path)
        
        if self.model is None:
            raise Exception("Falha ao carregar o modelo")
        
        print("Modelo carregado e configurado com sucesso!")

    def _load_exact_model(self, model_path):
        """Carrega modelo com arquitetura exata"""
        print(f"Carregando modelo de: {model_path}")
        
        try:
            # Carregar checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"Checkpoint carregado. Tipo: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                print(f"Chaves disponíveis: {list(checkpoint.keys())}")
                
                # Extrair state_dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("Usando 'model_state_dict'")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print("Usando 'state_dict'")
                else:
                    state_dict = checkpoint
                    print("Usando checkpoint completo como state_dict")
            else:
                print("Checkpoint não é um dicionário")
                return None
            
            # Verificar dimensões do classifier para confirmar input size
            if 'classifier.1.weight' in state_dict:
                classifier_input_size = state_dict['classifier.1.weight'].shape[1]
                print(f"Input size do classifier: {classifier_input_size}")
                
                # Calcular qual seria o tamanho da imagem necessário
                # classifier input = channels * height * width após features
                # 50176 = 256 * h * w
                # Assumindo 4 MaxPool2d (2x2), redução total = 16x
                # Se input original for 224x224, após features será 14x14
                # 256 * 14 * 14 = 50176 ✓
                
                expected_input_size = 50176
                if classifier_input_size != expected_input_size:
                    print(f"Input size inesperado: {classifier_input_size}, esperado: {expected_input_size}")
                    
                    # Tentar calcular tamanho necessário
                    feature_h_w = classifier_input_size // 256
                    feature_size = int(feature_h_w ** 0.5)
                    original_size = feature_size * 16  # 4 pools de 2x2
                    
                    print(f"💡 Sugestão: usar imagens {original_size}x{original_size}")
                    
                    # Ajustar transform se necessário
                    if original_size != 224:
                        print(f"Ajustando transform para {original_size}x{original_size}")
                        self.transform = transforms.Compose([
                            transforms.Resize((original_size, original_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])
                        ])
            
            # Detectar número de classes
            if 'classifier.4.weight' in state_dict:
                num_classes = state_dict['classifier.4.weight'].shape[0]
                print(f"Número de classes detectado: {num_classes}")
            else:
                num_classes = len(self.classes)
                print(f"Usando número padrão de classes: {num_classes}")
            
            # Criar modelo com arquitetura exata
            print("riando modelo com arquitetura exata...")
            model = ExactCNN(num_classes)
            
            # Carregar state_dict
            print("Carregando state_dict...")
            model.load_state_dict(state_dict, strict=True)
            
            # Configurar para inferência
            model.to(self.device)
            model.eval()
            
            print("Modelo carregado com arquitetura exata!")
            return model
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            traceback.print_exc()
            return None

    def predict_single_image(self, image_path):
        """Faz predição para uma única imagem"""
        if self.model is None:
            print("Modelo não carregado!")
            return None
        
        try:
            # Carregar e preprocessar a imagem
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            print(f"Tensor shape: {input_tensor.shape}")
            
            # Fazer predição
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs[0], dim=0)
                
                # Obter predição
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()
                predicted_class = self.classes[predicted_idx]
                
                # Criar dicionário com todas as probabilidades
                all_probabilities = {}
                for i, class_name in enumerate(self.classes):
                    all_probabilities[class_name] = probabilities[i].item()
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'predicted_idx': predicted_idx,
                    'all_probabilities': all_probabilities
                }
        
        except Exception as e:
            print(f"Erro na predição: {e}")
            traceback.print_exc()
            return None

    def get_model_info(self):
        """Retorna informações sobre o modelo"""
        if self.model is None:
            return "Modelo não carregado"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': str(self.model),
            'device': str(self.device),
            'classes': self.classes
        }