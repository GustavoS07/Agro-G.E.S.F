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
from ..models.cnn import CNN 


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
        
        print("✅ Modelo carregado e configurado com sucesso!")

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

                expected_input_size = 50176
                if classifier_input_size != expected_input_size:
                    print(f"Input size inesperado: {classifier_input_size}, esperado: {expected_input_size}")
                    
                    # Tenta calcular tamanho necessário
                    feature_h_w = classifier_input_size // 256
                    feature_size = int(feature_h_w ** 0.5)
                    original_size = feature_size * 16  # 4 pools de 2x2
                    
                    print(f"Sugestão: usar imagens {original_size}x{original_size}")
                    
                    # Ajusta o transform se necessário
                    if original_size != 224:
                        print(f"Ajustando transform para {original_size}x{original_size}")
                        self.transform = transforms.Compose([
                            transforms.Resize((original_size, original_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])
                        ])
            
            # Detecta o número de classes
            if 'classifier.4.weight' in state_dict:
                num_classes = state_dict['classifier.4.weight'].shape[0]
                print(f"Número de classes detectado: {num_classes}")
            else:
                num_classes = len(self.classes)
                print(f"Usando número padrão de classes: {num_classes}")
            
            #importando a estrutura do modelo
            model = CNN(num_classes)
            
            # Carregar state_dict
            print("Carregando state_dict...")
            model.load_state_dict(state_dict)
            
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
            # Carrega e preprocessar a imagem
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            print(f"Tensor shape: {input_tensor.shape}")
            
            # Faz a predição
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs[0], dim=0)
                
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()
                predicted_class = self.classes[predicted_idx]
                
                # Cria um dicionário com todas as probabilidades
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