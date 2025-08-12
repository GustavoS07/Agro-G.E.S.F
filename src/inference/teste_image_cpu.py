#!/usr/bin/env python3
"""
Teste interativo do modelo - SELECIONE SUA IMAGEM
"""

import torch
import sys
import os
from pathlib import Path

# Forçar CPU mesmo se GPU disponível
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Adicionar src ao path
sys.path.append('..')

from test_model import PlantDiseasePredictor

def list_images_in_directory(directory):
    """Lista todas as imagens em um diretório"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = []
    
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.lower().endswith(image_extensions):
                images.append(file)
    
    return sorted(images)

def show_menu_classes():
    """Mostra menu de classes disponíveis"""
    val_dirs = [
        '../../data/val/Saudavel',
        '../../data/val/Fungo', 
        '../../data/val/Lagarta',
        '../../data/val/Pinta_Preta',
        '../../data/val/Requeima',
        '../../data/val/Diabrotica_spreciosa'
    ]
    
    print("\n CLASSES DISPONÍVEIS:")
    available_classes = []
    
    for i, val_dir in enumerate(val_dirs, 1):
        if os.path.exists(val_dir):
            class_name = os.path.basename(val_dir)
            image_count = len(list_images_in_directory(val_dir))
            if image_count > 0:
                print(f"  {i}. {class_name} ({image_count} imagens)")
                available_classes.append((i, val_dir, class_name))
    
    return available_classes

def show_menu_images(selected_dir, class_name):
    """Mostra menu de imagens na classe selecionada"""
    images = list_images_in_directory(selected_dir)
    
    if not images:
        print(f" Nenhuma imagem encontrada em {selected_dir}")
        return None
    
    print(f"\n IMAGENS DISPONÍVEIS EM '{class_name}':")
    for i, image in enumerate(images, 1):
        print(f"  {i}. {image}")
    
    return images

def get_user_choice(prompt, max_option):
    """Pega escolha do usuário com validação"""
    while True:
        try:
            choice = input(prompt)
            if choice.lower() in ['q', 'quit', 'sair']:
                return None
            
            choice = int(choice)
            if 1 <= choice <= max_option:
                return choice
            else:
                print(f" Por favor, digite um número entre 1 e {max_option}")
        except ValueError:
            print("❌Por favor, digite um número válido")

def main():
    print(" Forçando uso de CPU para evitar problemas com ROCm...")
    print(" TESTE INTERATIVO DO MODELO DE DOENÇAS DE PLANTAS")
    print("=" * 60)
    
    # Inicializar preditor
    try:
        predictor = PlantDiseasePredictor('../../checkpoint.pth', device='cpu')
        print(" Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
    
    while True:
        # Mostrar menu de classes
        available_classes = show_menu_classes()
        
        if not available_classes:
            print(" Nenhuma classe com imagens encontrada!")
            break
        
        # Usuário seleciona classe
        print(f"\n Escolha uma classe (1-{len(available_classes)}) ou 'q' para sair:")
        class_choice = get_user_choice("Sua escolha: ", len(available_classes))
        
        if class_choice is None:
            print(" Saindo...")
            break
        
        # Pegar diretório e nome da classe selecionada
        selected_class = available_classes[class_choice - 1]
        _, selected_dir, class_name = selected_class
        
        # Mostrar imagens disponíveis
        images = show_menu_images(selected_dir, class_name)
        if not images:
            continue
        
        # Usuário seleciona imagem
        print(f"\n  Escolha uma imagem (1-{len(images)}) ou 'q' para voltar:")
        image_choice = get_user_choice("Sua escolha: ", len(images))
        
        if image_choice is None:
            continue
        
        # Fazer predição
        selected_image = images[image_choice - 1]
        image_path = os.path.join(selected_dir, selected_image)
        
        print(f"\n Testando: {selected_image}")
        print(f"  Classe real: {class_name}")
        print(" Processando...")
        
        result = predictor.predict_single_image(image_path)
        
        if result:
            print(f"\n Predição: {result['predicted_class']}")
            print(f" Confiança: {result['confidence']:.3f}")
            
            # Verificar se acertou
            correct = " Correto" if result['predicted_class'] == class_name else "❌ ERRADO"
            print(f"   {correct}")
            
            print("\n Top 3 probabilidades:")
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            
            for i, (classe, prob) in enumerate(sorted_probs, 1):
                print(f"   {i}. {classe}: {prob:.3f}")
        else:
            print(" Erro ao processar a imagem")
        
        # Perguntar se quer testar outra
        print("\n" + "="*50)
        continue_choice = input(" Testar outra imagem? (s/n): ").lower()
        if continue_choice not in ['s', 'sim', 'y', 'yes', '']:
            print(" Encerrando testes!")
            break

if __name__ == "__main__":
    main()