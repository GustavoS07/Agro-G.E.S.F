#!/usr/bin/env python3


import torch
import sys
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from test_model import PlantDiseasePredictor

def list_images_in_directory(directory):
    #Lista todas as imagens em um diretório
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = []
    
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.lower().endswith(image_extensions):
                images.append(file)
    
    return sorted(images)

def show_menu_classes():
    #Mostra o menu de classes disponíveis
    val_dirs = [
        '../../data/val/Saudavel',
        '../../data/val/Fungo', 
        '../../data/val/Lagarta',
        '../../data/val/Pinta_Preta',
        '../../data/val/Requeima',
        '../../data/val/Diabrotica_spreciosa'
    ]
    
    print("\nCLASSES DISPONÍVEIS:")
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
    #Mostra menu de imagens na classe selecionada
    images = list_images_in_directory(selected_dir)
    
    if not images:
        print(f"Nenhuma imagem encontrada em {selected_dir}")
        return None
    
    print(f"\nIMAGENS DISPONÍVEIS EM '{class_name}' (mostrando primeiras 10):")
    display_images = images[:10]  
    
    for i, image in enumerate(display_images, 1):
        print(f"  {i}. {image}")
    
    if len(images) > 10:
        print(f"  ... e mais {len(images) - 10} imagens")
        print(f"  (Digite qualquer número de 1 a {len(images)})")
    
    return images

def get_user_choice(prompt, max_option):
    #Pega a escolha do usuário com validação
    while True:
        try:
            choice = input(prompt)
            if choice.lower() in ['q', 'quit', 'sair']:
                return None
            
            choice = int(choice)
            if 1 <= choice <= max_option:
                return choice
            else:
                print(f"Por favor, digite um número entre 1 e {max_option}")
        except ValueError:
            print("Por favor, digite um número válido")

def test_random_images(predictor, num_tests=5):
    #Testa imagens aleatórias de diferentes classes
    import random
    
    val_dirs = [
        '../../data/val/Saudavel',
        '../../data/val/Fungo', 
        '../../data/val/Lagarta',
        '../../data/val/Pinta_Preta',
        '../../data/val/Requeima',
        '../../data/val/Diabrotica_spreciosa'
    ]
    
    print(f"\nTESTE RÁPIDO - {num_tests} imagens aleatórias")
    print("=" * 50)
    
    correct = 0
    total = 0
    
    for i in range(num_tests):
        # Escolher classe aleatória
        available_dirs = [d for d in val_dirs if os.path.exists(d)]
        if not available_dirs:
            break
            
        random_dir = random.choice(available_dirs)
        class_name = os.path.basename(random_dir)
        images = list_images_in_directory(random_dir)
        
        if not images:
            continue
            
        # Escolher imagem aleatória
        random_image = random.choice(images)
        image_path = os.path.join(random_dir, random_image)
        
        print(f"\n {i+1}. {random_image}")
        print(f" Classe real: {class_name}")
        
        result = predictor.predict_single_image(image_path)
        
        if result:
            predicted = result['predicted_class']
            confidence = result['confidence']
            
            print(f"Predição: {predicted} ({confidence:.3f})")
            
            if predicted == class_name:
                print("CORRETO")
                correct += 1
            else:
                print("ERRADO")
            
            total += 1
        else:
            print("Erro na predição")
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\nResultado: {correct}/{total} corretas ({accuracy:.1f}%)")

def main():
    print("=" * 60)
    
    try:
        predictor = PlantDiseasePredictor('../outputs/Modelo_Folha_90.pth', device='cpu')
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
    
    while True:
        print("\n OPÇÕES:")
        print("1. Testar imagem específica")
        print("2. Teste rápido (5 imagens aleatórias)")
        print("3. Sair")
        
        choice = get_user_choice("Sua escolha: ", 3)
        
        if choice is None or choice == 3:
            print(" Saindo...")
            break
        elif choice == 2:
            test_random_images(predictor)
            continue
        
        # Opção pra testar imagem específica
        available_classes = show_menu_classes()
        
        if not available_classes:
            print("Nenhuma classe com imagens encontrada!")
            continue
        
        # O usuário  seleciona classe
        print(f"\n Escolha uma classe (1-{len(available_classes)}) ou 'q' para voltar:")
        class_choice = get_user_choice("Sua escolha: ", len(available_classes))
        
        if class_choice is None:
            continue
        
        # Pega o diretório e nome da classe selecionada
        selected_class = available_classes[class_choice - 1]
        _, selected_dir, class_name = selected_class
        
        # Mostra as imagens disponíveis
        images = show_menu_images(selected_dir, class_name)
        if not images:
            continue
        
        # Permite o usuário selecionar a imagem
        print(f"\n  Escolha uma imagem (1-{len(images)}) ou 'q' para voltar:")
        image_choice = get_user_choice("Sua escolha: ", len(images))
        
        if image_choice is None:
            continue
        
        # Faz a predição
        selected_image = images[image_choice - 1]
        image_path = os.path.join(selected_dir, selected_image)
        
        print(f"\n Testando: {selected_image}")
        print(f"  Classe real: {class_name}")
        print(" Processando...")
        
        result = predictor.predict_single_image(image_path)
        
        if result:
            print(f"\n Predição: {result['predicted_class']}")
            print(f" Confiança: {result['confidence']:.3f}")
            
            # Verifica se acertou
            correct = " CORRETO" if result['predicted_class'] == class_name else " ERRADO"
            print(f"   {correct}")
            
            # Mostra o top 3 probabilidades
            print("\n Top 3 probabilidades:")
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            
            for i, (classe, prob) in enumerate(sorted_probs, 1):
                print(f"  {classe}: {prob:.3f}")
        else:
            print(" Erro ao processar a imagem")

if __name__ == "__main__":
    main()