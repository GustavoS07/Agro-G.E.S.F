#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from teste_image_cpu import  PlantDiseasePredictor
# Teste rápido
try:
    predictor = PlantDiseasePredictor('../outputs/Modelo_Folha_90.pth', device='cpu')
    print("Modelo carregado")
    
    # Testar com uma imagem
    result = predictor.predict_single_image('../../data/val/Pinta_Preta/teste6.jpg')

    if result:
        print(f"Predição: {result['predicted_class']}")
        print(f"Confiança: {result['confidence']:.3f}")
    else:
        print("Erro na predição")
        
except Exception as e:
    print(f"Erro: {e}")