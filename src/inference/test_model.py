import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# Adicionar src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.cnn import CNN

class PlantDiseasePredictor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Classes do seu dataset
        self.class_names = [
            'Diabrotica_spreciosa', 
            'Fungo', 
            'Lagarta', 
            'Pinta_Preta', 
            'Requeima', 
            'Saudavel'
        ]
        
        # Transformações (mesmo padrão do treinamento)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Carregar modelo
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """Carrega o modelo treinado"""
        model = CNN(num_classes=len(self.class_names))
        
        try:
            # Tentar carregar checkpoint completo
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Modelo carregado do checkpoint (época {checkpoint.get('epoch', 'N/A')})")
            else:
                # Carregar apenas state_dict
                model.load_state_dict(checkpoint)
                print("✅ Modelo carregado (state_dict)")
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return None
            
        model.to(self.device)
        model.eval()
        return model
    
    def predict_single_image(self, image_path):
        """Prediz uma única imagem"""
        try:
            # Carregar e preprocessar imagem
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Fazer predição
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'all_probabilities': {
                    self.class_names[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }
            
        except Exception as e:
            print(f"❌ Erro ao processar imagem: {e}")
            return None
    
    def predict_batch(self, image_folder):
        """Prediz todas as imagens de uma pasta"""
        results = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(image_folder, filename)
                result = self.predict_single_image(image_path)
                
                if result:
                    result['filename'] = filename
                    results.append(result)
                    print(f"📸 {filename}: {result['predicted_class']} "
                          f"(Confiança: {result['confidence']:.3f})")
        
        return results
    
    def evaluate_on_validation(self, val_dataloader):
        """Avalia o modelo no conjunto de validação"""
        correct = 0
        total = 0
        class_correct = {name: 0 for name in self.class_names}
        class_total = {name: 0 for name in self.class_names}
        
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Accuracy por classe
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_name = self.class_names[label]
                    class_total[class_name] += 1
                    if predicted[i] == label:
                        class_correct[class_name] += 1
        
        overall_accuracy = correct / total
        
        print(f"\n🎯 ACCURACY GERAL: {overall_accuracy:.3f} ({correct}/{total})")
        print("\n📊 ACCURACY POR CLASSE:")
        for class_name in self.class_names:
            if class_total[class_name] > 0:
                acc = class_correct[class_name] / class_total[class_name]
                print(f"  {class_name}: {acc:.3f} ({class_correct[class_name]}/{class_total[class_name]})")
        
        return overall_accuracy, class_correct, class_total

# Exemplo de uso
if __name__ == "__main__":
    # Inicializar preditor
    predictor = PlantDiseasePredictor('../../checkpoint.pth')
    
    # Testar uma imagem específica
    result = predictor.predict_single_image('../../data/raw/Fungo/fungo.JPG ')
    # print(result)
    
    # Testar pasta de imagens
    # results = predictor.predict_batch('data/val/Saudavel')
    
    print("🚀 Preditor inicializado com sucesso!")
    print("📋 Classes disponíveis:", predictor.class_names)
