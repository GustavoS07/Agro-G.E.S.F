## Como Baixar e Instalar

Siga os passos abaixo para clonar o repositório e configurar o projeto:

> [!IMPORTANT]
> Certifique-se de estar em um ambiente compatível (Linux ou Windows) antes de prosseguir.

1. **Clone o repositório**
   ```bash
   git clone https://github.com/GustavoS07/Agro-G.E.S.F.git
   ```
2. **Acesse o Diretório do Projeto**
    ```bash
   cd Agro-G.E.S.F
   git checkout Aprendizado_Estruturado
   ```
> [!TIP]
> Considere usar um ambiente virtual para evitar conflitos de dependências.
  3. **Instale as dependências**
      ```bash
      pip install torch torchvision Pillow
      ```
  4. **Rodando o Teste**
     ```bash
     cd src/inference
     python3 teste.py
     ```

>[!WARNING]
>Caso encontre erros durante a execução, verifique as versões das dependências e se todas as configurações foram feitas corretamente.
