from PIL import Image
from Cadastro import abrir_tela_cadastro 
from SobreNos import abrir_sobre_nos
import customtkinter as ctk

# Inicializa o CustomTkinter
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("blue") 

# Janela Principal
janela = ctk.CTk()
janela.title("AGRO G.E.S.F")
largura_inicial = 1200
altura_inicial = 700

largura_tela = janela.winfo_screenwidth()
altura_tela = janela.winfo_screenheight()

x = int((largura_tela / 2) - (largura_inicial / 2))
y = int((altura_tela / 2) - (altura_inicial / 2))
janela.geometry(f"{largura_inicial}x{altura_inicial}+{x}+{y}")

# Ícone da Janela
janela.iconbitmap("./Imgs/IconeAgro.ico")

# ====== IMAGEM DE FUNDO DINÂMICA ======
imagem_original = Image.open("./Imgs/FundoJanela.png")

imagem_ctk = ctk.CTkImage(
    light_image=imagem_original,
    dark_image=imagem_original,
    size=(largura_inicial, altura_inicial)
)

background_label = ctk.CTkLabel(janela, text="", image=imagem_ctk)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Atualiza imagem de fundo ao redimensionar janela
def redimensionar_imagem(event):
    nova_largura = event.width
    nova_altura = event.height
    nova_ctk_image = ctk.CTkImage(
        light_image=imagem_original.resize((nova_largura, nova_altura)),
        dark_image=imagem_original.resize((nova_largura, nova_altura)),
        size=(nova_largura, nova_altura)
    )
    background_label.configure(image=nova_ctk_image)
    background_label.image = nova_ctk_image

janela.bind("<Configure>", redimensionar_imagem)

# Frame principal
frame = ctk.CTkFrame(janela, width=721, height=789, corner_radius=20)
frame.place(relx=0.5, rely=0.5, anchor="center")
frame.configure(fg_color="#FEF2D5")

# Imagens
Logo = Image.open("./Imgs/LogoFinal.png")
imagemConvert = ctk.CTkImage(light_image=Logo, size=(300, 160))
Logo_Label = ctk.CTkLabel(master=frame, image=imagemConvert, text="")
Logo_Label.pack(pady=(5, 10))  

Nome_img = Image.open("./Imgs/Perfil.png")
Nome_Img_Convert = ctk.CTkImage(light_image=Nome_img, size=(35, 35))

Senha_img = Image.open("./Imgs/SenhaLogo.png")
Senha_img_Convert = ctk.CTkImage(light_image=Senha_img, size=(35,35))

# Rótulo Nome
Rotulo_nome = ctk.CTkLabel(
    master=frame,
    text="Nome   ",
    image=Nome_Img_Convert,
    compound="right",
    font=("Lato", 20, "bold")
)
Rotulo_nome.pack(padx=60, pady=(10, 2), anchor="w") 

# Entrada Nome
Nome_entrada = ctk.CTkEntry(
    master=frame,
    width=350,
    height=40,
    placeholder_text="",
    fg_color="#DADADA",
    border_width=0,
    corner_radius=10
)
Nome_entrada.pack(padx=60, pady=(0, 15), anchor="w")  

# Rótulo Senha
Rotulo_senha = ctk.CTkLabel(
    master=frame,
    text="Senha   ",
    image=Senha_img_Convert,
    compound="right",
    font=("Lato", 20, "bold")
)
Rotulo_senha.pack(padx=60, pady=(5, 2), anchor="w")

# Entrada Senha
Senha_entrada = ctk.CTkEntry(
    master=frame,
    width=350,
    height=40,
    placeholder_text="",
    fg_color="#DADADA",
    border_width=0,
    corner_radius=10,
    show="*"
)
Senha_entrada.pack(padx=60, pady=(0, 20), anchor="w")

# Funções dos Botões
def Entrar_button_function():
    print("Está funcioanando")

def Cadastrar_button_function():
    janela.withdraw()
    abrir_tela_cadastro(janela)

# Botões
Entrar_button = ctk.CTkButton(
    master=frame,
    text="ENTRAR",
    command=Entrar_button_function,
    corner_radius=10,
    width=270,
    height=60,
    font=("Lato", 25),
    fg_color="#22532C",
    hover_color="#1a4022"
)
Entrar_button.pack(padx=60, pady=(0, 45))

# label "Não tem uma conta?"
Sem_Conta_label = ctk.CTkLabel(
    master=frame,
    text="Não tem uma conta?",
    font=("Lato,", 15)
)
Sem_Conta_label.place(x=170, y=450)
Cadastrar_button = ctk.CTkButton(
    master=frame,
    text="CRIAR UMA CONTA",
    command=Cadastrar_button_function,
    corner_radius=10,
    width=190,
    height=30,
    font=("Lato", 13),
    fg_color="#ADA339",
    hover_color="#9C9434"
)
Cadastrar_button.pack(padx=60, pady=(0, 50))


# tirar este botão quando tivermos dashboard
Provisorio_button = ctk.CTkButton(
    master=frame,
    text="PROVISÓRIO",
    command=lambda: abrir_sobre_nos(janela),
    corner_radius=10,
    width=190,
    height=30,
    font=("Lato", 13),
    fg_color="#A9A9A9",
    hover_color="#909090"
)
Provisorio_button.pack(padx=60, pady=(0, 20))

janela.mainloop()
