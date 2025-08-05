from PIL import Image, ImageTk
from Cadastro import abrir_tela_cadastro 
from SobreNos import abrir_sobre_nos
import customtkinter as ctk

# Inicializa o CustomTkinter
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("blue") 

# Janela Principal
janela = ctk.CTk()
janela.title("AGRO G.E.S.F")
largura = 1200
altura = 700

largura_tela = janela.winfo_screenwidth()
altura_tela = janela.winfo_screenheight()

x = int((largura_tela / 2) - (largura / 2))
y = int((altura_tela / 2) - (altura / 2))
janela.geometry(f"{largura}x{altura}+{x}+{y}")
janela.configure(fg_color="#8C9C85") 

# Ícone para Janela
janela.iconbitmap("./Imgs/IconeAgro.ico")

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

# Entrada de nome
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

# Entrada de senha
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

# Função: Entrar 
def Entrar_button_function():
    print("Sei lá porra")

# Função: Cadastrar
def Cadastrar_button_function():
    janela.withdraw()
    abrir_tela_cadastro(janela)

# Botão "Entrar"
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
Entrar_button.pack(padx=60, pady=(0, 30))

# Botão Cadastrar-se
Cadastrar_button = ctk.CTkButton(
    master=frame,
    text="CADASTRAR-SE",
    command=Cadastrar_button_function,
    corner_radius=10,
    width=190,
    height=30,
    font=("Lato", 13),
    fg_color="#F9BB1F",
    hover_color="#d19d19"
)
Cadastrar_button.pack(padx=60, pady=(0, 50))

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
