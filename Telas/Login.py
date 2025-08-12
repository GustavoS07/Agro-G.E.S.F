from pathlib import Path
from PIL import Image
import customtkinter as ctk
from Cadastro import abrir_tela_cadastro
from SobreNos import abrir_sobre_nos

# dicionario de cores

cores = {
    'frame_bg': "#FEF2D5",
    'entrada_bg': "#DADADA",
    'verde_primario': "#22532C",
    'verde_primario_hover': "#1a4022",
    'amarelo_secundario': "#ADA339",
    'amarelo_secundario_hover': "#9C9434",
    'cinza': "#A9A9A9",
    'cinza_hover': "#909090"
}

#dicionario de imagens

caminho_imgs = {
    'icon': "./Imgs/IconeAgro.ico",
    'logo': "./Imgs/LogoFinal.png",
    'Usuario': "./Imgs/icons/Perfil.png",
    'senha': "./Imgs/icons/SenhaLogo.png",
    'moita_1': "./Imgs/visu_element/Moita_1.png",
    'moita_2': "./Imgs/visu_element/Moita_2.png",
    'moita_flutu': "./Imgs/visu_element/Moita_flutuante.png",
    'moita_flutu2': "./Imgs/visu_element/Moita_flutuante2.png"
}

# função pra centralizar a janela
def centralizar_janela(window, width, height):
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()
    x = (screen_w - width) // 2
    y = (screen_h - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

#função para carregar todas as imagens
def carregar_ctk_imagem(path, size):
    if Path(path).is_file():
        img = Image.open(path)
        return ctk.CTkImage(light_image=img, size=size)
    else:
        print(f"Imagem não encontrada: {path}")
        return None

#cria um campo com imagem na direita, assim precisando apenas chamar a função e colocar um nome para ela
def criar_campo_com_imagem(master, text, icon_path, size, show=None):
    icon_img = carregar_ctk_imagem(icon_path, size)
    label = ctk.CTkLabel(
        master, 
        text = text, 
        image=icon_img, 
        compound="right", 
        font=("Lato", 20, "bold"))
    label.pack(padx=60, pady=(10, 2), anchor="w")

    entry = ctk.CTkEntry(
        master=master,
        width=350,
        height=40,
        fg_color=cores['entrada_bg'],
        border_width=0,
        corner_radius=10,
        show=show
    )
    entry.pack(padx=60, pady=(0, 15), anchor="w")
    return entry


# interface
def main():
    # Inicializa o CustomTkinter
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    # Janela Principal
    janela = ctk.CTk()
    janela.title("AGRO G.E.S.F")
    centralizar_janela(janela, 1200, 700)

    if Path(caminho_imgs['icon']).is_file():
        janela.iconbitmap(caminho_imgs['icon'])

    janela.configure(fg_color="#274022")

    # Frame principal
    frame = ctk.CTkFrame(
        master=janela, 
        width=721, 
        height=789, 
        corner_radius=20, 
        fg_color=cores['frame_bg'])
    frame.place(relx=0.5, rely=0.5, anchor="center")

    # Logo
    logo_img = carregar_ctk_imagem(caminho_imgs['logo'], (300, 160))
    if logo_img:
        logo_label = ctk.CTkLabel(
            master=frame, 
            image=logo_img, 
            text="")
        logo_label.pack(pady=(5, 10))

    # Moitas (cantos da tela)
    moita1_img = carregar_ctk_imagem(caminho_imgs['moita_1'], (350, 130))
    moita2_img = carregar_ctk_imagem(caminho_imgs['moita_2'], (350, 130))
    
    # chama a imagem moita_1
    if moita1_img:
        moita1_label = ctk.CTkLabel(
            master=janela, 
            image=moita1_img, 
            text="")
        moita1_label.place(relx=1.0, rely=1.0, anchor="se")
        
    #chama a imagem moita_2
    if moita2_img:
        moita2_label = ctk.CTkLabel(
            master=janela, 
            image=moita2_img, 
            text="")
        moita2_label.place(relx=0.0, rely=1.0, anchor="sw")
        
    # moitas flutuantes
    
    moita_flu_1 = carregar_ctk_imagem(caminho_imgs['moita_flutu'], (230, 320))
    moita_flu_2 = carregar_ctk_imagem(caminho_imgs['moita_flutu2'], (220, 290))
    
    #chama moita_flu_1
    if moita_flu_1:
        moita_flu_1_label = ctk.CTkLabel(
            master=janela,
            image=moita_flu_1,
            text=""
        )
        moita_flu_1_label.place(relx=0.0, rely=0.0, anchor="nw", x=30)
        
    # chama moita_flu_2
    if moita_flu_2:
        moita_flu_2_label = ctk.CTkLabel(
            master=janela,
            image=moita_flu_2,
            text=""
        )
        moita_flu_2_label.place(relx=1.0, rely=0.0, anchor="ne", x=-30)

    # Campos de entrada
    nome_entrada = criar_campo_com_imagem(frame, "Nome  ", caminho_imgs['Usuario'], (35, 35))
    senha_entrada = criar_campo_com_imagem(frame, "Senha  ", caminho_imgs['senha'], (35, 35), show="*")

    # Funções dos botões
    
    #Função desligada pois não tem banco de dados ainda
    """def entrar():
        nome = nome_entrada.get().strip()
        senha = senha_entrada.get()
        if nome and senha:
            print(f"Tentativa de login: {nome}")
        else:
            print("Preencha todos os campos!")"""

    def cadastrar():
        janela.withdraw()
        abrir_tela_cadastro(janela)

    def sobre_nos():
        abrir_sobre_nos(janela)

    # Botões
    entrar_btn = ctk.CTkButton(
        master=frame,
        text="ENTRAR",
        command=None,
        corner_radius=10,
        width=270,
        height=60,
        font=("Lato", 25),
        fg_color=cores['verde_primario'],
        hover_color=cores['verde_primario_hover']
    )
    entrar_btn.pack(padx=60, pady=(0, 45))

    sem_conta_label = ctk.CTkLabel(master=frame, text="Não tem uma conta?", font=("Lato", 15))
    sem_conta_label.place(x=170, y=450)

    cadastrar_btn = ctk.CTkButton(
        master=frame,
        text="CRIAR UMA CONTA",
        command=cadastrar,
        corner_radius=10,
        width=190,
        height=30,
        font=("Lato", 13),
        fg_color=cores['amarelo_secundario'],
        hover_color=cores['amarelo_secundario_hover']
    )
    cadastrar_btn.pack(padx=60, pady=(0, 50))

    provisorio_btn = ctk.CTkButton(
        master=frame,
        text="SOBRE NÓS",
        command=sobre_nos,
        corner_radius=10,
        width=190,
        height=30,
        font=("Lato", 13),
        fg_color=cores['cinza'],
        hover_color=cores['cinza_hover']
    )
    provisorio_btn.pack(padx=60, pady=(0, 20))

    janela.mainloop()

if __name__ == "__main__":
    main()
