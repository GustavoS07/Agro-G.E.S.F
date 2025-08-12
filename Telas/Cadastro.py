from pathlib import Path
from PIL import Image
import tkinter as tk
import customtkinter as ctk

# dicionário de cores
cores = {
    'cor_fundo': "#274022",
    'frame_bg': "#FEF2D5",
    'entrada_bg': "#DADADA",
    'verde_primario': "#4C8042",
    'verde_primario_hover': "#3F6937",
    'amarelo_secundario': "#ADA339",
    'amarelo_secundario_hover': "#918930"
}

# caminhos das imagens
caminho_imgs = {
    'icon': "./Imgs/IconeAgro.ico",
    'logo': "./Imgs/LogoFinal.png",
    'conta': "./Imgs/icons/Perfil.png",
    'email': "./Imgs/icons/email.png",
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

# função para carregar imagens
def carregar_img(path, size):
    if Path(path).is_file():
        img = Image.open(path)
        return ctk.CTkImage(light_image=img, size=size)
    else:
        print(f"Imagem não encontrada: {path}")
        return None

# cria campo com imagem
def criar_campo_com_imagem(master, text, icon_path, size, show=None):
    icon_img = carregar_img(icon_path, size) if icon_path else None
    label = ctk.CTkLabel(master, text=text, image=icon_img, compound="right", font=("Lato", 15, "bold"))
    label.pack(padx=60, pady=(5, 2), anchor="w")

    entrada = ctk.CTkEntry(
        master=master,
        width=350,
        height=40,
        fg_color=cores['entrada_bg'],
        border_width=0,
        corner_radius=10,
        show=show
    )
    entrada.pack(padx=60, pady=(0, 5), anchor="w")
    return entrada

# janela de cadastro
def abrir_tela_cadastro(janela_login):
    cadastro = tk.Toplevel(janela_login)
    cadastro.title("Cadastro - AGRO G.E.S.F")
    cadastro.configure(bg=cores['cor_fundo'])
    cadastro.resizable(True, True)
    centralizar_janela(cadastro, 1200, 700)

    # Ícone
    if Path(caminho_imgs['icon']).is_file():
        cadastro.iconbitmap(caminho_imgs['icon'])

    # === Moitas nos cantos ===
    moita1_img = carregar_img(caminho_imgs['moita_1'], (350, 130))
    moita2_img = carregar_img(caminho_imgs['moita_2'], (350, 130))
    moita_flu_1 = carregar_img(caminho_imgs['moita_flutu'], (230, 320))
    moita_flu_2 = carregar_img(caminho_imgs['moita_flutu2'], (220, 290))

    # inferior direita
    if moita1_img:
        moita1_label = ctk.CTkLabel(
            master=cadastro, 
            image=moita1_img, 
            text="")
        moita1_label.place(relx=1.0, rely=1.0, anchor="se")

    # inferior esquerda
    if moita2_img:
        moita2_label = ctk.CTkLabel(
            master=cadastro, 
            image=moita2_img, 
            text="")
        moita2_label.place(relx=0.0, rely=1.0, anchor="sw")

    # superior esquerda
    if moita_flu_1:
        moita_flu_1_label = ctk.CTkLabel(
            master=cadastro, 
            image=moita_flu_1, 
            text="")
        moita_flu_1_label.place(relx=0.0, rely=0.0, anchor="nw", x=30)

    # superior direita
    if moita_flu_2:
        moita_flu_2_label = ctk.CTkLabel(
            master=cadastro, 
            image=moita_flu_2, 
            text="")
        moita_flu_2_label.place(relx=1.0, rely=0.0, anchor="ne", x=-30)

    # Frame principal
    frame_princ = ctk.CTkFrame(
        master=cadastro, 
        width=721, height=789, 
        corner_radius=20, 
        fg_color=cores['frame_bg'])
    frame_princ.place(relx=0.5, rely=0.475, anchor="center")

    # Logo
    logo_img = carregar_img(caminho_imgs['logo'], (300, 160))
    if logo_img:
        ctk.CTkLabel(
            master=frame_princ, 
            image=logo_img, 
            text="").pack(pady=(5, 10))
        
    # Campos de entrada
    criar_campo_com_imagem(frame_princ, "Nome   ", caminho_imgs['conta'], (35, 35))
    criar_campo_com_imagem(frame_princ, "Email   ", caminho_imgs['email'], (35, 35))
    criar_campo_com_imagem(frame_princ, "Senha   ", caminho_imgs['senha'], (35, 35), show="*")
    criar_campo_com_imagem(frame_princ, "Repetir Senha   ", None, (0, 0), show="*")

    # Frame de botões
    botoes_frame = ctk.CTkFrame(
        master=frame_princ, 
        fg_color="transparent")
    botoes_frame.pack(pady=(10, 10))

    cadastrar_botao = ctk.CTkButton(
        master=botoes_frame,
        text="CADASTRAR",
        width=170,
        height=40,
        fg_color=cores['verde_primario'],
        hover_color=cores['verde_primario_hover'],
        font=("Lato", 17),
        corner_radius=10
    ).grid(row=0, column=1, padx=5)

    # Botão voltar
    def voltar_para_login():
        cadastro.destroy()
        janela_login.deiconify()

    voltar_botao = ctk.CTkButton(
        master=frame_princ,
        text="VOLTAR",
        width=130,
        height=35,
        fg_color=cores['amarelo_secundario'],
        hover_color=cores['amarelo_secundario_hover'],
        font=("Lato", 13),
        corner_radius=20,
        command=voltar_para_login
    ).pack(pady=(5, 20))
