import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from pathlib import Path

# Caminho das imagens
caminho_imgs = {
    'icon': "./Imgs/IconeAgro.ico",
    'moita_1': "./Imgs/visu_element/Moita_1.png",    
    'moita_2': "./Imgs/visu_element/Moita_2.png",
    'foto_enzo': "./Imgs/devs/Foto_Enzo.png",
    'foto_felipe': "./Imgs/devs/Foto_Felipe.png",
    'foto_sakiri': "./Imgs/devs/Foto_Sakiri.png",
    'foto_gustavo': "./Imgs/devs/Foto_Gustavo.png"
}

# Função para centralizar a janela
def centralizar_janela(window, width, height):
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()
    x = (screen_w - width) // 2
    y = (screen_h - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

# Função para carregar imagens CTk
def carregar_ctk_imagem(path, size):
    if Path(path).is_file():
        img = Image.open(path)
        return ctk.CTkImage(light_image=img, size=size)
    else:
        print(f"Imagem não encontrada: {path}")
        return None

# Função para deixar imagem redonda
def imagem_redonda(caminho_imagem, tamanho=(150, 150)):
    img = Image.open(caminho_imagem).resize(tamanho).convert("RGBA")
    mascara = Image.new('L', tamanho, 0)
    draw = ImageDraw.Draw(mascara)
    draw.ellipse((0, 0) + tamanho, fill=255)
    img.putalpha(mascara)
    return img

def abrir_sobre_nos(janela_login):
    janela_login.withdraw()

    JanelaSobre = tk.Toplevel(janela_login)
    JanelaSobre.title("AGRO G.E.S.F Sobre Nós")
    centralizar_janela(JanelaSobre, 1200, 700)
    JanelaSobre.configure(bg="#274022")
    JanelaSobre.resizable(True, True)

    if Path(caminho_imgs['icon']).is_file():
        JanelaSobre.iconbitmap(caminho_imgs['icon'])

    # ====== Adicionando as moitas no mesmo padrão ======
    moita1_img = carregar_ctk_imagem(caminho_imgs['moita_1'], (350, 130))
    moita2_img = carregar_ctk_imagem(caminho_imgs['moita_2'], (350, 130))

    if moita1_img:
        ctk.CTkLabel(master=JanelaSobre, image=moita1_img, text="").place(relx=1.0, rely=1.0, anchor="se")
    if moita2_img:
        ctk.CTkLabel(master=JanelaSobre, image=moita2_img, text="").place(relx=0.0, rely=1.0, anchor="sw")

    # ====== Conteúdo da página ======
    Foto_Enzo = ctk.CTkImage(light_image=imagem_redonda(caminho_imgs['foto_enzo']), size=(150, 150))
    Foto_felipe = ctk.CTkImage(light_image=imagem_redonda(caminho_imgs['foto_felipe']), size=(150, 150))
    Foto_Sakiri = ctk.CTkImage(light_image=imagem_redonda(caminho_imgs['foto_sakiri']), size=(150, 150))
    Foto_Gustavo = ctk.CTkImage(light_image=imagem_redonda(caminho_imgs['foto_gustavo']), size=(150, 150))

    Label_Titulo = ctk.CTkLabel(
        master=JanelaSobre,
        text="Equipe do AGRO G.E.S.F",
        font=("Poppins", 35, "bold"),
        text_color="#FFFFFF"
    ).pack(pady=(60, 70))

    label_Enzo = ctk.CTkLabel(
        master=JanelaSobre, 
        text="Enzo Costa", 
        image=Foto_Enzo, 
        compound="bottom", 
        font=("Roboto", 20, "bold"),
        text_color=("#FFFFFF")
        ).place(x=100, y=140)
    
    Label_felipe = ctk.CTkLabel(
        master=JanelaSobre, 
        text="Felipe Vieira", 
        image=Foto_felipe, 
        compound="bottom", 
        font=("Roboto", 20, "bold"),
        text_color=("#FFFFFF")
        ).place(x=390, y=140)
    
    Label_Sakiri = ctk.CTkLabel(
        master=JanelaSobre, 
        text="Sakiri Moon", 
        image=Foto_Sakiri, 
        compound="bottom", 
        font=("Roboto", 20, "bold"),
        text_color=("#FFFFFF")
        ).place(x=680, y=140)
    
    Label_Gustavo = ctk.CTkLabel(
        master=JanelaSobre, 
        text="Gustavo Souza", 
        image=Foto_Gustavo, 
        compound="bottom", 
        font=("Roboto", 20, "bold"),
        text_color=("#FFFFFF")
        ).place(x=950, y=140)

    texto_enzo = (
        "Lorem ipsum dolor sit amet\n"
        "consectetur adipiscing elit\n"
        "Sed do eiusmod tempor incididunt\n"
        "labore et dolore magna aliqua\n"
        "Ut enim ad minim veniam"
    )

    frame_widht = 220
    frame_height = 220

    def criar_frame_info(x, y, texto):
        frame = ctk.CTkFrame(master=JanelaSobre, width=frame_widht, height=frame_height, fg_color="#FEF2D5", corner_radius=20)
        frame.place(x=x, y=y)
        ctk.CTkLabel(master=frame, text=texto, font=("Lato", 14), text_color="#000000", justify="left", wraplength=200).place(relx=0.5, rely=0.5, anchor="center")

    criar_frame_info(60, 350, texto_enzo)
    criar_frame_info(360, 350, texto_enzo)
    criar_frame_info(649, 350, texto_enzo)
    criar_frame_info(920, 350, texto_enzo)

    # Botão para voltar
    def voltar_para_login():
        JanelaSobre.destroy()
        janela_login.deiconify()

    #botão provisório(trocar quando tiver dashboard)
    voltar_botão = ctk.CTkButton(
        master=JanelaSobre,
        text="VOLTAR",
        command=voltar_para_login,
        corner_radius=10,
        width=120,
        height=40,
        font=("Lato", 14),
        fg_color="#ADA339",
        hover_color="#918930"
    ).place(x=520, y=620)
