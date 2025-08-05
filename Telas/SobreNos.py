# SobreNos.py

import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

def imagem_redonda(caminho_imagem, tamanho=(150, 150)):
    img = Image.open(caminho_imagem).resize(tamanho).convert("RGBA")
    mascara = Image.new('L', tamanho, 0)
    draw = ImageDraw.Draw(mascara)
    draw.ellipse((0, 0) + tamanho, fill=255)
    img.putalpha(mascara)
    return img

def abrir_sobre_nos(janela_login):
    janela_login.withdraw()  # esconde a tela de login

    JanelaSobre = tk.Toplevel(janela_login)
    JanelaSobre.title("AGRO G.E.S.F Sobre Nós")
    largura = 1200
    altura = 700

    largura_tela = JanelaSobre.winfo_screenwidth()
    altura_tela = JanelaSobre.winfo_screenheight()
    x = int((largura_tela / 2) - (largura / 2))
    y = int((altura_tela / 2) - (altura / 2))

    JanelaSobre.geometry(f"{largura}x{altura}+{x}+{y}")
    JanelaSobre.configure(bg="#8C9C85")
    JanelaSobre.resizable(True, True)
    JanelaSobre.iconbitmap("./Imgs/IconeAgro.ico")

    # imagens
    Foto_Enzo = ctk.CTkImage(light_image=imagem_redonda("./Imgs/Foto_Enzo.png"), size=(150, 150))
    Foto_feipe = ctk.CTkImage(light_image=imagem_redonda("./Imgs/Foto_Felipe.png"), size=(150, 150))
    Foto_Sakiri = ctk.CTkImage(light_image=imagem_redonda("./Imgs/Foto_Sakiri.png"), size=(150, 150))
    Foto_Gustavo = ctk.CTkImage(light_image=imagem_redonda("./Imgs/Foto_Gustavo.png"), size=(150, 150))

    ctk.CTkLabel(
        master=JanelaSobre,
        text="Equipe do AGRO G.E.S.F",
        font=("Poppins", 35,"bold"),
        text_color="#FFFFFF"
    ).pack(pady=(60,70))

    ctk.CTkLabel(master=JanelaSobre, text="Enzo Costa", image=Foto_Enzo, compound="bottom", font=("Roboto", 20,"bold")).place(x=100, y=140)
    ctk.CTkLabel(master=JanelaSobre, text="Felipe Vieria", image=Foto_feipe, compound="bottom", font=("Roboto", 20,"bold")).place(x=390, y=140)
    ctk.CTkLabel(master=JanelaSobre, text="Sakiri Moon", image=Foto_Sakiri, compound="bottom", font=("Roboto", 20,"bold")).place(x=680, y=140)
    ctk.CTkLabel(master=JanelaSobre, text="Gustavo Souza", image=Foto_Gustavo, compound="bottom", font=("Roboto", 20,"bold")).place(x=950, y=140)

    texto_enzo = (
        "Lorem ipsum dolor sit amet\n"
        "consectetur adipiscing elit\n"
        "Sed do eiusmod tempor incididunt\n"
        "labore et dolore magna aliqua\n"
        "Ut enim ad minim veniam"
    )

    frame_widht = 220
    frame_height = 220

    Frame_Enzo = ctk.CTkFrame(
        master=JanelaSobre, 
        width=frame_widht, 
        height=frame_height, 
        fg_color="#FEF2D5", 
        corner_radius=20)
    Frame_Enzo.place(x=60 , y= 350)
    
    Rotulo_frame_Enzo = ctk.CTkLabel(
        master=Frame_Enzo, 
        text=texto_enzo, 
        font=("Lato", 14), 
        text_color="#000000", 
        justify="left", 
        anchor="center", 
        wraplength=200).place(relx=0.5, rely=0.5, anchor="center")

    Frame_Felipe = ctk.CTkFrame(
        master=JanelaSobre, 
        width=frame_widht, 
        height=frame_height, 
        fg_color="#FEF2D5", 
        corner_radius=20)
    Frame_Felipe.place(x=360 , y= 350)
    
    Rotulo_frame_Felipe = ctk.CTkLabel(
        master=Frame_Felipe, 
        text=texto_enzo, 
        font=("Lato", 14), 
        text_color="#000000", 
        justify="left", 
        anchor="center", 
        wraplength=200).place(relx=0.5, rely=0.5, anchor="center")

    Frame_Sakiri = ctk.CTkFrame(
        master=JanelaSobre, 
        width=frame_widht, 
        height=frame_height, 
        fg_color="#FEF2D5", 
        corner_radius=20)
    Frame_Sakiri.place(x=649, y=350)
    
    Rotulo_frame_Sakiri = ctk.CTkLabel(
        master=Frame_Sakiri, 
        text=texto_enzo, 
        font=("Lato", 14), 
        text_color="#000000", 
        justify="left", 
        anchor="center", 
        wraplength=200).place(relx=0.5, rely=0.5, anchor="center")

    Frame_Gustavo = ctk.CTkFrame(
        master=JanelaSobre, 
        width=frame_widht, 
        height=frame_height, 
        fg_color="#FEF2D5", 
        corner_radius=20)
    Frame_Gustavo.place(x=920, y=350)
    
    Rotulo_frame_Gustavo = ctk.CTkLabel(
        master=Frame_Gustavo, 
        text=texto_enzo, 
        font=("lato", 14), 
        text_color="#000000", 
        justify="left", 
        anchor="center", 
        wraplength=200).place(relx=0.5, rely=0.5, anchor="center")

    # Botão para voltar à tela de login
    def voltar_para_login():
        JanelaSobre.destroy()
        janela_login.deiconify()

    ctk.CTkButton(
        master=JanelaSobre,
        text="VOLTAR",
        command=voltar_para_login,
        corner_radius=10,
        width=120,
        height=40,
        font=("Lato", 14),
        fg_color="#22532C",
        hover_color="#1a4022"
    ).place(x=20, y=620)
