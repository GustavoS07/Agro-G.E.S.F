from PIL import Image,ImageTk
import tkinter as tk
import customtkinter as ctk
import os

# gambiarra do caralho usando o toplevel do Tkinter ao invés do CustomTk pra poder botar a porra do icone
# se explodir dps deixa sem icone então
def abrir_tela_cadastro(janela_login):
    cadastro = tk.Toplevel(janela_login)
    cadastro.title("Cadastro - AGRO G.E.S.F")
    largura = 1200
    altura = 700

    # Pega tamanho da tela
    largura_tela = janela_login.winfo_screenwidth()
    altura_tela = janela_login.winfo_screenheight()

    # Calcula posição centralizada
    x = int((largura_tela / 2) - (largura / 2))
    y = int((altura_tela / 2) - (altura / 2))

    # Define geometria da janela de cadastro
    cadastro.geometry(f"{largura}x{altura}+{x}+{y}")
    cadastro.configure(bg="#8C9C85")
    cadastro.resizable(True, True)


    # Ícone da janela
    cadastro.iconbitmap("./Imgs/IconeAgro.ico")

    # Frame principal
    frame_princ = ctk.CTkFrame(cadastro, width=721, height=789, corner_radius=20, fg_color="#FEF2D5")
    frame_princ.place(relx=0.5, rely=0.475, anchor="center")

    # Carregamento de imagens
    logo_convert = ctk.CTkImage(light_image=Image.open("./Imgs/LogoFinal.png"), size=(300, 160))
    nome_icon = ctk.CTkImage(light_image=Image.open("./Imgs/Perfil.png"), size=(35, 35))
    email_icon = ctk.CTkImage(light_image=Image.open("./Imgs/email.png"), size=(35, 35))
    senha_icon = ctk.CTkImage(light_image=Image.open("./Imgs/SenhaLogo.png"), size=(35, 35))

    # Logo
    ctk.CTkLabel(master=frame_princ, image=logo_convert, text="").pack(pady=(5, 10))

    # Nome
    Nome_rotulo = ctk.CTkLabel(
        master=frame_princ, 
        text="Nome   ", 
        image=nome_icon, 
        compound="right", 
        font=("Lato", 15, "bold")).pack(padx=60, pady=(5, 2), anchor="w")
    
    Nome_entrada = ctk.CTkEntry(
        master=frame_princ, 
        width=350, 
        height=40, 
        fg_color="#DADADA", 
        border_width=0, 
        corner_radius=10).pack(padx=60, pady=(0, 5), anchor="w")

    # Email
    email_rotulo = ctk.CTkLabel(
        master=frame_princ, 
        text="Email   ", 
        image=email_icon, 
        compound="right", 
        font=("Lato", 15, "bold")).pack(padx=60, pady=(6, 2), anchor="w")
    
    email_entrada = ctk.CTkEntry(
        master=frame_princ, 
        width=350, height=40, fg_color="#DADADA", 
        border_width=0, corner_radius=10).pack(padx=60, pady=(0, 6), anchor="w")

    # Senha
    Senha_rotulo = ctk.CTkLabel(
        master=frame_princ, 
        text="Senha   ", 
        image=senha_icon, 
        compound="right", 
        font=("Lato", 15, "bold")).pack(padx=60, pady=(7, 2), anchor="w")
    
    Senha_entrada = ctk.CTkEntry(
        master=frame_princ, 
        width=350, 
        height=40, 
        fg_color="#DADADA", 
        border_width=0, 
        corner_radius=10, 
        show="*").pack(padx=60, pady=(0, 7), anchor="w")

    # Repetir senha
    Rep_Senha_rotulo = ctk.CTkLabel(
        master=frame_princ, 
        text="Repetir Senha", 
        font=("Lato", 15, "bold")).pack(padx=60, pady=(8, 2), anchor="w")
    
    REp_Senha_Entrada = ctk.CTkEntry(
        master=frame_princ, 
        width=350, 
        height=40, 
        fg_color="#DADADA", 
        border_width=0, 
        corner_radius=10, 
        show="*").pack(padx=60, pady=(0, 8), anchor="w")

    # Frame dos botões
    botoes_frame = ctk.CTkFrame(
        master=frame_princ, 
        fg_color="transparent")
    botoes_frame.pack(pady=(10, 10))

    # Botões lado a lado
    cancelar = ctk.CTkButton(
        master=botoes_frame, 
        text="CANCELAR", 
        width=150, 
        height=40, 
        fg_color="#A33B3B",
        hover_color="#8c3030", font=("Lato", 14), corner_radius=10,
        command=cadastro.destroy).grid(row=0, column=0, padx=0)

    cadastrar = ctk.CTkButton(
        master=botoes_frame, 
        text="CADASTRAR", 
        width=150, 
        height=40, 
        fg_color="#22532C",
        hover_color="#1a4022", 
        font=("Lato", 14), 
        corner_radius=10).grid(row=0, column=1, padx=0.5)

    # Botão voltar
    def voltar_para_login():
        cadastro.destroy()
        janela_login.deiconify()

    cancelar = ctk.CTkButton(
        master=frame_princ, 
        text="VOLTAR", 
        width=200, height=35, 
        fg_color="#F9BB1F", 
        hover_color="#d19d19",
        font=("Lato", 13), 
        corner_radius=20, 
        command=voltar_para_login).pack(pady=(5, 20))