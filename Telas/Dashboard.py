import customtkinter as ctk
from PIL import Image
from pathlib import Path
from Cor_Imgs import cores, caminho_imgs
from SobreNos import abrir_sobre_nos


# centrailizar janelas 
def centralizar_janela(window, width, height):
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()
    x = (screen_w - width) // 2
    y = (screen_h - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")
    
# faz uma pré-carregamento das imagens 
def carregar_ctk_imagem(path, size):
    if Path(path).is_file():
        img = Image.open(path)
        return ctk.CTkImage(light_image=img, size=size)
    return None

def main():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    
    dashBoard = ctk.CTk()
    dashBoard.title("Dashboard")
    centralizar_janela(dashBoard, 1200, 700)
    dashBoard.configure(fg_color=cores['cor_fundo'])
    
    if Path(caminho_imgs['icon']).is_file():
        dashBoard.iconbitmap(caminho_imgs['icon'])
        
    
    # Imagens visuais
    
    Folha_inferior = carregar_ctk_imagem(caminho_imgs['folha_dash_menu_inferior'], (250,250))
    Folha_superior = carregar_ctk_imagem(caminho_imgs['folha_dash_menu_superior'], (250,250))
    
    
    if Folha_inferior:
        Folha_inferior_label = ctk.CTkLabel(
        master=dashBoard,
        image=Folha_inferior,
        text=""
    )
    Folha_inferior_label.place(relx=1.0, rely=1.0, anchor="se")

    if Folha_superior:
        Folha_superior_label = ctk.CTkLabel(
        master=dashBoard,
        image=Folha_superior,
        text=""
    )
    Folha_superior_label.place(relx=0.0, rely=0.0, anchor="nw")
    
    
    
    # Frames (trocar para outros arquivos)
    frame_dashboard = ctk.CTkFrame(master=dashBoard, fg_color=cores['cor_fundo'])
    frame_glossario = ctk.CTkFrame(master=dashBoard, fg_color=cores['cor_fundo'])
    frame_config = ctk.CTkFrame(master=dashBoard, fg_color=cores['cor_fundo'])
    frame_dados = ctk.CTkFrame(master=dashBoard, fg_color=cores['cor_fundo'])
    
    
    # Labels para diferenciar páginas
    ctk.CTkLabel(
        frame_dashboard, 
        text="Página Home",
        font=("Lato", 30),
        text_color=cores['branco']
    ).pack(pady=20)
    
    ctk.CTkLabel(
        frame_glossario, 
        text="Página Glossário",
        font=("Lato", 30),
        text_color=cores['branco']
    ).pack(pady=20)
    
    ctk.CTkLabel(
        frame_config, 
        text="Página Configurações",
        font=("Lato", 30),
        text_color=cores['branco']
    ).pack(pady=20)
    
    ctk.CTkLabel(
        frame_dados,
        text="Pagina de Dados",
        font=("Lato", 30),
        text_color=cores['branco']
    ).pack(pady=20)
    
    # Função para trocar página
    def Trocar_Pagina(pagina):
        frame_dashboard.lower()
        frame_glossario.lower()
        frame_config.lower()
        frame_dados.lower()
        pagina.lift()
        
    # Carregar imagens do menu
    img_home = carregar_ctk_imagem(caminho_imgs['home_menu'], (30,30))
    img_gloss = carregar_ctk_imagem(caminho_imgs['glossario_menu'], (30,30))
    img_config = carregar_ctk_imagem(caminho_imgs['config_menu'], (30,30))
    img_dados = carregar_ctk_imagem(caminho_imgs['dados_menu'],(30,30))
    img_devs = carregar_ctk_imagem(caminho_imgs['dev_menu'], (30,30))
    
    # Barra de menu
    barra_menu = ctk.CTkFrame(
        dashBoard,
        height=60,
        width=350,
        corner_radius=20,
        fg_color=cores['Fundo_Menu']
    )
    barra_menu.pack(side="bottom", pady=10, anchor="s")
    
    # Botões do menu
    dash_buttom = ctk.CTkButton(
        barra_menu,
        image=img_home,
        text="",
        width=40,
        height=40,
        hover_color=cores['branco'],
        fg_color="transparent",
        command=lambda: Trocar_Pagina(frame_dashboard)
    )
    dash_buttom.pack(side="left", expand=True, padx=50, pady=10)
    
    Glosso_buttom = ctk.CTkButton(
        barra_menu,
        image=img_gloss,
        text="",
        width=40,
        height=40,
        fg_color="transparent",
        hover_color=cores['branco'],
        command=lambda: Trocar_Pagina(frame_glossario)
    )
    Glosso_buttom.pack(side="left", expand=True, padx=50, pady=10)
    
    Config_buttom = ctk.CTkButton(
        barra_menu,
        image=img_config,
        text="",
        width=40,
        height=40,
        fg_color="transparent",
        hover_color=cores['branco'],
        command=lambda:Trocar_Pagina(frame_config)
    )
    Config_buttom.pack(side="left", expand=True, padx=50, pady=10)

    
    Dados_buttom = ctk.CTkButton(
        barra_menu,
        image=img_dados,
        text="",
        width=40,
        height=40,
        fg_color="transparent",
        hover_color=cores['branco'],
        command=lambda:Trocar_Pagina(frame_dados)
        
    )
    Dados_buttom.pack(side="left", expand=True, padx=50, pady=10)
    
    Devs_buttom = ctk.CTkButton(
        barra_menu,
        text="",
        image=img_devs,
        width=40,
        height=40,
        fg_color="transparent",
        hover_color=cores['branco'],
        command=lambda: abrir_sobre_nos(dashBoard)
    )
    Devs_buttom.pack(side="left", expand=True, padx=50, pady=10)
    
    # Começa na Home
    Trocar_Pagina(frame_dashboard)
    
    dashBoard.mainloop()
    
if __name__ == "__main__":
    main()
