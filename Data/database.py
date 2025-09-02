import sqlite3 as sql
import os

def delete():
    connection = sql.connect('database.db')
    cursor = connection.cursor()

    try:
        cursor.execute("""
               DROP TABLE usuario
        """)
        resposta = cursor.fetchone()
        if resposta:
            cursor.commit()
            cursor.close()
        else:
            print("Erro")
    except Exception as e:
        print(f"Erro {e}")

def createTables():
    connection = sql.connect('database.db')
    cursor = connection.cursor()

    try:
        cursor.execute("""
                CREATE TABLE usuario(
                id INTEGER PRIMARY KEY,
                nome TEXT , 
                senha TEXT,
                email TEXT,
                imagemPerfil TEXT
            ) 
        """)
        resposta = cursor.fetchone()
        if resposta:
            cursor.commit()
            cursor.close()
        else:
            print("Erro")
    except Exception as e:
        print(f"Erro {e}")

def cadastrarUsuario(nome,email,senha,rep_senha,imagemPerfil):
    connection = sql.connect('database.db')
    cursor = connection.cursor()

    try:
        command = "INSERT INTO usuario(nome,email,senha,imagemPerfil) VALUES(?,?,?,?)"
        cursor.execute(command,(nome,email,senha,imagemPerfil))
        connection.commit()
            
    except Exception as e:
        print(f"Deu ruim hein chefe o erro é: {e}")


def tentarLogin(nome,senha,on_sucess=None):
    connection = sql.connect('database.db')
    cursor = connection.cursor()
    try:
        comando = "SELECT * FROM usuario WHERE nome == ? AND senha == ? "
        cursor.execute(comando,(nome,senha))
        resposta = cursor.fetchone()

        if resposta:
            print("passei visse")
            if on_sucess:    
                abrir_dashboard
        else:
            print("Erro no else")
    except Exception as e:
        print(f"Erro [{e}]")