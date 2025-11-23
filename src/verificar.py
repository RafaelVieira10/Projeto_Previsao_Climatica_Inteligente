import os

def listar_arquivos():
    print("--- 🕵️ DIAGNÓSTICO DE ARQUIVOS ---")
    
    # 1. Onde estou rodando?
    cwd = os.getcwd()
    print(f"📍 Diretório atual (Raiz): {cwd}")
    
    # 2. Verificando a pasta data/raw
    path_raw = os.path.join(cwd, 'data', 'raw')
    print(f"\n📂 Verificando pasta: {path_raw}")
    
    if os.path.exists(path_raw):
        arquivos = os.listdir(path_raw)
        if not arquivos:
            print("❌ A PASTA ESTÁ VAZIA!")
        else:
            print("✅ Arquivos encontrados (Nomes Exatos):")
            for arq in arquivos:
                print(f"   -> '{arq}'") # Aspas mostram se tem espaço em branco
    else:
        print("❌ A PASTA data/raw NÃO EXISTE!")

if __name__ == "__main__":
    listar_arquivos()