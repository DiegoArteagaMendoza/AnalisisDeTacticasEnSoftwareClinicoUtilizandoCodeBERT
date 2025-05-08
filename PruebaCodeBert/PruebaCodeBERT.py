from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
from git import Repo
from tqdm import tqdm
import os
import torch
import csv
from collections import Counter

# Clonar el repositorio a analizar
def clonar_repositorio(url, directorio_destino):
    if not os.path.exists(directorio_destino):
        Repo.clone_from(url, directorio_destino)
        print(f"Repositorio clonado en {directorio_destino}")
    else:
        print(f"El repositorio ya existe en {directorio_destino}")

clonar_repositorio("https://github.com/openemr/openemr.git", "repositorio_openemr")

# Cargar modelo y tokenizador
modelo_entrenado = "../EntrenamientoCodeBert/results5epocas/checkpoint-1626"
print("¿Directorio del modelo existe?", os.path.isdir(modelo_entrenado))

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
modelo = RobertaForSequenceClassification.from_pretrained(modelo_entrenado, local_files_only=True)
modelo.eval()

if tokenizer and modelo_entrenado:
    print("Modelo y tokenizer cargados correctamente.")

# Enviar a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)
print("Dispositivo utilizado: ", device)

# Listado de tácticas
tacticas = [
    "acl", "input_validation", "encryption", "sanitization", "blowfish",
    "digest_authentication", "kerberos", "ldap", "md5", "oauth2", "rsa",
    "session_management", "sftp", "sha256", "sha512", "ssh", "vpn", "aes",
    "audit_trail", "3des", "tls", "token_based_authentication"
]

# Leer repositorio
def leer_repositorio(carpeta):
    extensiones_validas = [".php", ".js", ".py", ".java", ".cpp"]
    archivos = []

    for carpeta, _, archivos_carpeta in os.walk(carpeta):
        for archivo in archivos_carpeta:
            if any(archivo.endswith(ext) for ext in extensiones_validas):
                archivos.append(os.path.join(carpeta, archivo))
    
    return archivos

# Analizar archivos
def analizar_archivos(archivo):
    with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
        contenido = f.read()

    inputs = tokenizer(contenido, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = modelo(**inputs)
    logits = outputs.logits
    predicciones = torch.softmax(logits, dim=1)
    indice_predicho = torch.argmax(predicciones, dim=1).item()

    if predicciones[0][indice_predicho] < 0.9:  # umbral de confianza
        return None
    return tacticas[indice_predicho]

# Ejecución principal
def analizar_sistema():
    carpeta = "repositorio_openemr"
    archivos = leer_repositorio(carpeta)
    print(f"Total de archivos leídos: {len(archivos)}")
    reporte = []
    conteo_tacticas = Counter()

    for archivo in tqdm(archivos, desc="Analizando archivos"):
        resultado = analizar_archivos(archivo)

        if resultado:
            print(f"[{archivo}] -> Táctica detectada: {resultado}.")
            reporte.append((archivo, resultado))
            conteo_tacticas[resultado] += 1
        else:
            print(f"[{archivo}] -> No se detectaron tácticas.")
            reporte.append((archivo, "Ninguna"))

    # Guardar reporte en CSV
    with open("reporte_tacticas.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Archivo", "Táctica Detectada"])
        writer.writerows(reporte)

    # Mostrar conteo de tácticas
    print("\nResumen de Tácticas Detectadas:")
    for tactica, cantidad in conteo_tacticas.items():
        print(f"{tactica}: {cantidad}")

    print("\nReporte guardado como 'reporte_tacticas.csv'")

if __name__ == "__main__":
    analizar_sistema()
