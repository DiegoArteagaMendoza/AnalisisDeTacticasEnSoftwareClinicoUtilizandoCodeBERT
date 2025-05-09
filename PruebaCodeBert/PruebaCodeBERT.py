from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
from git import Repo
from tqdm import tqdm
from collections import Counter
from lime.lime_text import LimeTextExplainer
import numpy as np
import os
import torch
import csv

# Clonar el repositorio a analizar
def clonar_repositorio(url, directorio_destino):
    if not os.path.exists(directorio_destino):
        Repo.clone_from(url, directorio_destino)
        print(f"Repositorio clonado en {directorio_destino}")
    else:
        print(f"El repositorio ya existe en {directorio_destino}")

clonar_repositorio("https://github.com/openemr/openemr.git", "repositorio_openemr")

# Cargar modelo y tokenizador
modelo_entrenado = "../EntrenamientoCodeBert/results10epocas/checkpoint-5420"
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

# Explicabilidad con LIME
def explicabilidad(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = modelo(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

explainer = LimeTextExplainer(class_names=tacticas)


# Analizar archivos
def analizar_archivos(archivo, explicacion=False):
    max_length_tokens = 512
    with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
        contenido = f.read()

    if not contenido or contenido.strip() == "":
        print(f"[{archivo}] -> Archivo vacío.")
        return None

    # Tokeniza y recorta el texto si es muy largo
    tokens = tokenizer.tokenize(contenido)
    if len(tokens) > max_length_tokens:
        tokens = tokens[:max_length_tokens]
        contenido = tokenizer.convert_tokens_to_string(tokens)

    inputs = tokenizer(contenido, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = modelo(**inputs)
    logits = outputs.logits
    predicciones = torch.softmax(logits, dim=1)
    indice_predicho = torch.argmax(predicciones, dim=1).item()

    if predicciones[0][indice_predicho] < 0.9:
        return None

    if explicacion:
        try:
            exp = explainer.explain_instance(
                contenido,
                explicabilidad,
                num_features=10,
                num_samples=50
            )
            os.makedirs("explicaciones", exist_ok=True)
            nombre_archivo = os.path.basename(archivo).replace("/", "_").replace("\\", "_")
            exp.save_to_file(f"explicaciones/{nombre_archivo}_exp.html")
        except ValueError as e:
            print(f"[{archivo}] -> Error al generar explicación: {e}")

    return tacticas[indice_predicho]

# Ejecución principal
def analizar_sistema():
    carpeta = "repositorio_openemr"
    archivos = leer_repositorio(carpeta)
    print(f"Total de archivos leídos: {len(archivos)}")
    reporte = []
    conteo_tacticas = Counter()

    for archivo in tqdm(archivos, desc="Analizando archivos"):
        resultado = analizar_archivos(archivo, explicacion=True)

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



# def analizar_archivos(archivo):
#     with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
#         contenido = f.read()

#     inputs = tokenizer(contenido, return_tensors="pt", truncation=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     outputs = modelo(**inputs)
#     logits = outputs.logits
#     predicciones = torch.softmax(logits, dim=1)
#     indice_predicho = torch.argmax(predicciones, dim=1).item()

#     if predicciones[0][indice_predicho] < 0.9:  # umbral de confianza
#         return None
#     return tacticas[indice_predicho]


# def analizar_archivos(archivo, explicacion):
#     with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
#         contenido = f.read()

#     # if explicacion:
#     #     exp = explainer.explain_instance(contenido, explicabilidad, num_features=10, num_samples=50)
#     #     exp.save_to_file(f"{archivo}_exp.html")
#     #     print(f"Explicacion generada en {archivo}_exp.html")

#     inputs = tokenizer(contenido, return_tensors="pt", truncation=True, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     outputs = modelo(**inputs)
#     logits = outputs.logits
#     predicciones = torch.softmax(logits, dim=1)
#     indice_predicho = torch.argmax(predicciones, dim=1).item()

#     if predicciones[0][indice_predicho] < 0.9:
#         return None
    
#     if explicacion:
#         exp = explainer.explain_instance(contenido, explicabilidad, num_features=10, num_samples=50)
#         exp.save_to_file(f"{archivo}_exp.html")
#         print(f"Explicacion generada en {archivo}_exp.html")

#     return tacticas[indice_predicho]