import os
from git import Repo
from transformers import AutoTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# Directorio de clonación
REPO_URL = "https://github.com/openemr/openemr.git"
REPO_DIR = "repositorio_openemr"
PDF_OUTPUT = "explicabilidad_reporte.pdf"

# Modelo y tokenizer
MODEL_DIR = "checkpoint-5420"
MODEL_NAME = "microsoft/codebert-base"

# Lista de tácticas (clases)
TACTICAS = [
    "acl", "input_validation", "encryption", "sanitization", "blowfish",
    "digest_authentication", "kerberos", "ldap", "md5", "oauth2", "rsa",
    "session_management", "sftp", "sha256", "sha512", "ssh", "vpn", "aes",
    "audit_trail", "3des", "tls", "token_based_authentication"
]

# Número de características LIME
NUM_FEATURES = 10  # reduce tamaño
NUM_SAMPLES = 300  # muestras en LIME

# Dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)


# -- FUNCIONES UTILES --

def clonar_repositorio(url, destino):
    if not os.path.exists(destino):
        Repo.clone_from(url, destino)
        print(f"Repositorio clonado en {destino}")
    else:
        print(f"Repositorio ya existe en {destino}")


def leer_repositorio(carpeta):
    extensiones = [".php", ".js", ".py", ".java", ".cpp"]
    rutas = []
    for root, _, files in os.walk(carpeta):
        for f in files:
            if any(f.endswith(ext) for ext in extensiones):
                rutas.append(os.path.join(root, f))
    return rutas


def cargar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    modelo = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    modelo.to(DEVICE).eval()
    return tokenizer, modelo


def obtener_predicciones(textos, tokenizer, modelo):
    inputs = tokenizer(textos, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = modelo(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def generar_explanacion(texto, explainer, tokenizer, modelo):
    return explainer.explain_instance(
        texto,
        lambda x: obtener_predicciones(x, tokenizer, modelo),
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES,
        top_labels=1
    )


def build_bar_chart(exp_as_list, salida_png):
    # exp_as_list = [(feature, weight), ...]
    features, pesos = zip(*exp_as_list)
    fig, ax = plt.subplots()
    ax.barh(range(len(features)), pesos)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(salida_png)
    plt.close(fig)


def generar_pdf(report_data, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elementos = []

    for archivo, tactica, exp in report_data:
        elementos.append(Paragraph(f"<b>Archivo:</b> {archivo}", styles['Heading2']))
        elementos.append(Paragraph(f"<b>Táctica detectada:</b> {tactica}", styles['Normal']))
        exp_list = exp.as_list(label=exp.available_labels()[0])
        png_path = f"temp_{os.path.basename(archivo)}.png"
        build_bar_chart(exp_list, png_path)
        elementos.append(Image(png_path, width=400, height=200))
        elementos.append(Spacer(1, 12))

    doc.build(elementos)
    print(f"PDF generado: {output_path}")


# -- FLUJO PRINCIPAL --
if __name__ == "__main__":
    # Clonar repositorio
    clonar_repositorio(REPO_URL, REPO_DIR)

    # Cargar modelo
    tokenizer, modelo = cargar_modelo()

    # Inicializar LIME
    explainer = LimeTextExplainer(class_names=TACTICAS)

    archivos = leer_repositorio(REPO_DIR)
    print(f"Archivos a analizar: {len(archivos)}")

    reporte = []
    for ruta in tqdm(archivos, desc="Analizando archivos"):
        try:
            with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
                contenido = f.read()
            if not contenido.strip():
                continue

            # Truncar a 512 tokens
            tokens = tokenizer.tokenize(contenido)
            if len(tokens) > 512:
                contenido = tokenizer.convert_tokens_to_string(tokens[:512])

            # Predicción
            inputs = tokenizer(contenido, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            logits = modelo(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            label = torch.argmax(probs, dim=1).item()

            if probs[0][label] < 0.9:
                continue

            # Explicación
            exp = generar_explanacion(contenido, explainer, tokenizer, modelo)
            tact = TACTICAS[label]
            reporte.append((ruta, tact, exp))

        except Exception as e:
            print(f"Error procesando {ruta}: {e}")

    # Generar PDF
    generar_pdf(reporte, PDF_OUTPUT)
