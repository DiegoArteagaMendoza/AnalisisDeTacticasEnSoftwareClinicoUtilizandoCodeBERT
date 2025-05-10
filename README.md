 # Requisitos y Configuración

- Utilizar un **entorno virtual** con Python **3.9.13**  
- El modelo fue entrenado localmente utilizando una **RTX 3050 de 6GB**

## Instalación de dependencias

Al instalar las dependencias desde `requirements.txt`, puede ocurrir un error relacionado con las librerías de `torch`. En ese caso, se recomienda instalar manualmente las versiones compatibles de `torch`, `torchvision` y `torchaudio` según tu entorno CUDA.

### Comando utilizado en entorno local (CUDA 11.8):

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### Archivos necesarios

Debido al tamaño del modelo, es necesario descargar los siguientes archivos:

- `model.safetensors`
- `optimizer.pt`

Puedes encontrarlos en el siguiente enlace de Google Drive:  
🔗 [Descargar modelo y optimizador](https://drive.google.com/drive/folders/1mXOTvICQqAnh4yiTzzZupZyvu5IkMKLb?usp=sharing)
