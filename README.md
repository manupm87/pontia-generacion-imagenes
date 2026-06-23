# Finetuning de Stable Diffusion con ilustraciones de libros antiguos

Entrega final del módulo **Modelos de Generación de Imagen**.

Este proyecto hace un *finetuning* completo del modelo
[`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4)
sobre el dataset de **ilustraciones de libros antiguos**
[`gigant/oldbookillustrations`](https://huggingface.co/datasets/gigant/oldbookillustrations),
para llevar el estilo del modelo hacia el de los grabados e ilustraciones antiguas.

Solo se finetunea la **UNet** (el VAE y el text encoder se mantienen congelados). El modelo
resultante se sube a Hugging Face:
[`manupm87/finetuned-sd-1.4-oldbookillustrations`](https://huggingface.co/manupm87/finetuned-sd-1.4-oldbookillustrations).

## ¿Qué hace el notebook?

Todo el trabajo está en **`project.ipynb`**, organizado en pasos:

1. Genera una imagen con el modelo **original** (antes del finetuning).
2. Carga y prepara el dataset (`Resize` + `CenterCrop` a 512x512).
3. Carga las componentes del modelo por separado (tokenizer, scheduler, text encoder, VAE, UNet).
4. Prepara el entrenamiento (optimizador y `accelerate`).
5. Entrena la UNet (finetuning) con un *trigger* de estilo añadido a cada descripción.
6. Sube el pipeline completo y su *model card* a Hugging Face.
7. Genera una imagen con el modelo **finetuneado** (mismo prompt).
8. Compara antes vs. después.
9. Compara 5 prompts con la **misma semilla** en ambos modelos.

Los pasos costosos (entrenamiento y generación) tienen *safeguards*: se saltan
automáticamente si el resultado ya existe en disco. Usa los flags `force_retrain` /
`force_regenerate` del panel de configuración para forzarlos.

## Requisitos

- **GPU NVIDIA con CUDA.** El notebook hace `assert torch.cuda.is_available()`, así que no
  arranca en CPU.
- Python 3.12 (el usado en el entorno del proyecto).
- Una cuenta de Hugging Face con un **token de escritura**
  (https://huggingface.co/settings/tokens), necesario para descargar con mejores cuotas y
  para subir el modelo finetuneado.

## Cómo ejecutarlo

1. **Crear el entorno e instalar dependencias:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configurar el token de Hugging Face.** Copia la plantilla y rellena tu token:

   ```bash
   cp .env.template .env
   # edita .env y pon: HF_API_KEY=hf_xxxxxxxx  (token con permisos de escritura)
   ```

3. **Abrir y ejecutar el notebook** `project.ipynb` (en Jupyter / VS Code) de arriba a abajo.

   Todos los ajustes (modelo base, dataset, hiperparámetros, rutas de salida) están
   centralizados en la celda **«Panel de configuración»**; modifícalos ahí si hace falta.

## Salidas

- `finetuned-model-oldbooks/` — pipeline finetuneado guardado en local.
- `imagen_antes_finetuning.png` / `imagen_despues_finetuning.png` — comparación del prompt principal.
- `comparison_images/` — las 5 imágenes por modelo de la comparación con semilla fija.
- `hf_cache/` — descargas de Hugging Face (modelos y datasets) guardadas dentro del proyecto
  para poder inspeccionarlas.

> Estas salidas, junto con `.env`, están en `.gitignore` y no se versionan.
