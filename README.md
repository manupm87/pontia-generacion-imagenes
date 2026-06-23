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

## Notebooks del proyecto

El repo tiene **tres** notebooks, pensados para ejecutarse en este orden:

| Notebook | Qué hace | Salida principal |
|---|---|---|
| **`project.ipynb`** | *Finetuning completo* de la UNet sobre el dataset. | Pipeline en `finetuned-model-oldbooks/` + repo [`...oldbookillustrations`](https://huggingface.co/manupm87/finetuned-sd-1.4-oldbookillustrations) |
| **`project_lora.ipynb`** | Entrena una **LoRA** para el mismo estilo (mucho más ligera y rápida). | Adaptador en `lora-oldbooks/` + repo [`...oldbookillustrations-lora`](https://huggingface.co/manupm87/finetuned-sd-1.4-oldbookillustrations-lora) |
| **`project_comparison.ipynb`** | **Solo compara** (no entrena): genera una rejilla original vs. finetuning vs. LoRA con los mismos 5 prompts y semilla. | `comparison_finetuning_vs_lora.png` |

`project_comparison.ipynb` necesita que los otros dos se hayan ejecutado antes (usa sus
modelos locales; si no existen, los descarga de Hugging Face). El resto de este README se
centra en `project.ipynb`; los conceptos de configuración (panel, *trigger* de estilo,
*safeguards*) son equivalentes en los tres.

## Resultados

Rejilla generada por `project_comparison.ipynb` con los **mismos 5 prompts y la misma
semilla** en todos los modelos. De arriba a abajo: modelo **original sin *trigger***, original
**con *trigger***, **finetuning completo** con *trigger* y **LoRA** con *trigger*.

![Comparación original vs. finetuning vs. LoRA](comparison_finetuning_vs_lora.png)

Tanto el finetuning como la LoRA llevan claramente la imagen al estilo de grabado /
ilustración de libro antiguo, mientras que el modelo original (con o sin *trigger*) mantiene
un aspecto mucho más fotográfico.

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

## Hiperparámetros clave: batch size y épocas

Dos de los ajustes del panel de configuración controlan directamente el entrenamiento:

### `batch_size` (por defecto: `6`)

Es el **número de imágenes que se procesan juntas** en cada paso antes de actualizar los
pesos. El dataset tiene 4154 imágenes, así que con `batch_size = 6` cada época son
`4154 / 6 ≈ 693` pasos.

- **Más grande** → gradientes más estables (promedio sobre más ejemplos) y entrenamiento más
  rápido en tiempo de reloj, pero consume **más memoria de GPU** y da **menos
  actualizaciones** de pesos por época (lo que, con un *learning rate* fijo, puede ralentizar
  el aprendizaje).
- **Más pequeño** → cabe en GPUs con menos VRAM y hace más actualizaciones por época, pero los
  gradientes son más ruidosos.

`6` es un valor equilibrado: cabe de sobra en la GPU usada (RTX 5090) entrenando la UNet a
512x512, y mantiene los gradientes razonablemente estables. **Es un ajuste de
memoria/estabilidad, no un mando de calidad**, así que no hace falta tocarlo salvo que falte
o sobre VRAM.

### `num_epochs` (por defecto: `5`)

Es el **número de pasadas completas** sobre todo el dataset. Con 5 épocas son
`693 × 5 ≈ 3465` pasos de entrenamiento (~10 min/época en la RTX 5090, ~50 min en total).

- **Muy pocas épocas** → el modelo apenas capta el estilo (*underfitting*): las imágenes
  «después» se parecen demasiado a las de «antes».
- **Demasiadas épocas** → *overfitting* del estilo: el modelo empieza a ignorar el prompt y
  todo tiende a la misma ilustración de libro antiguo, perdiendo fidelidad al texto.

> **Ojo:** la *loss* de difusión es muy ruidosa por diseño (sube y baja entre pasos), así que
> **no sirve** para decidir cuándo parar. La forma correcta de juzgarlo es **mirar las
> imágenes** de las comparaciones (pasos 8 y 9).

### ¿Hay que cambiarlos?

Para esta entrega **se pueden dejar tal cual** (`batch_size = 6`, `num_epochs = 5`): producen
buenos resultados en la GPU de referencia. Como guía para ajustarlos según lo que veas en las
comparaciones:

- Si el modelo finetuneado **ignora el prompt** o todas las imágenes salen iguales → baja a
  **2–3 épocas**.
- Si el **estilo es demasiado débil** → mantén 5 épocas (o apóyate en el *trigger* de estilo /
  baja el *learning rate*) antes que subir más las épocas.
- Cambia `batch_size` solo por motivos de **memoria/velocidad**: súbelo si te sobra VRAM (y
  considera subir un poco el *learning rate*), bájalo si te quedas sin memoria.

## Salidas

- `finetuned-model-oldbooks/` — pipeline finetuneado guardado en local (`project.ipynb`).
- `lora-oldbooks/` — adaptador LoRA (`pytorch_lora_weights.safetensors`) y su model card (`project_lora.ipynb`).
- `imagen_antes_finetuning.png` / `imagen_despues_finetuning.png` — comparación del prompt principal.
- `lora_imagen_antes.png` / `lora_imagen_despues.png` — comparación del prompt principal (LoRA).
- `comparison_images/` — las 5 imágenes por modelo de la comparación con semilla fija (`project.ipynb`).
- `lora_comparison_images/` — las 5 imágenes original vs. LoRA con semilla fija (`project_lora.ipynb`).
- `comparison_finetuning_vs_lora/` + `comparison_finetuning_vs_lora.png` — rejilla original vs.
  finetuning vs. LoRA (`project_comparison.ipynb`).
- `hf_cache/` — descargas de Hugging Face (modelos y datasets) guardadas dentro del proyecto
  para poder inspeccionarlas.

> Estas salidas, junto con `.env`, están en `.gitignore` y no se versionan.
