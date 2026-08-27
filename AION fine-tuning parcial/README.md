# Experimento 5 — AION fine-tuning parcial

Detección de lentes gravitacionales fuertes con fine-tuning parcial del modelo fundacional
astronómico [AION](https://huggingface.co/polymathic-ai/aion-base) (`polymathic-ai/aion-base`).
Trabajo del curso INF659 Deep Learning (PUCP), dentro de una tesis más amplia sobre detección
de lentes gravitacionales con deep learning.

> **Estado:** el notebook está completo y documentado, pero **no ha sido ejecutado todavía**
> en este repositorio (no tiene outputs guardados ni un `artifacts_experimento5/` generado).
> Los resultados numéricos que sí se incluyen aquí (carpeta
> `resultados_experimento_B_aion_large/`) pertenecen a un experimento relacionado pero
> **distinto** (Exp. B, AION-*large*), ejecutado en otro pipeline. Ver la sección
> [Resultados](#resultados) para el detalle.

## Contenido del repositorio

| Archivo / carpeta | Descripción |
|---|---|
| `experimento5_aion_finetuning_parcial.ipynb` | Notebook principal: carga AION-base, congela todo el encoder, descongela solo los últimos bloques, entrena una cabeza MLP end-to-end y evalúa con el protocolo anti-leakage del proyecto. |
| `explain_for_video.md` | Resumen en texto plano de la metodología (qué se reutilizó, qué cambió, por qué), pensado como guion de apoyo. |
| `comparacion_experimentos_lentes_gravitacionales.md` | Comparación de 7 configuraciones (baseline, AION-base/large congelado y fine-tuned, ConvNeXt V2, ViT) sobre el mismo dataset y split. **Versión Markdown**, con tablas legibles en GitHub. |
| `comparacion_experimentos_lentes_gravitacionales.docx` | Fuente original en Word del documento anterior. |
| `resultados_experimento_B_aion_large/` | Métricas y gráficas de Experimento B (AION-*large*, fine-tuning en dos fases, ablación de 2 vs. 8 bloques). Ver `PROVENANCE.md` dentro — incluye un problema de procedencia de datos detectado y corregido. |
| `video final Data mining/` | Video de presentación del proyecto. |

## Metodología (Experimento 5)

```text
lote H5 -> preprocesamiento/augmentation -> bandas g,r,i -> LegacySurveyImage
-> CodecManager.encode -> AION.encode (entrenable solo en los últimos bloques)
-> mean pooling -> LayerNorm + MLP -> logit binario
```

- **Reutilizado del Experimento 4** (AION congelado + MLP, no incluido en este repo):
  lectura HDF5, split oficial (train/val/test, 0/1/2), normalización calculada solo con
  train, máscara de píxeles inválidos a cero, augmentation solo en train, integración con
  AION (`LegacySurveyImage`, `CodecManager`, bandas `DES-G/R/I`, `NUM_ENCODER_TOKENS=600`), y
  el set completo de métricas (accuracy, precision, recall, F1, AUROC, AUPRC, Brier,
  log-loss, TPR0, TPR10, matriz de confusión, bootstrap).
- **Lo que cambia en el Experimento 5:** en vez de usar embeddings precalculados de un AION
  100% congelado, se descongelan solo los últimos `N_UNFROZEN_BLOCKS` bloques del encoder
  (por defecto 1) y se entrenan junto con la cabeza. Como el encoder cambia en cada
  actualización, los embeddings ya no pueden precalcularse: se recalculan por batch, y la
  cabeza usa `LayerNorm` en vez de un `StandardScaler` ajustado sobre embeddings fijos.
- **Por qué solo las últimas capas:** las primeras capas de un modelo preentrenado capturan
  patrones generales; las últimas son más específicas y adaptables a la tarea. Descongelar
  solo el final reduce sobreajuste y costo computacional frente a un fine-tuning completo
  (reservado para un Experimento 6 posterior — el código incluso levanta un error si se
  intenta forzar `ALLOW_FULL_FINETUNING=True`).
- **Por qué LR diferenciado:** la cabeza parte de pesos aleatorios y usa `HEAD_LR` normal
  (1e-3 por defecto); el backbone preentrenado usa `BACKBONE_LR = HEAD_LR * 0.1` para no
  destruir las representaciones astronómicas ya aprendidas.
- **Cómo se evitan fugas de datos (data leakage):** percentiles/media/std de normalización
  calculados solo con train; augmentation solo en train; el checkpoint y el umbral de
  clasificación (Youden) se eligen solo con validation; el conjunto de test se evalúa una
  única vez, al final.
- **TPR0 / TPR10:** se ordena el test por probabilidad descendente. TPR0 es el recall máximo
  alcanzable con 0 falsos positivos; TPR10, con como máximo 9 falsos positivos (<10 FP).
  Son las métricas más relevantes para un caso de uso de inspección visual real, donde el
  volumen de candidatas revisables es limitado.

## Cómo ejecutar el notebook

### Requisitos

```bash
pip install -r requirements.txt
```

Ver [`requirements.txt`](requirements.txt). El paquete `polymathic-aion` descarga los pesos
de `polymathic-ai/aion-base` desde Hugging Face en el primer uso (~314 M parámetros); se
recomienda GPU con al menos 8 GB de VRAM para un batch size razonable, aunque el notebook cae
automáticamente a un modo de prueba rápida (`EXP5_PRUEBA_RAPIDA=1`) si no detecta CUDA.

### Dataset

El notebook espera `bologna_synthetic_v2.h5` (dataset sintético de lentes gravitacionales,
20 000 imágenes de 4 bandas u,g,r,i, 101×101 px, con split oficial 16 000/2 000/2 000 y
semilla fija embebida). **No está incluido en este repositorio** por su tamaño; colocarlo en
la raíz del proyecto o en `data/bologna_synthetic_v2.h5`. En Google Colab, el notebook lo
busca en `MyDrive/LensDetector15/data/` y lo copia al runtime.

### Variables de entorno relevantes

| Variable | Default | Descripción |
|---|---|---|
| `EXP5_PRUEBA_RAPIDA` | `1` sin CUDA, `0` con CUDA | Si es `1`, submuestrea el dataset (160/48/48 imágenes) para validar el pipeline rápido. |
| `EXP5_BATCH_SIZE` | `4` (GPU) / `2` (CPU) | Tamaño de lote — chico porque AION participa en el forward/backward. |
| `EXP5_N_UNFROZEN_BLOCKS` | `1` | Cuántos bloques finales del encoder se descongelan. |
| `EXP5_HEAD_LR` | `1e-3` | Learning rate de la cabeza MLP. |
| `EXP5_BACKBONE_LR_MULT` | `0.1` | Multiplicador sobre `HEAD_LR` para el LR del backbone. |
| `EXP5_WEIGHT_DECAY` | `1e-4` | Weight decay de AdamW. |
| `EXP5_MAX_EPOCHS` | `10` (`2` en prueba rápida) | Épocas máximas. |
| `EXP5_PATIENCE` | `3` (`1` en prueba rápida) | Épocas sin mejora antes de early stopping. |
| `EXP5_GRAD_CLIP_NORM` | `1.0` | Norma máxima de gradiente. |
| `EXP5_WARMUP_RATIO` | `0.1` | Fracción de pasos de warmup del scheduler coseno. |

### Salidas

Al ejecutarse por completo, el notebook escribe en `artifacts_experimento5/` (no versionada,
ver `.gitignore`): `config.json`, `preprocessing_stats.json`, `training_log.csv`,
`best_model.pt`, `metrics.json`, `test_predictions.csv`, `training_curves.png`,
`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `explain_for_video.md` y
`comparison_experimento4_vs_5.csv`.

## Resultados

Este notebook produce sus propios resultados en `artifacts_experimento5/metrics.json` una
vez ejecutado (ver arriba) — **no incluidos en este repositorio**, ya que el notebook aún no
se corrió aquí.

La carpeta [`resultados_experimento_B_aion_large/`](resultados_experimento_B_aion_large/)
contiene resultados de un experimento **relacionado pero distinto**: Experimento B
(AION-*large*, fine-tuning en dos fases, ablación de 2 vs. 8 bloques descongelados), corrido
en un pipeline separado (`PF_v2`) no incluido en este repositorio. Se conservan como
referencia porque el documento comparativo los cita. **Léase el `PROVENANCE.md` de esa
carpeta antes de citar cualquier número de ahí** — durante la preparación de este repositorio
se detectó que la subcarpeta antes llamada `8_bloq` en realidad contenía una corrida piloto
de 1 bloque (120 imágenes de test), no el resultado final de 8 bloques citado en la tabla
comparativa; ya se corrigió y documentó.

Para la comparación completa entre las 7 configuraciones evaluadas en la tesis (baseline,
AION-base/large congelado y fine-tuned, ConvNeXt V2, ViT), ver
[`comparacion_experimentos_lentes_gravitacionales.md`](comparacion_experimentos_lentes_gravitacionales.md).

## Licencias de terceros

- **Modelo AION** (`polymathic-ai/aion-base`): sujeto a la licencia publicada por Polymathic
  AI en Hugging Face. Revisar sus términos antes de un uso distinto al de investigación.
  Este repositorio no redistribuye los pesos, solo el código que los descarga.
- **Dataset Bologna** (`bologna_synthetic_v2.h5`): no incluido en este repositorio; consultar
  la fuente original del Bologna Strong Gravitational Lens Finding Challenge para sus
  términos de uso.

## Licencia

El código de este repositorio se distribuye bajo licencia MIT — ver [`LICENSE`](LICENSE).
