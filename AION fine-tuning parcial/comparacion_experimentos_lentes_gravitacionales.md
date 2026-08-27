# Comparación de arquitecturas para detección de lentes gravitacionales

**Baseline · AION-base · AION-large (LP→FT) · ConvNeXt V2-tiny · ViT-base**

Sebastian Hernández — Pontificia Universidad Católica del Perú (PUCP)
Tesis · Detección de lentes gravitacionales con deep learning · Julio 2026

> Versión en Markdown de `comparacion_experimentos_lentes_gravitacionales.docx`, generada
> para que las tablas se lean bien en GitHub. El `.docx` original se conserva como fuente.

## 1. Objetivo y alcance

Este documento compara siete configuraciones evaluadas sobre el mismo dataset sintético de
lentes gravitacionales (Bologna, `bologna_synthetic_v2.h5`: 20 000 imágenes de 4 bandas
u,g,r,i, split 16 000/2 000/2 000 con semilla fija), con el fin de contrastar el costo
computacional frente a la ganancia en desempeño de detección al escalar de un modelo
compacto entrenado desde cero a modelos fundacionales (AION) y a arquitecturas
preentrenadas más compactas (ConvNeXt V2, ViT).

El baseline (CNN + autoatención, Thuruthipilly et al.) fue re-evaluado a través del mismo
chasis (PF_v2) sobre el conjunto de test de 2000 imágenes, de modo que sus métricas son
directamente comparables con las del resto de modelos.

**Nota metodológica importante:** no todos los modelos ven la misma información de entrada.
AION (Exp. 4, 5, A, B) usa solo 3 de las 4 bandas (g,r,i) a resolución nativa 101×101; el
baseline, ConvNeXt V2 y ViT usan las 4 bandas (u,g,r,i), y ConvNeXt/ViT además redimensionan
a 224×224 por requerirlo su preentrenamiento en ImageNet. Estas asimetrías se retoman en la
sección 6.

## 2. Modelos comparados

- **Baseline:** red convolucional con autoatención (~3.19 M parámetros), entrenada desde
  cero sobre las 4 bandas (u,g,r,i), siguiendo el diseño de Thuruthipilly et al.
- **Exp. 4 — AION-base congelado + MLP:** encoder AION-base (≈314 M parámetros) 100%
  congelado como extractor de características; cabeza MLP (Linear–ReLU–Dropout–Linear)
  entrenada sobre el embedding, bandas g,r,i, pooling mean, variante de flujo normalizado.
- **Exp. 5 — AION-base, fine-tuning parcial:** mismo encoder, con el último bloque del
  encoder descongelado (7 279 209 parámetros entrenables, 2.31% del total); cabeza MLP con
  tasas de aprendizaje diferenciadas (cabeza 1e-3, backbone 1e-4). *(Este es el experimento
  del notebook de este repositorio; ver [README.md](README.md) — no fue re-ejecutado para
  este documento, ver nota en esa sección.)*
- **Exp. A — AION-large congelado + MLP:** encoder AION-large (≈860 M parámetros)
  congelado; ablación de pooling (mean / max / mean+max) y de flujo (crudo / normalizado)
  elegida por AUROC de validación; ganó la variante normalizado·mean.
- **Exp. B — AION-large, fine-tuning parcial LP→FT:** esquema de dos fases (Kumar et al.
  2022) — primero la cabeza sobre embeddings cacheados (fase LP), luego se descongelan los
  últimos bloques del encoder (8 bloques, elegidos por ablación entre 1 y 8) con tasas
  diferenciadas (cabeza 1e-4, backbone 1e-5); evaluado también con Test-Time Augmentation
  (TTA, promedio de 8 transformaciones diedrales). *(Resultados en
  [`resultados_experimento_B_aion_large/`](resultados_experimento_B_aion_large/), con notas
  de procedencia en su `PROVENANCE.md`.)*
- **Exp. C — ConvNeXt V2-tiny, fine-tuning parcial:** backbone convolucional preentrenado en
  ImageNet-22k, adaptado a 4 canales inflando el stem (promedio de pesos RGB); bandas
  u,g,r,i, resolución 224×224 (redimensionada desde 101×101). Se descongelan 2 de sus 4
  etapas — en la práctica ~95% de los parámetros (26 628 482), casi un fine-tuning completo,
  no parcial en sentido estricto. AdamW, 20 épocas, batch 32. Evaluado también con TTA.
- **Exp. 3 — ViT-base, fine-tuning parcial:** Vision Transformer (`vit_base_patch16_224`,
  preentrenado en ImageNet), adaptado a 4 canales vía timm (`in_chans=4`, infla el patch
  embedding promediando pesos RGB); bandas u,g,r,i, resolución 224×224. Se descongelan los
  últimos 4 de 12 bloques + norm final + head (28 353 793 parámetros, ≈33% del total) —
  calibrado deliberadamente para un número absoluto de entrenables similar a ConvNeXt,
  permitiendo una comparación arquitectónica más limpia. AdamW con LR diferenciado (head
  3e-4, backbone 3e-5), scheduler coseno con warmup, batch 16 con AMP, 20 épocas con early
  stopping. Evaluado también con TTA (solo AUROC disponible).

## 3. Tabla comparativa de métricas

Todas las métricas provienen de `metrics.json` generado automáticamente al final de cada
notebook, sobre el mismo conjunto de test (2000 imágenes) y con intervalos de confianza al
95% por bootstrap. Los valores con TTA se muestran entre paréntesis cuando están disponibles;
ninguna otra métrica con TTA fue calculada para Exp. B ni Exp. 3, salvo el AUROC.

| Métrica | Baseline (CNN+atención) | Exp. 4 AION-base congelado | Exp. 5 AION-base FT parcial | Exp. A AION-large congelado | Exp. B AION-large FT parcial · 8 bloques | Exp. C ConvNeXt V2-tiny FT ~95% | Exp. 3 ViT-base FT ~33% |
|---|---|---|---|---|---|---|---|
| AUROC (test) | 0.9786 | 0.8110 | 0.8915 | 0.8467 | 0.9133 (TTA: 0.9391) | 0.9801 (TTA: 0.9850) | 0.9688 (TTA: 0.9738) |
| AUPRC (test) | 0.9829 | 0.8350 | 0.9067 | 0.8722 | 0.9278 | 0.9843 | 0.9743 |
| Accuracy (umbral Youden) | 0.9345 | 0.738 | 0.814 | 0.768 | 0.8385 | 0.940 | 0.9165 |
| Precision | 0.9637 | 0.787 | 0.848 | 0.846 | 0.8636 | 0.9527 | 0.9298 |
| Recall (sensibilidad) | 0.903 | 0.652 | 0.765 | 0.655 | 0.804 | 0.926 | 0.9010 |
| F1-score | 0.9324 | 0.713 | 0.804 | 0.738 | 0.8327 | 0.9391 | 0.9152 |
| TPR @ 0 FP (pureza extrema) | 0.710 | 0.110 | 0.209 | 0.202 | 0.419 | 0.739 | 0.551 |
| TPR @ ≤10 FP (de 1000 negativos) | 0.828 | 0.286 | 0.419 | 0.396 | 0.522 | 0.877 | 0.735 |
| Brier score (menor mejor) | 0.0504 | 0.1771 | 0.1387 | 0.1605 | 0.1167 | 0.0486 | 0.0656 |
| Log-loss (menor mejor) | 0.1734 | 0.5262 | 0.4392 | 0.4892 | 0.3741 | 0.1885 | 0.2232 |
| Parámetros totales | ≈3.19 M | ≈314.45 M | ≈314.45 M | ≈859.65 M | ≈859.65 M | ≈28.03 M | ≈86.00 M |
| Parámetros entrenables | ≈3.19 M (100%)† | No registrado | 7 278 209 (2.31%) | 262 657 (0.03%) | 101 189 121 (11.76%) | 26 628 482 (≈95%) | 28 353 793 (≈33%) |
| Bloques / etapas descongeladas | — | 0/24 (congelado) | 1/24 | 0/24 (congelado) | 8/24 | 2/4 etapas‡ | 4/12 bloques‡ |
| Épocas de entrenamiento | No reportado | No confirmado | No confirmado | 38 (early stopping) | 22 (fase FT) | 20 (presupuesto)§ | 19 de 20 (early stopping) |
| Tiempo de entrenamiento | No reportado | No registrado | 56.0 min (0.93 h) | 25.7 min (0.43 h) | 201.1 min (3.35 h) | 20.4 min (0.34 h) | 13.9 min (0.23 h) |

† Baseline entrenado desde cero (sin encoder preentrenado): se asume 100% de parámetros
entrenables por diseño, ya que ningún componente está congelado; no es un campo explícito de
su `metrics.json`.

‡ ConvNeXt V2 cuenta "etapas" (4 en total) y ViT cuenta "bloques" (12 en total); estas
unidades no son directamente comparables entre sí ni con los 24 bloques del encoder de
AION-large — se muestran para referencia de la fracción del modelo ajustada, no para
comparar cantidades absolutas.

§ Para ConvNeXt V2 solo se confirmó el presupuesto configurado (20 épocas); no se confirmó si
el entrenamiento usó early stopping ni en qué época quedó el mejor checkpoint — a diferencia
de Exp. A, B y 3 (ViT), donde sí está documentado. Verificar en el `metrics.json` de
`experimento_C_convnextv2` (campo `mejor_epoca` o equivalente) o en `experiment_log.csv` para
completar este dato con precisión.

## 4. Análisis y discusión

### 4.1 Tendencia con el fine-tuning

En ambos tamaños de AION, el fine-tuning parcial superó de forma consistente a usar el
encoder congelado: +0.080 de AUROC en AION-base (0.811 → 0.891) y +0.067 en AION-large
(0.847 → 0.913, o +0.092 con TTA hasta 0.939). El ajuste de un número reducido de bloques del
encoder aporta señal real, no ruido — en ambos casos la fase de fine-tuning superó de forma
clara a la línea base de linear probing.

### 4.2 Tendencia con el tamaño del modelo

Comparando los encoders congelados entre sí (Exp. 4 vs Exp. A), pasar de AION-base a
AION-large mejoró el AUROC de 0.811 a 0.847 (+0.036). Comparando los fine-tuned entre sí
(Exp. 5 vs Exp. B), la mejora fue de 0.891 a 0.913 sin TTA (+0.022) o hasta 0.939 con TTA
(+0.048). El mayor tamaño del modelo ayuda, pero con rendimientos decrecientes frente al
aumento de ~2.7× en parámetros totales (314 M → 860 M) y de ~3.6× en tiempo de entrenamiento
(56 min → 201 min).

### 4.3 Costo computacional

El costo crece de forma no lineal con la capacidad adaptada dentro de la familia AION: Exp.
4 y Exp. A (congelados) requieren solo la extracción de embeddings y el entrenamiento de una
cabeza ligera (25.7 min en el caso de A); Exp. 5 y Exp. B (fine-tuning) exigen
backpropagation a través de parte del encoder, multiplicando el tiempo por 2-8× (56 min y
201 min respectivamente).

El contraste con ConvNeXt V2 y ViT es marcado: ambos son modelos casi un orden de magnitud
más pequeños en total (28 M y 86 M parámetros, frente a los ~860 M de AION-large) y, pese a
ajustar una fracción mucho mayor de sus parámetros (~95% y ~33% respectivamente, frente al
2.99%-11.76% de AION-large), entrenan en una fracción del tiempo: 20.4 min (ConvNeXt) y 13.9
min (ViT), frente a los 149-201 min de Exp. 5/B. El costo de AION no proviene solo del
porcentaje ajustado, sino del tamaño absoluto del modelo que hay que propagar en cada paso.

### 4.4 Régimen de pureza extrema (TPR0 / TPR10)

Las métricas de TPR a 0 y a ≤10 falsos positivos son las más relevantes para un caso de uso
de inspección visual real, donde el volumen de candidatas revisables es limitado. Ningún
modelo AION supera el ~42% de recall en el régimen más estricto (TPR0); Exp. B es el mejor de
esa familia (TPR0 = 0.419, TPR10 = 0.522), pero queda muy por debajo tanto del baseline
(TPR0 = 0.710, TPR10 = 0.828) como de ConvNeXt V2, que es el modelo con mejor comportamiento
en este frente de todo el estudio (TPR0 = 0.739, TPR10 = 0.877 — y 0.809/0.903 con TTA). ViT
queda en una posición intermedia (TPR0 = 0.551, TPR10 = 0.735), por delante de todo AION pero
por detrás del baseline y de ConvNeXt.

Este patrón — ConvNeXt > baseline > ViT > AION en el régimen de mayor pureza — es consistente
con la idea de que el sesgo inductivo convolucional (localidad, equivarianza a traslaciones)
encaja mejor con la morfología de arcos de lente que una arquitectura de atención pura, que
necesita más datos o más capas ajustadas para compensar esa falta de sesgo geométrico.

### 4.5 Ablación del número de bloques descongelados (Exp. B)

La ablación del número de bloques descongelados en AION-large (de 1 a 8) mostró una
tendencia monótonamente creciente en el AUROC de validación: 0.8238 (1 bloque), 0.8357 (2),
0.8513 (3), 0.8648 (4), 0.8715 (5), 0.8767 (6), 0.8827 (7) y 0.8856 (8). Se eligió la
configuración de 8 bloques por ser la de mayor AUROC de validación; al reentrenarla con el
presupuesto completo de épocas alcanzó un AUROC de test de 0.9133 (0.9391 con TTA). La curva
sigue creciente en el extremo (7 → 8 bloques aún aporta +0.003 en validación), por lo que
descongelar el encoder completo podría rendir algo más, a costa de acercarse a un fine-tuning
total que difumina la etiqueta de «ajuste parcial».

La siguiente tabla documenta explícitamente la mejora de la corrida original de 2 bloques
(12-jul, preservada a partir de las celdas de salida del notebook ejecutado, ya que su
`metrics.json` en disco fue sobrescrito accidentalmente por una corrida piloto posterior)
frente a la ablación ampliada de 8 bloques (14-jul):

| Métrica | 2 bloques (original, 12-jul) | 8 bloques (ampliado, 14-jul) | Δ |
|---|---|---|---|
| AUROC (test) | 0.8964 | 0.9133 | +0.0169 |
| AUROC con TTA | 0.9214 | 0.9391 | +0.0177 |
| AUPRC | 0.9121 | 0.9278 | +0.0157 |
| Accuracy | 0.817 | 0.8385 | +0.022 |
| Precision | 0.870 | 0.8636 | −0.006 |
| Recall | 0.745 | 0.804 | +0.059 |
| F1-score | 0.803 | 0.8327 | +0.030 |
| TPR @ 0 FP | 0.336 | 0.419 | +0.083 |
| TPR @ ≤10 FP | 0.409 | 0.522 | +0.113 |
| Brier score | 0.1280 | 0.1167 | mejora |
| Log-loss | 0.4014 | 0.3741 | mejora |
| Parámetros entrenables | 25 690 881 (2.99%) | 101 189 121 (11.76%) | +75.5 M |
| Tiempo de entrenamiento | 149.4 min | 201.1 min | +51.7 min |

El `test_loss` de la corrida de 2 bloques no fue impreso como campo separado en el notebook
original; se asume igual al log-loss (0.4014), consistente con la relación observada entre
ambos campos en la corrida de 8 bloques.

### 4.6 Comparación con arquitecturas preentrenadas más compactas (ConvNeXt V2 y ViT)

Incorporando ConvNeXt V2 y ViT, el ranking completo por AUROC (sin TTA) queda: ConvNeXt V2
(0.9801) > Baseline (0.9786) > ViT (0.9688) > AION-B, 8 bloques (0.9133) > AION-5 (0.8915) >
AION-A (0.8467) > AION-4 (0.8110). Con TTA, ConvNeXt V2 (0.9850) amplía su ventaja sobre el
baseline, y ViT (0.9738) se acerca pero no llega a superarlo.

El hallazgo más relevante del estudio completo es que ConvNeXt V2 — un modelo ~30× más
pequeño que AION-large y ~9× más rápido de entrenar que Exp. B — es, hasta la fecha, el mejor
modelo evaluado en el proyecto, superando incluso al baseline entrenado desde cero. Esto
indica que la ventaja no proviene simplemente de "modelo compacto vs. modelo fundacional
grande", sino de qué tan bien encaja el sesgo inductivo de la arquitectura (y su
preentrenamiento) con la tarea: un backbone convolucional preentrenado en imágenes naturales,
adaptado con fine-tuning agresivo, superó tanto a un modelo entrenado desde cero como a los
modelos fundacionales astronómicos evaluados.

Con presupuesto de entrenables comparable (~26-28 M), ConvNeXt superó a ViT en todas las
métricas, especialmente en pureza extrema (TPR0 0.739 vs 0.551 sin TTA). Esto refuerza la
lectura de la sección 4.4: para esta tarea, el sesgo inductivo convolucional parece pesar más
que el número de parámetros ajustados.

## 5. Conclusiones preliminares

- El fine-tuning parcial, incluso de un solo bloque, aporta una mejora sustancial y
  consistente sobre usar AION congelado, en ambos tamaños de modelo evaluados.
- AION-large aporta una mejora modesta sobre AION-base, que no escala proporcionalmente con
  el aumento de tamaño (2.7×) ni de tiempo de cómputo (3.6×). Ampliar la ablación de bloques
  de 2 a 8 mejoró el resultado (+0.017 AUROC sin TTA), a costa de ~4× más parámetros
  entrenables y ~35% más tiempo.
- El régimen de pureza extrema (TPR0/TPR10) sigue siendo el punto más débil de todos los
  modelos AION evaluados, y debe reportarse junto al AUROC para no sobrestimar su utilidad
  práctica.
- El baseline compacto (~3.19 M parámetros, entrenado desde cero) supera a todos los modelos
  AION evaluados, incluido el mejor de ellos (Exp. B, 8 bloques: 0.9133 sin TTA, 0.9391 con
  TTA). Esto indica que, frente a AION, adaptar un modelo fundacional de cientos de millones
  de parámetros no compensa el costo frente a un modelo compacto entrenado desde cero.
- Sin embargo, el mejor modelo global del estudio es ConvNeXt V2-tiny (AUROC 0.9801 sin TTA,
  0.9850 con TTA), que supera incluso al baseline — y lo hace siendo ~9× más rápido de
  entrenar que el mejor AION. ViT queda muy cerca del baseline (0.9688 / 0.9738 con TTA) pero
  sin superarlo. El orden final por desempeño es: **ConvNeXt V2 > Baseline > ViT > AION** (en
  cualquiera de sus variantes).

## 6. Limitaciones y trabajo futuro

- La ablación de bloques descongelados en Exp. B se amplió a {1…8}; la curva sigue creciente
  en el extremo (7 → 8 bloques), por lo que descongelar el encoder completo (fine-tuning
  total) es la extensión natural restante.
- Asimetrías de entrada no controladas entre familias de modelos: AION usa 3 bandas (g,r,i) a
  resolución nativa 101×101, mientras que baseline, ConvNeXt y ViT usan 4 bandas (u,g,r,i) —
  y estos dos últimos además redimensionan a 224×224. No puede descartarse que parte de la
  ventaja de ConvNeXt/ViT/baseline sobre AION provenga de esta banda adicional o del cambio
  de resolución, no solo de la arquitectura; una corrida de control con AION recibiendo las 4
  bandas (o con ConvNeXt/ViT restringidos a g,r,i) permitiría aislar el efecto.
- El gap ConvNeXt-vs-baseline (y ViT-vs-baseline) se basa en una sola semilla de entrenamiento
  por modelo; dado que las diferencias en algunos casos son pequeñas (p. ej. ViT 0.9688 vs
  baseline 0.9786), se recomiendan corridas multi-semilla antes de afirmar significancia.
- Los parámetros entrenables exactos de Exp. 4 no quedaron registrados en su `metrics.json`;
  se recomienda añadir ese campo en futuras corridas para comparaciones de costo completas.
- El `metrics.json` original de Exp. B (2 bloques) fue sobrescrito en el servidor por una
  corrida piloto posterior; sus métricas se reconstruyeron a partir de las celdas de salida
  del notebook ejecutado (12-jul), preservado por separado. El checkpoint y las predicciones
  crudas de esa corrida específica no son recuperables sin reentrenar desde cero. *(Ver
  también [`resultados_experimento_B_aion_large/PROVENANCE.md`](resultados_experimento_B_aion_large/PROVENANCE.md)
  para un incidente análogo detectado en la corrida de 8 bloques.)*
- Solo se calculó el AUROC con TTA para Exp. B y Exp. 3 (ViT); el resto de la suite de
  métricas con TTA (AUPRC, Brier, log-loss, etc.) requeriría re-inferencia desde
  `test_predictions` con la función `evaluate_full()` estandarizada — sí disponible de forma
  completa para Exp. C (ConvNeXt).
- Evaluación realizada sobre dataset sintético (Bologna); la generalización a datos de
  surveys reales (Legacy Survey, HSC) queda pendiente.
- No se ha calculado aún el bootstrap pareado de ΔAUROC entre modelos (disponible en el
  chasis del notebook PF_v2, función `bootstrap_pareado_auroc`) para determinar significancia
  estadística de las diferencias reportadas, incluida la de ConvNeXt frente al baseline.

**Fuentes:** `baseline_cnn_atencion/metrics.json` · `experimento4_aion_congelado_mlp/metrics.json`
· `experimento5_aion_finetuning_parcial/metrics.json` · `pf_v2_aion_large/experimento_A/metrics.json`
· `pf_v2_aion_large/experimento_B/metrics.json` (8 bloques) · `executed_pf_v2_final.ipynb`
(2 bloques, reconstruido) · `pf_v2_aion_large/ablacion_bloques_ft.csv` ·
`experimento_C_convnextv2/seed42_all4/metrics.json` · `experimento3_vit/seed42_all4/metrics.json`

*Ninguna de estas fuentes (salvo la de Exp. 5) está incluida en este repositorio; se listan
tal como aparecen en el documento original para trazabilidad.*
