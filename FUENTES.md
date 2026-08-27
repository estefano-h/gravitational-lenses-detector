# Fuentes

Referencias de modelos, datos, métodos, métricas y software usados en este proyecto de
detección de lentes gravitacionales fuertes (INF659 Deep Learning, PUCP — tesis en curso).

Este archivo cubre el proyecto completo, no solo el Experimento 5 incluido en este
repositorio: se listan también las fuentes de los experimentos comparados en
[`comparacion_experimentos_lentes_gravitacionales.md`](AION%20fine-tuning%20parcial/comparacion_experimentos_lentes_gravitacionales.md)
(baseline, AION-large, ConvNeXt V2, ViT), cuyo código no está en esta carpeta. La sección
[Dónde se usa cada fuente](#dónde-se-usa-cada-fuente) indica la correspondencia.

Los enlaces apuntan a DOI o arXiv cuando existen. Se usa formato autor-año; adaptar al estilo
que exija el documento final de tesis.

---

## 1. Modelos y pesos preentrenados

**AION-1** — modelo fundacional astronómico usado en los Experimentos 4, 5, A y B
(`polymathic-ai/aion-base`, `polymathic-ai/aion-large`).

> Parker, L., Lanusse, F., Shen, J., et al. (Polymathic AI Collaboration). 2025.
> *AION-1: Omnimodal Foundation Model for Astronomical Sciences.*
> Advances in Neural Information Processing Systems 38 (NeurIPS 2025).
> arXiv:2510.17960 — <https://arxiv.org/abs/2510.17960>

- Pesos: <https://huggingface.co/polymathic-ai/aion-base> · <https://huggingface.co/polymathic-ai/aion-large>
- Código y librería `polymathic-aion`: <https://github.com/PolymathicAI/AION>
- Proyecto: <https://polymathic-ai.org/>
- Preentrenado sobre Legacy Survey, HSC, SDSS, DESI y Gaia; variantes de 300 M a 3.1 B
  parámetros. Este proyecto usa la modalidad `LegacySurveyImage` con las bandas `DES-G`,
  `DES-R`, `DES-I`.

**ConvNeXt V2** — backbone convolucional del Experimento C (`convnextv2-tiny`, preentrenado
en ImageNet-22k).

> Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., & Xie, S. 2023.
> *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.*
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
> (CVPR 2023), 16133–16142. arXiv:2301.00808 — <https://arxiv.org/abs/2301.00808>

- Pesos: <https://huggingface.co/facebook/convnextv2-tiny-22k-224>
- Código original: <https://github.com/facebookresearch/ConvNeXt-V2>
- Antecedente directo (ConvNeXt V1): Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C.,
  Darrell, T., & Xie, S. 2022. *A ConvNet for the 2020s.* CVPR 2022. arXiv:2201.03545.

**Vision Transformer (ViT)** — backbone del Experimento 3 (`vit_base_patch16_224`,
preentrenado en ImageNet).

> Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. 2021.
> *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.*
> International Conference on Learning Representations (ICLR 2021).
> arXiv:2010.11929 — <https://arxiv.org/abs/2010.11929>

**Transformer / autoatención** — base teórica del baseline y de AION.

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
> Kaiser, Ł., & Polosukhin, I. 2017. *Attention Is All You Need.*
> Advances in Neural Information Processing Systems 30 (NeurIPS 2017), 5998–6008.
> arXiv:1706.03762 — <https://arxiv.org/abs/1706.03762>

**Baseline (CNN + autoatención)** — arquitectura de referencia del proyecto, re-implementada
y re-evaluada sobre el mismo split.

> Thuruthipilly, H., Zadrozny, A., Pollo, A., & Biesiada, M. 2022.
> *Finding strong gravitational lenses through self-attention. Study based on the
> Bologna Lens Challenge.* Astronomy & Astrophysics, 664, A4.
> DOI: [10.1051/0004-6361/202142463](https://doi.org/10.1051/0004-6361/202142463) —
> arXiv:2110.09202

- De este trabajo proviene el diseño de referencia («Lens Detector 15») que da nombre a la
  carpeta de trabajo del proyecto.

---

## 2. Datos

**Bologna Strong Gravitational Lens Finding Challenge** — origen del dataset sintético
`bologna_synthetic_v2.h5` (20 000 imágenes, 4 bandas u,g,r,i, 101×101 px).

> Metcalf, R. B., Meneghetti, M., Avestruz, C., et al. 2019.
> *The strong gravitational lens finding challenge.* Astronomy & Astrophysics, 625, A119.
> DOI: [10.1051/0004-6361/201832797](https://doi.org/10.1051/0004-6361/201832797) —
> arXiv:1802.03609

- Sitio del challenge: <http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html>
- El archivo `.h5` **no** se distribuye en este repositorio. El split oficial
  (16 000 / 2 000 / 2 000, semilla fija) y las etiquetas provienen de este dataset.

**DESI Legacy Imaging Surveys** — survey sobre el que se define la modalidad
`LegacySurveyImage` de AION y las bandas `DES-G/R/I` usadas en los experimentos con AION.

> Dey, A., Schlegel, D. J., Lang, D., et al. 2019.
> *Overview of the DESI Legacy Imaging Surveys.* The Astronomical Journal, 157(5), 168.
> DOI: [10.3847/1538-3881/ab089d](https://doi.org/10.3847/1538-3881/ab089d) — arXiv:1804.08657

**Dark Energy Camera (DECam)** — instrumento asociado a los filtros DES usados por esa
modalidad.

> Flaugher, B., Diehl, H. T., Honscheid, K., et al. 2015. *The Dark Energy Camera.*
> The Astronomical Journal, 150(5), 150.
> DOI: [10.1088/0004-6256/150/5/150](https://doi.org/10.1088/0004-6256/150/5/150) —
> arXiv:1504.02900

**ImageNet** — corpus de preentrenamiento de ConvNeXt V2 y ViT.

> Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. 2009.
> *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR 2009, 248–255.
> DOI: [10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848)

---

## 3. Métodos de entrenamiento y transferencia

**LP→FT (linear probing seguido de fine-tuning)** — esquema de dos fases del Experimento B.

> Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. 2022.
> *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.*
> ICLR 2022 (Oral). arXiv:2202.10054 — <https://arxiv.org/abs/2202.10054>

**AdamW (weight decay desacoplado)** — optimizador usado en todos los experimentos.

> Loshchilov, I., & Hutter, F. 2019. *Decoupled Weight Decay Regularization.*
> ICLR 2019. arXiv:1711.05101 — <https://arxiv.org/abs/1711.05101>

**Scheduler coseno** — decaimiento del learning rate.

> Loshchilov, I., & Hutter, F. 2017. *SGDR: Stochastic Gradient Descent with Warm Restarts.*
> ICLR 2017. arXiv:1608.03983 — <https://arxiv.org/abs/1608.03983>

**Warmup del learning rate** — fase inicial (`WARMUP_RATIO = 0.1`).

> Goyal, P., Dollár, P., Girshick, R., et al. 2017. *Accurate, Large Minibatch SGD:
> Training ImageNet in 1 Hour.* arXiv:1706.02677 — <https://arxiv.org/abs/1706.02677>

**Gradient clipping por norma** — estabilidad del entrenamiento (`GRAD_CLIP_NORM = 1.0`).

> Pascanu, R., Mikolov, T., & Bengio, Y. 2013. *On the difficulty of training recurrent
> neural networks.* ICML 2013. arXiv:1211.5063 — <https://arxiv.org/abs/1211.5063>

**Precisión mixta automática (AMP)** — usada cuando hay CUDA.

> Micikevicius, P., Narang, S., Alben, J., et al. 2018. *Mixed Precision Training.*
> ICLR 2018. arXiv:1710.03740 — <https://arxiv.org/abs/1710.03740>

**Layer Normalization** — primera capa de la cabeza MLP en el Experimento 5.

> Ba, J. L., Kiros, J. R., & Hinton, G. E. 2016. *Layer Normalization.*
> arXiv:1607.06450 — <https://arxiv.org/abs/1607.06450>

**Dropout** — regularización de la cabeza MLP.

> Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. 2014.
> *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.*
> Journal of Machine Learning Research, 15(56), 1929–1958.
> <https://jmlr.org/papers/v15/srivastava14a.html>

**Test-Time Augmentation (TTA)** — promedio sobre las 8 transformaciones diedrales, usado en
los Experimentos B, C y 3. Práctica estándar sin una fuente canónica única; puede citarse su
uso original en:

> Krizhevsky, A., Sutskever, I., & Hinton, G. E. 2012. *ImageNet Classification with Deep
> Convolutional Neural Networks.* NeurIPS 2012, 1097–1105.

---

## 4. Métricas y estadística

**Índice J de Youden** — criterio de selección del umbral de clasificación en validation.

> Youden, W. J. 1950. *Index for rating diagnostic tests.* Cancer, 3(1), 32–35.
> DOI: [10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3](https://doi.org/10.1002/1097-0142(1950)3:1%3C32::AID-CNCR2820030106%3E3.0.CO;2-3)

**Brier score** — calibración de las probabilidades predichas.

> Brier, G. W. 1950. *Verification of Forecasts Expressed in Terms of Probability.*
> Monthly Weather Review, 78(1), 1–3.
> DOI: [10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)

**Bootstrap** — intervalos de confianza al 95% de todas las métricas de test.

> Efron, B. 1979. *Bootstrap Methods: Another Look at the Jackknife.*
> The Annals of Statistics, 7(1), 1–26.
> DOI: [10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552)

**Curvas ROC y AUROC** — interpretación y buenas prácticas.

> Fawcett, T. 2006. *An introduction to ROC analysis.*
> Pattern Recognition Letters, 27(8), 861–874.
> DOI: [10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010)

**Precision-Recall frente a ROC** — justificación de reportar AUPRC junto a AUROC.

> Davis, J., & Goadrich, M. 2006. *The Relationship Between Precision-Recall and ROC Curves.*
> ICML 2006, 233–240. DOI: [10.1145/1143844.1143874](https://doi.org/10.1145/1143844.1143874)

**Nota sobre TPR0 / TPR10.** No son métricas con una fuente canónica propia: son el recall
alcanzable con 0 y con menos de 10 falsos positivos sobre el conjunto de test, definidas en
este proyecto siguiendo la lógica de priorizar pureza extrema que motiva
[Metcalf et al. 2019](#2-datos) («reducir falsos positivos será particularmente importante»).
Su definición operativa está en el `README.md` del Experimento 5.

---

## 5. Software

| Herramienta | Versión mínima | Referencia |
|---|---|---|
| PyTorch | ≥2.1 | Paszke, A., Gross, S., Massa, F., et al. 2019. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019. arXiv:1912.01703 |
| NumPy | ≥1.24 | Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 2020. *Array programming with NumPy.* Nature, 585, 357–362. DOI: 10.1038/s41586-020-2649-2 |
| pandas | ≥2.0 | McKinney, W. 2010. *Data Structures for Statistical Computing in Python.* Proc. 9th Python in Science Conf., 56–61. DOI: 10.25080/Majora-92bf1922-00a |
| Matplotlib | ≥3.7 | Hunter, J. D. 2007. *Matplotlib: A 2D Graphics Environment.* Computing in Science & Engineering, 9(3), 90–95. DOI: 10.1109/MCSE.2007.55 |
| scikit-learn | ≥1.3 | Pedregosa, F., Varoquaux, G., Gramfort, A., et al. 2011. *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830 |
| h5py / HDF5 | ≥3.8 | Collette, A. 2013. *Python and HDF5.* O'Reilly Media. <https://www.h5py.org/> |
| polymathic-aion | ≥0.1 | Ver [AION-1](#1-modelos-y-pesos-preentrenados). <https://github.com/PolymathicAI/AION> |
| huggingface_hub | ≥0.20 | <https://github.com/huggingface/huggingface_hub> |
| safetensors | ≥0.4 | <https://github.com/huggingface/safetensors> |
| timm | — | Wightman, R. 2019. *PyTorch Image Models.* GitHub. DOI: 10.5281/zenodo.4414861. Usado en los Experimentos 3 (ViT) y C (ConvNeXt V2) para adaptar el stem a 4 canales |
| Jupyter | ≥1.0 | Kluyver, T., Ragan-Kelley, B., Pérez, F., et al. 2016. *Jupyter Notebooks — a publishing format for reproducible computational workflows.* IOS Press, 87–90 |
| Google Colab | — | Entorno de ejecución con GPU usado para las corridas de AION. <https://colab.research.google.com/> |

Ver [`requirements.txt`](AION%20fine-tuning%20parcial/requirements.txt) para las versiones
exactas declaradas del Experimento 5.

---

## 6. Fuentes internas del proyecto

Artefactos generados por el propio proyecto que sustentan las cifras del documento
comparativo. **Ninguno está incluido en este repositorio salvo donde se indica**; se listan
para trazabilidad.

| Artefacto | Experimento | Estado |
|---|---|---|
| `baseline_cnn_atencion/metrics.json` | Baseline CNN + atención | Externo a este repo |
| `experimento4_aion_congelado_mlp/metrics.json` | Exp. 4 — AION-base congelado | Externo a este repo |
| `experimento5_aion_finetuning_parcial/metrics.json` | Exp. 5 — AION-base FT parcial | **Pendiente**: el notebook de este repo aún no se ejecuta |
| `pf_v2_aion_large/experimento_A/metrics.json` | Exp. A — AION-large congelado | Externo a este repo |
| `pf_v2_aion_large/experimento_B/metrics.json` (8 bloques) | Exp. B — AION-large LP→FT | Solo gráficas en este repo; falta el `metrics.json` (ver `PROVENANCE.md`) |
| `executed_pf_v2_final.ipynb` (2 bloques, reconstruido) | Exp. B — corrida de 2 bloques | Métricas reconstruidas en `2_bloques/metrics_expB_2bloques.json` |
| `pf_v2_aion_large/ablacion_bloques_ft.csv` | Exp. B — ablación 1–8 bloques | Externo a este repo |
| `experimento_C_convnextv2/seed42_all4/metrics.json` | Exp. C — ConvNeXt V2-tiny | Externo a este repo |
| `experimento3_vit/seed42_all4/metrics.json` | Exp. 3 — ViT-base | Externo a este repo |

Las advertencias de procedencia de los resultados de AION-large están en
[`resultados_experimento_B_aion_large/PROVENANCE.md`](AION%20fine-tuning%20parcial/resultados_experimento_B_aion_large/PROVENANCE.md).

---

## 7. Dónde se usa cada fuente

| Componente del repositorio | Fuentes principales |
|---|---|
| Encoder y tokenización (`AION.from_pretrained`, `CodecManager`, `LegacySurveyImage`) | Parker et al. 2025; Dey et al. 2019; Flaugher et al. 2015 |
| Dataset, split y etiquetas (`bologna_synthetic_v2.h5`) | Metcalf et al. 2019 |
| Diseño del baseline comparado | Thuruthipilly et al. 2022; Vaswani et al. 2017 |
| Descongelamiento parcial y LR diferenciado | Kumar et al. 2022; Loshchilov & Hutter 2019 |
| Scheduler coseno con warmup | Loshchilov & Hutter 2017; Goyal et al. 2017 |
| Estabilidad del entrenamiento (clipping, AMP) | Pascanu et al. 2013; Micikevicius et al. 2018 |
| Cabeza MLP (`LayerNorm`, `Dropout`) | Ba et al. 2016; Srivastava et al. 2014 |
| Métricas y umbral | Youden 1950; Brier 1950; Fawcett 2006; Davis & Goadrich 2006 |
| Intervalos de confianza | Efron 1979 |
| Experimentos C y 3 (comparación) | Woo et al. 2023; Dosovitskiy et al. 2021; Deng et al. 2009; Wightman 2019 |
| Stack de cómputo | Ver [sección 5](#5-software) |

---

## 8. Licencias de terceros

- **AION** (`polymathic-ai/aion-base`, `aion-large`): publicado bajo licencia MIT en Hugging
  Face. Este repositorio no redistribuye los pesos, solo el código que los descarga.
  Verificar los términos vigentes antes de un uso distinto al de investigación.
- **ConvNeXt V2** (`facebook/convnextv2-tiny-22k-224`): consultar la licencia declarada en su
  tarjeta de modelo en Hugging Face y en el repositorio de Meta AI.
- **ViT / timm**: consultar la licencia de `pytorch-image-models` y la de los pesos
  específicos usados.
- **Dataset Bologna**: no incluido aquí; consultar los términos de uso del Bologna Strong
  Gravitational Lens Finding Challenge.
- **Código de este repositorio**: licencia MIT, ver [`LICENSE`](LICENSE).

---

## Notas de mantenimiento

- Verificado contra la fuente original (agosto 2026): AION-1, Bologna Challenge,
  Thuruthipilly et al. 2022, ConvNeXt V2, LP→FT (Kumar et al. 2022), DESI Legacy Imaging
  Surveys. El resto son citas estándar; confirmar volumen, página y DOI contra la biblioteca
  de la PUCP antes de la entrega final de la tesis.
- Al añadir un experimento nuevo, agregar aquí sus modelos, datos y métodos, y actualizar la
  [sección 6](#6-fuentes-internas-del-proyecto) y la [sección 7](#7-dónde-se-usa-cada-fuente).
- Existe una versión BibTeX de estas referencias en [`fuentes.bib`](fuentes.bib), pensada para
  el documento de tesis.
