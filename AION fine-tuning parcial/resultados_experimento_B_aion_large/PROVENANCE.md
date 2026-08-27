# Procedencia de estos resultados

**Importante:** estos artefactos **no** provienen del notebook de este repositorio
(`experimento5_aion_finetuning_parcial.ipynb`, AION-**base**, un bloque descongelado).
Pertenecen a **Experimento B** — AION-**large**, fine-tuning en dos fases (LP→FT) con
ablación del número de bloques descongelados (1 a 8) — ejecutado en un notebook/pipeline
distinto (`PF_v2`, `pf_v2_aion_large/experimento_B/`) que no está incluido en esta carpeta.
Se conservan aquí porque el documento de comparación
(`comparacion_experimentos_lentes_gravitacionales.md`) los usa como referencia frente a
Experimento 5.

## Qué carpeta es cada cosa

| Carpeta | Contenido real | Estado |
|---|---|---|
| `2_bloques/` | `metrics_expB_2bloques.json` — corrida final de 2 bloques descongelados (12-jul-2026), `n_test=2000` | **Válido**, pero reconstruido (ver nota abajo) |
| `8_bloques_final/` | Gráficas (`roc.png`, `confusion.png`, etc.) de la corrida final de 8 bloques (14-jul-2026), `n_test=2000`, AUROC=0.9133 | **Válido** — coincide exactamente con la fila "8 bloques" del documento comparativo. **Falta su `metrics.json`** (no estaba en la carpeta original; solo se recuperaron las imágenes). |
| `8_bloques_piloto_no_usar/` | `metrics.json` + gráficas de una corrida **piloto** de 1 bloque, `n_test=120`, AUROC=0.78 | **No usar como resultado final.** Estaba guardada bajo el nombre "8_bloq" en la carpeta original, lo cual era engañoso: sus números (piloto, 1 bloque, 120 imágenes de test) no tienen relación con "8 bloques". Se conserva sin borrar por transparencia, pero renombrada para que no se confunda con el resultado final. |

### Cómo se detectó el problema

Al revisar la carpeta original (`large/8_bloq/metrics.json`) contra la tabla del documento
comparativo, los números no coincidían (AUROC 0.78 vs. 0.9133 citado para "8 bloques").
Comparando las imágenes PNG de `large/8_bloq/` y de `large2/` se confirmó que:

- `large/8_bloq/` contenía en realidad una corrida piloto de 1 bloque con solo 120 imágenes
  de test (`modo: "piloto"`, `n_bloques_elegido: 1`).
- `large2/` (sin `metrics.json`, solo imágenes) contenía la corrida final real de 8 bloques
  que sí coincide con la tabla del documento comparativo.

Es decir, en algún momento el `metrics.json` de la corrida final de 8 bloques fue
sobrescrito por una corrida piloto posterior — el mismo tipo de incidente que ya está
documentado explícitamente para la corrida de 2 bloques (ver nota siguiente), pero que en
este caso no había sido señalado.

### Nota heredada sobre la corrida de 2 bloques

El `metrics.json` original de la corrida de 2 bloques fue sobrescrito en el servidor por una
corrida piloto posterior. Sus métricas se reconstruyeron a partir de las celdas de salida del
notebook ya ejecutado (12-jul-2026), preservado por separado. El checkpoint y las
predicciones crudas de esa corrida específica no son recuperables sin reentrenar desde cero.
Ver el campo `_nota_reconstruccion` dentro de `2_bloques/metrics_expB_2bloques.json`.

## Pendiente

- Reconstruir o reentrenar para obtener un `metrics.json` propio de `8_bloques_final/`
  (hoy solo hay gráficas).
- Si se vuelve a ejecutar el pipeline `PF_v2`, evitar que una corrida piloto sobrescriba el
  `metrics.json` de una corrida final — por ejemplo, escribiendo cada corrida en una carpeta
  con timestamp en vez de reutilizar siempre el mismo nombre de archivo.
