# Experimento 5 - AION fine-tuning parcial

## Que se reutilizo del Experimento 4
- Secciones 1-3: semillas, lectura HDF5, split oficial, normalizacion train-only, mascara a cero y augmentation solo en train.
- Integracion AION: `LegacySurveyImage`, `CodecManager`, bandas `DES-G`, `DES-R`, `DES-I`, canales g,r,i y `NUM_ENCODER_TOKENS = 600`.
- Metricas del piloto/Renzo: accuracy, precision, recall, F1, AUROC, AUPRC, Brier, log-loss, TPR0, TPR10, matriz de confusion y bootstrap.

## Que cambia en el Experimento 5
- El Experimento 4 entrenaba una MLP sobre embeddings fijos.
- Aqui se congelan todos los parametros de AION y luego se descongelan solo los ultimos bloques detectados del encoder.
- La cabeza usa `LayerNorm` porque los embeddings cambian durante el fine-tuning.

## Por que solo ultimas capas
Las primeras capas suelen capturar patrones generales; las ultimas son mas adaptables a la tarea. Asi reducimos sobreajuste y costo frente a fine-tuning completo.

## Por que LR diferenciado
La cabeza parte de cero y usa un LR normal. El backbone preentrenado usa un LR diez veces menor para no destruir representaciones astronomicas utiles.

## Por que no embeddings precalculados
Durante fine-tuning parcial los ultimos bloques de AION cambian despues de cada update. Por lo tanto los embeddings tambien cambian y deben recalcularse por batch.

## Como se evitan fugas de datos
Percentiles/media/std solo con train; augmentation solo en train; checkpoint y umbral solo con validation; test se evalua una sola vez al final.

## Como se calculan TPR0 y TPR10
Se ordena test por probabilidad descendente. TPR0 usa 0 falsos positivos. TPR10 usa maximo 9 falsos positivos, equivalente a menos de 10 FP.

## Como comparar contra AION congelado
Comparar accuracy, AUROC, AUPRC, F1, TPR0 y TPR10 junto con parametros entrenables y costo. Si la mejora es pequena, no se fuerza una conclusion: el resultado puede indicar que AION congelado ya captura casi toda la senal util.
