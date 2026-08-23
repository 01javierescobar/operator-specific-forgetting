# Backlog — Paper I (v1.1 post-arXiv)

Nada de esto bloquea la submisión a arXiv: son las mejoras experimentales que
suben el paper de "sólido y honesto" a "demostrado". Los MAJORs estructurales
del round-3 (R1 generalista) se cierran acá.

## Tareas

| # | Tarea | Qué cierra |
|---|-------|-----------|
| 1 | **TOST con presupuesto igualado** (re-entrenar un brazo con la receta del otro) | El confound operador+receta: hoy es comparación de composites |
| 2 | **Rerun n=8 seeds** (brazos wave_complex + delta_nlms) | Cierra el TOST con ~90% potencia (SD observada ≈0.018) |
| 3 | **Sweep de carga c a D fijo** (variar n en D=64/128, versión entrenada) | Convierte "consistente con c en 2 puntos" en ley de escalado demostrada |
| 4 | **Multi-codebook entrenado** (K=3–4 codebooks × 3 seeds) | La crítica single-codebook a nivel entrenado |
| 5 | **Controles del ridge probe**: held-out-key, shuffle de pares, baseline frozen-probe, test de permutación vs chance | Blindar la disociación energético-funcional contra fuga de vocabulario |
| 6 | **Nulos sistematizados**: selectividad bajo 3 convenciones (nulo=fallo / excluido / separado) | Elimina la ambigüedad del estimando |
| 7 | **Medición de ‖k‖ en entrenamiento** + trayectoria observada vs predicción escalar | Respaldo numérico directo al argumento de estabilidad §4 |
| 8 | **Perturbaciones no-colineales de clave**: ruido de fase por-coordenada, clave con dirección cambiada, ruido aditivo | Extender la celda nula más allá del gauge de fase global |

## Dependencias

- **Cluster A (sin entrenamiento, CPU local):** #5 → #6 → #7 → #8.
  Todo corre sobre checkpoints existentes.
- **Cluster B (requiere GPU):** #1+#2 son UNA campaña fusionada
  ("TOST v2": presupuestos igualados × n=8); #3+#4 son OTRA campaña
  fusionada (factorial carga × codebook × seeds).
- **Regla crítica:** Cluster A ANTES que Cluster B. Las campañas B miden su
  resultado final con el ridge probe; si #5 destapa un confound, el arreglo
  obliga a rediseñar el instrumento antes de quemar GPU.

## Plan por fases

```
Fase 1 (CPU, local):   #6 nulos -> #5 controles probe -> #7 normas de clave
                       [validar instrumentos de medición]
Fase 2 (campaña GPU):  #1+#2 fusionados  [TOST v2: igualado x 8 seeds]
                       ~4-5 h GPU secuencial / ~2.5 h con 2 workers
Fase 3 (campaña GPU):  #3+#4 fusionados  [factorial: carga x codebook x seed]
                       ~1-2 días según versión (sintética/entrenada)
Cualquier momento:     #8 perturbaciones no-colineales (probe-only)
```

## Estimaciones individuales

| # | Cómputo | Trabajo total |
|---|---------|---------------|
| 1 | ~3–4 h GPU | medio día – 1 día |
| 2 | ~4–5 h GPU sec. | ~1 día (desatendido) |
| 3 | CPU (sintética) / 8–15 h (entrenada) | medio día / 1–2 días |
| 4 | ~3–6 h cómputo | ~1 día |
| 5 | sin cómputo pesado | medio día – 1 día |
| 6 | probes CPU-minutos/checkpoint | medio día |
| 7 | checkpoint: medio día / trayectoria: ~1 día | medio día – 1 día |
| 8 | probes sobre checkpoints | ~1 día |

Backlog completo: ~1 semana efectiva (mayormente esperando GPUs).

## Recursos ya disponibles

- Harnesses MC: `tests/varL_montecarlo_scaling.py`, `tests/varL_n_sweep.py`,
  `tests/varL_decomposition.py`, `tests/varL_delta_method.py`,
  `tests/varL_lemmas.py`, `tests/varL_calibration.py`,
  `tests/codebook_ensemble_probe.py`
- Worklog de derivación: `lab/derivaciones/varL.md`
- Checkpoints entrenados servidos en `outputs/*/cache/` para todos los probes
