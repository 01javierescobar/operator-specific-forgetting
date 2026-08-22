# PROTOCOL.md - Método del laboratorio ONDA

Cambia raramente: cuando cambia el método, no cuando recalibra un umbral
(eso va a GATES.md) ni la estructura (eso va a AGENTS.md). Heredado del lab
viejo (S1-S110); diferencias marcadas con [ONDA].

## Contrato causal G0-G3 (gate obligatorio antes de cualquier entrenamiento)

| Gate | Costo | Qué mide | Stop |
|---|---|---|---|
| **G0 contrato** | segundos | shapes, invariante de sufijo futuro (cambiar suffix no cambia logits del prefijo), paridad `full==rollout` (forward token a token == forward completo), gradientes conectados | Sin G0 nada se discute |
| **G1 overfit** | <1 min | 16-64 secuencias fijas, memorización perfecta (token_accuracy 1.0) | Si no memoriza 32 seqs, no escala |
| **G2 memoria** | <5 min | delays 2/4/8/16, eval 16, curva de retención | Cae en delay 16 = no retiene |
| **G3 free decode** | <10 min | 200 seqs × 12 free steps, mide `em_free_id`, `loop_degenerate_rate`, `first_div_mean`, `avg_top1_prob` | PROBE no gate de promoción (L-013) |

Todo gate genera JSON en `outputs/<variant>/`. Sin JSON no hay resultado.

## Free-decode es PROBE, no gate de promoción (L-013)

`em_free_id` en OOD cae a 0.000 para TODA arquitectura en d=64, incluido el
transformer con skill ID perfecto. Es techo UNIVERSAL, no defecto de la arq.

- G3 mide LOOP_DEGEN como probe interno: ¿el colapso es específico de esta
  arq o universal?
- LOOP_DEGEN alto + OOD-formato bajo = defecto de la arq (intervenir).
- LOOP_DEGEN alto + OOD-formato alto = techo universal (no intervenir).
- G3 es gate de CONTRATO (paridad incremental), no de capacidad.

## Niveles de evidencia N0-N4

| Nivel | Qué | Mínimo |
|---|---|---|
| **N0** | sanity/QA: shapes, paridad, weights hash, TestDocsSync | sin N0 nada es válido |
| **N1** | quick signal: 1 seed vs baseline (gate actual), reporte taxonomía 3 ejes (GATES.md) | preliminar |
| **N2** | candidate: 3 seeds, medias ± std | mínimo para "supera/refuta" |
| **N3** | formal: 5-6 seeds + OOD battery + probes + ablaciones | marca CONFIRMED/REFUTED (vía pipeline promoción d=128) |
| **N4** | paper-ready: + intervención causal o generalización entre arq | |

Single-seed se documenta como preliminar, NUNCA como veredicto.

[ONDA] Para un paper, N2 es el mínimo publicable en claims de capacidad;
N4 es obligatorio para el claim principal (intervención causal o
transferencia entre arquitecturas). El claim del paper se declara en
`paper/CLAIM.md` junto con su nivel de evidencia objetivo.

## Regla de gasto de cómputo

1 seed hasta señal → 3 sólo con señal confirmada → 5-6 sólo si 3 confirman y
la decisión importa. **2 seeds inconsistentes = NO escalar a 5, diagnosticar.**
Un single-seed delta ±0.10 es ruido (std ~0.18 en suma EM a d=64).

## Funnel de 3 fases (anti-Kaggle-caro)

| Fase | Probe | Costo | Stop / Discrimina |
|---|---|---|---|
| **0 signal_init** | NASWOT + signal propagation + effective rank | <5 s | `slope_act > 1.5` → ABORTAR pre-entrenamiento. GDN marcado VANISHING (slope −1.507) predijo S77 MIXTO antes de entrenar. |
| **1 OOD transfer v1** | `tests/diagnostic_ood_transfer.py` con `PROTO_REGISTRY` + `--proto` | min | ÚNICO que discrimina: transformer NO-GENERALIZA vs v6 GENERALIZA. |
| **2 dyck2 OP-TIN** | `tests/common_smoke.py` con `dyck2` | min | Bucket d3+ con thr 0.40. Memoria de stack. |

Fase 0 es FILTRO DE SALUD NEGATIVO (necesario, no suficiente): descarta init
roto, NO predice escalabilidad. Falsos positivos documentados (L-028: grad
10x dispares; L-055: slope −1.915 = eje degenerado de 2 puntos, real +0.031
por bloque). Complementar con `diagnostic_early_slope.py` (S105) y reportar
con la taxonomía 3 ejes de GATES.md.

[ONDA] Los scripts de diagnóstico (signal_init, early_slope, ood_transfer,
emit_traj, lookahead_cut) viven en el lab viejo (`../modelo-2/tests/`) y se
portan cuando un prototipo los necesita. El runner nuevo de cada prototipo
debe incluir al menos la Fase 0-lite del contrato (spread de grad + slope de
varianza por bloque, L-028) como en `loopy_osc_smoke.py`.

## Pipeline de promoción d=64 → d=128 (guardarraíl anti-inversión de signo)

1. Ganador d=64: 3+ seeds con delta >= 1 std de la reference en suma EM, en
   al menos 2 tasks.
2. Mini-smoke d=128 en Kaggle: 1 seed (el ganador), ~50% épocas del preset
   smoke, 2 tasks con delta más alto + state_tracking de control, 2xT4
   (~30-45 min). Busca confirmar SIGNO, no 4/4.
3. Si signo se mantiene → smoke completo d=128 (4 tasks, 2xT4 pareadas, 3
   seeds).
4. Si pasa `recommend_kaggle` → hereda rol de candidata.

[ONDA] El régimen paper (TinyStories/MQAR paper, transformer_lit) es la
escalera final del claim: la evidencia del lab viejo (L-016/064) muestra que
a d=64 el gate mide profundidad/capacidad, no tipo de mecanismo. Ningún
claim de "novedad ondulatoria" se sostiene solo con d=64.

## Diagnóstico antes que arquitectura (L-013, lección metodológica)

Antes de atribuir un fallo a la arquitectura:
1. Diagnósticar el instrumento (correr baseline con skill ID perfecto en el
   mismo test).
2. Si el baseline también cae → es hardness del test, no defecto de la arq.
3. Si el baseline pasa → sí es defecto de la arq, intervenir.

## Baselines congelados (verificación por hash)

- `prototypes/transformer/model.py` (GPT-2 fiel, suma EM 1.534 @ d=64 cpu_quick).
- `prototypes/transformer_lit/` (port Zoology, régimen paper).
- `prototypes/kimi_kda/` (porte Kimi K3 KDA, control calibración).
- `prototypes/causal_state_lm/` (tracer bullet del contrato G0-G3).
- Verificación: `tools/check_frozen.py --strict` por SHA256. Hashes en GATES.md.
- Reglas: roles disjuntos; nunca mezclar; nunca tener un solo baseline; nunca
  mutar. Para variar → archivo nuevo (sufijo `_v2`).

## Regla del claim (anti-20.000-líneas, específica de ONDA)

- `paper/CLAIM.md` declara: (1) la afirmación novedosa, (2) el experimento
  decisivo que la mataría, (3) el nivel de evidencia objetivo (N2/N3/N4).
- Toda iteración O## arranca escribiendo su PREDICCION contra el claim en
  `logs/O##_test.md` ANTES de correr nada.
- Un resultado que no toca el claim (ni lo confirma ni lo mata) se documenta
  como iteración exploratoria y no recibe presupuesto de seeds.
- El claim cambia SOLO al cerrar una iteración que lo refuta (nunca en
  mitad de una).

## Probes bankables (los que discriminan; portar desde ../modelo-2/tests/)

- `tests/diagnostic_signal_init.py` (Fase 0 NASWOT).
- `tests/diagnostic_early_slope.py` (S105, N1 no bloqueante) — pendiente de
  loss temprana `Slope_50 = (L_0 - L_50)/(50·L_0)` (50 pasos, config FIJA
  del gate, mqar_1hop por defecto). Registro N1 en
  `outputs/<proto>/early_slope.json`, nunca gate.
- `tests/diagnostic_ood_transfer.py` con `PROTO_REGISTRY` (Fase 1).
- `tests/common_smoke.py` con `dyck2` (Fase 2, bucket d3+ thr 0.40).
- `tests/diagnostic_emit_traj.py` (métricas LOOP_DEGEN como probe, no gate).
- `tests/diagnostic_lookahead_cut.py` (formaliza trampa TF lookahead).
- `tests/diagnostic_ood_free_decode_v2.py` (probe interno, NUNCA gate de
  promoción — el antiguo `diagnostic_ood_free_decode.py` era falso-bloqueador).
- Batería VM (S101-102): length-gen + intervención + traza halt + R²;
  probe estándar para arq latentes.