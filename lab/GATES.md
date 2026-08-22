# GATES.md - Thresholds y configuración del gate (LAB ONDA)

Cambia al recalibrar un umbral. Definición de niveles y método -> PROTOCOL.md.
Estructura y prohibiciones -> AGENTS.md. Heredado del lab viejo (S1-S110);
hashes recalculados al clonar (idénticos al original, verificado bit-a-bit).

## Gate de razonamiento 5-task (preset cpu_quick, d=64)

| Sub-task | Mide | Umbral | Rol en cpu_quick |
|---|---|---|---|
| `mqar_1hop` | retrieval asociativo 1-salto (induction heads, cross-chunk) | EM >= 0.85 | techo / sonda asintótica (informativo, NO bloqueante) |
| `mqar_2hop` | retrieval encadenado 2-saltos (chain vía M cross-chunk) | EM >= 0.80 | techo / sonda asintótica (informativo, NO bloqueante) |
| `copy_reverse` | manipulación simbólica / mantenimiento de orden (reverse cross-chunk) | EM >= 0.90 | BLOQUEANTE |
| `state_tracking` | actualización de estado secuencial (assign/goto) | EM >= 0.70 | BLOQUEANTE |
| `forget_retrieval` | composición con borrado intermedio (store + FORGET `k_i` + retrieve par NO borrado) | EM >= 0.70 | techo / sonda asintótica (informativo, NO bloqueante) |

A cpu_quick el umbral mqar es un PISO DE TECHO, no una barra alcanzable:
ninguna arquitectura lo pasa a este presupuesto, incluido el baseline
(transformer 0.058; 0.9986 solo en régimen paper con 32.7M tokens, L-016).
Ver "Regla de decisión N1 en cpu_quick" abajo: solo `copy_reverse` y
`state_tracking` deciden descarte en N1.

### `forget_retrieval` (task #5, gate crítico DIAGNÓSTICO)

- En cpu_quick es NO bloqueante (sonda de techo): nadie alcanza 0.70 a este
  presupuesto (GDN2 con erase nativo: 0.113). Su valor es DIAGNÓSTICO: es la
  única task cuyo pase indica capacidad de borrado nativo (la tesis del lab).
- Falla aislada en ella NO descarta; "acumula pero no borra" se lee como
  debilidad estructural candidata, no como veredicto.

- Vocab MQAR, marker `FORGET_ID` = alias de `PICK_ID=6`.
- Formato: `BOS [k1 v1 ... kn vn] SEP FORGET k_i SEP QUERY k_j ANSWER v_j`
  con `j != i`.
- Anti-atajo: `n_pairs >= 5` + permutación random sobre `max(N_KEYS, N_VALS)`.
- `include_forget=False` = legacy 4-tasks (sin gate crítico).

### Presets cpu_quick (seqs cortas, CPU i3 8GB)

- mqar: `n_pairs_range=(5,9)` (seqs 16-24, 1-2 chunks con `chunk_size=16`).
- copy_reverse: `min_len,max_len=(20,22)` (3 chunks).
- state_tracking: `n_events=(6,10)` (seq 32-44).

## Regla de decisión `recommend_kaggle`

```
recommend_kaggle = (passed >= pass_threshold) AND (passes_baseline >= 2)
```

- `passes_baseline[t] = own_em >= transformer_em - 0.0` (estricto, sin margen).
- `pass_threshold = max(3, n_total - 1)`:
  - 4 tasks (legacy): `pass_threshold = 3`.
  - 5 tasks (default, con forget_retrieval): `pass_threshold = 4`.

Sin snapshot `outputs/transformer/benchmark.json` → regla simple:
`passed >= pass_threshold`.

### Regla de decisión N1 en cpu_quick (de facto formalizada, S105)

La barra mecánica de arriba en cpu_quick es FILTRO DE TECHO, no criterio de
promoción: a d=64 el gate mide profundidad/capacidad, no tipo de mecanismo
(L-031), y el baseline solo discrimina con copy_reverse (L-032). La lectura
N1 formal es:

- **Bloqueantes (sólo estas deciden descarte):** `copy_reverse` EM >= 0.90
  y `state_tracking` EM >= 0.70, SIEMPRE con paridad G0 estricta
  (`full == rollout` + invariante de sufijo + gradientes conectados).
- **Sondas de techo (informativas, NO bloqueantes):** `mqar_1hop`,
  `mqar_2hop`, `forget_retrieval`. Un fallo aislado en ellas NO descarta; un
  pase o mejora vs baseline es señal fuerte y alimenta la lectura de
  patrones + la selección de tasks del mini-smoke d=128.
- **Pase relativo:** `passes_baseline >= 2` con Δ >= 0 (sin margen) contra
  el snapshot congelado `outputs/transformer/benchmark.json`.
- Los umbrales absolutos de mqar/forget cobran vida recién en d=128 / régimen
  paper (transformer_lit: 0.9986 EM @ 32.7M tokens, L-016). Allí la barra
  mecánica de `recommend_kaggle` vuelve a ser el criterio real.

### Taxonomía de reporte N1 (3 ejes, S106)

Todo resultado N1 se reporta con 3 ejes por task — Mecánica (E1), Salud
init (E2), Control relativo (Δ vs baseline) — nunca como gate binario.
E2 es FILTRO DE SALUD NEGATIVO (necesario, no suficiente): descarta init
roto, NO predice escalabilidad (L-028, L-055; slopes tempranos ~constantes
entre arquitecturas a d=64, S105).

| E1: Mecánica (bloqueantes) | E2: Salud init | Control Δ vs baseline | Clasificación / Acción |
|---|---|---|---|
| PASS (copy >= 0.90 Y st >= 0.70) | PASS | Δ >= 0 | **Candidato Sólido:** mini-smoke d=128. Celda SIN precedente a d=64 cpu_quick (ver nota). |
| PASS | FAIL (abort Fase 0 / vanishing) | Δ >= 0 | **Artefacto Numérico:** gate y sonda se contradicen (L-028); revisar grad por bloque (L-055) antes de escalar. |
| FAIL | PASS | Δ ≈ 0 (baseline también cae) | **Régimen Asfixiado / Indeterminado:** la task no diagnostica mecánica a d=64 (transformer mqar 0.058, L-016); NO descarta. |
| FAIL | PASS | Δ < 0 en task donde el baseline pasa (copy_reverse) | **Mecánica Rota:** falla donde el baseline tracciona → descarte (wave_iq 0/5). |
| Cualquiera | FAIL (abort slope_act > 1.5) | Cualquiera | **Descarte Inmediato** (L-017): no gastar entrenamiento. |

Nota empírica (L-024/040/056): a d=64 la conjunción copy >= 0.90 ∧ st >= 0.70
nunca ocurrió — copy lo logran las familias atención (transformer, loopy,
ternario); st las delta-rule (KDA 0.70, GDN2 0.713) con copy < 0.90. La fila
1 se espera recién en d=128 / régimen; cpu_quick es pre-filtro, no prueba de
capacidad. Lectura agregada del perfil: sección "Lectura de patrones".

### Snapshot activo del baseline (para `passes_baseline`)

Se regenera en O00 (seed 42, cpu_quick, d=64, L2) con
`python tests/transformer_smoke.py --save-baseline`:
`outputs/transformer/benchmark.json` — suma EM 1.534 (copy_reverse 1.000,
state_tracking 0.367, mqar_1hop 0.058, mqar_2hop 0.042, forget_retrieval 0.067).
Con él, la regla estricta `passes_baseline >= 2` está operativa. Los
checkpoints del transformer viven en `outputs/transformer/cache/` (los probes
OOD los reutilizan sin reentrenar).

### Lectura de patrones

Lectura agregada del perfil (taxonomía 3 ejes arriba) + patrones por task:

- Pasa threshold + baseline → subir al pipeline d=128.
- Falla threshold → NO subir a Kaggle; iterar arq o híperes en local.
- 0/n vs transformer → correr `diagnostic_<arch>.py` antes de archivar.
- Mejora sólo mqar_1hop, falla 2hop → retrieval directo pero no composición.
- Mejora copy_reverse pero falla mqar → bias de longitud, no induction heads.
- Falla sólo forget_retrieval → acumula pero no borra (debilidad estructural);
  candidatear mecanismo con erase nativo.
- Regresión en cualquier task vs baseline anterior → revisar modificación.

## Apples-to-apples (obligatorio)

Mismo `--smoke` preset, mismo `--d_model`, misma `--seed`, mismas épocas (o
sin override). Seed distinta sólo para estimar varianza.

## Config del modelo FIJO

- `d_model=64` (cpu_quick), `d_model=128` (smoke/full).
- `n_layers=2`, `n_heads=4` (transformer), `ff_mult=2`.
- `forget_factor=0.98` (proto).
- Vocabulario reducido por task (89/19/17), NO comparte las 71 del pipeline.
- Optimizador: AdamW `lr=1e-3`, `weight_decay=0.01`, `clip=1.0`.
- Marker de respuesta: `ANSWER_ID` (id=8 en vocab local).

## Mini-smoke d=128 (paso 2 del pipeline promoción)

1 seed (ganador d=64), ~50% épocas del preset smoke, 2 tasks con delta
más alto + state_tracking de control, 2xT4 (~30-45 min). Busca confirmar
SIGNO, no 4/4. Si signo se mantiene → smoke completo d=128 (4 tasks, 2xT4
pareadas, 3 seeds).

## Gate d=128 reescrito (decisión del auditor, O03-N1; los probes cargan la ciencia)

EM quedó saturado en 1.000 en los brazos wave → los umbrales EM originales
(0.113 / 0.30) están MUERTOS como criterio en d=128. El gate pasa a los
probes de operador (fuga_power, ridge, drift), pre-registrado:

| Criterio d=128 | Umbral |
|---|---|
| `fuga_power` reread, canal complex, drift clave | 0.000 (exacto; invarianza del proyector, O04b verificado 0.0000 en 3 seeds a todo δ) |
| `fuga_power` reread, canal complex, drift estado | `2(1−cosδ)(1+c)` (ley del auditor, O04b verificado: 0.3213-0.3261 vs 0.3175 @ δ=0.5) |
| `fuga_power` reread, canal re, drift clave | `sin⁴δ + c(1−cos2δ)/4` (O04b verificado: 0.0876-0.0885 vs 0.0869 @ δ=0.5) |
| `fuga_power` reread, canal re, drift estado | `(1−cosδ)² + c(1−cosδ)` (O04b verificado: 0.0499-0.0510 vs 0.0513 @ δ=0.5) |
| `fuga_power` represent, canal complex | `c + 2(1−cosδ)` (agnóstico a variante; O04: 0.549-0.562 vs 0.542 @ δ=0.5) |
| `fuga_power` represent, canal re | `c/2 + (1−cosδ)²` (incrementos exactos; piso +10% por elipticidad del crosstalk entrenado) |
| ridge sobre residual | `cosine ≈ 0` (fuga-sistema = fuga-operador) |
| EM en FR | sanity check solamente, SIN umbral |

El 2×2 (O04b, verificado en 3 seeds de modelos entrenados + verificación
independiente del auditor con decodificación): **solo UNA celda tiene
borrado verdadero — clave/complex, donde el residual es EXACTAMENTE cero**
(no pequeño: nada que decodificar). Las otras tres degradan a copia-
decodificable: el residual es una copia estructurada del valor borrado y la
fuga en potencia la SUBESTIMA en >1 orden de magnitud (clave/re decodifica
al 43% @ δ=0.2 con fuga 0.018; estado/complex 1.000 @ δ=0.5; estado/re
0.748). Teorema de garantía condicional: el borrado exacto es propiedad del
proyector con clave fresca — garantía ABSOLUTA bajo drift de reloj,
copia-decodificable bajo drift de estado. En wave_mem v1 (codebook fijo)
ninguno de los dos drifts ocurre naturalmente: el 2×2 es un stress test que
delimita dónde vive la garantía, no un modo de fallo operativo. Convencion
de medicion FIEL = Re truncado en wh Y en el readout de la query (lo que ve
el head entrenado). Convencion del auditor (Re solo en wh, readout complejo)
reportada en la verificacion sintetica: clave/re = sin²δ + c/2 (0.2884 vs
0.2868-0.289); su B-Re 0.0870 RETIRADO por el auditor (era un hibrido sin
celda fisica limpia).

## Gate O05 (mini-smoke d=128, adiciones del auditor, 2026-08-19)

Al pre-registro O05 se agregan 3 criterios (logs/O05_test.md):

1. **EM_erased por celda con guarda de residual nulo** (junto a
   |med−pred| < 0.02 en fuga): se reporta EM solo cuando ‖residual‖ >
   ε·‖v_erased‖; si no, se reporta "residual nulo" (más fuerte que
   cualquier EM bajo; evita el artefacto argmax 0/0 → índice 0 = ítem
   borrado, EM espurio 0.88-0.95 en la simulación del auditor).
   Predicción pre-registrada: clave/complex = residual nulo; las otras
   tres celdas muestran el gradiente de decodificación (dirección).
2. **C1 explícito a c emparejada**: las leyes son función de c = (n−1)/D,
   no de D. Criterio: |ley_d128(c) − ley_d64(c)| < 0.02 con el MISMO c
   (probe d=64 re-ejecutado con n_pairs [32,48] → c=0.3047 = c de O05).
3. **TOST reportado tal cual** con 5 seeds: si el IC 90% sigue cruzando
   ε = 0.02, "casi-equivalente" va al paper sin redondear.

Métrica de fuga O04: POOLED (Σ\|r\|²/Σ\|v\|²), no media de ratios — la cola
pesada de ‖v‖² del modelo entrenado (p90/p50 ≈ 5.6) infla la media de
ratios (exceso espurio ~0.026 que desaparece con la métrica pooled).

### Estado del gate O05 tras la ejecución (2026-08-19, kernel v5 COMPLETE)

- **VALIDADO**: 4/4 criterios del 2×2 + 3 adiciones del auditor cerrados
  (detalle: `logs/O05_test.md` RESULTADO/VEREDICTO, `memory.md` O-008).
- max|med−pred| de las 4 leyes = 0.0049 (5 seeds × 4 celdas × 6 δ, criterio
  0.02); C1 c emparejada 4/4 (diff ≤ 0.0064); clave/complex = residual nulo
  en 6 δ × 5 seeds; TOST casi-equivalente (IC90 [−0.0206, 0.0144], cruza
  ε=0.02 por 0.0006, reportado tal cual).
- Los umbrales de este gate quedan CONGELADOS como están: son el criterio
  que habilita la promoción del claim de ONDA (tag de cierre / paper). No
  recalibrar sin una auditoría nueva.
- Presupuesto de entrenamiento calibrado POR FAMILIA (lección O05): wave
  60ep×600 (satura EM 1.000), delta_nlms 150ep×1200 (receta n1).

Contexto N1 que lo motiva: wave_complex y wave_re entrenan FR a EM 1.000 en
6/6 (verificado con claves frescas seed=7777, held-out, no memorización);
delta_forget no entrena FR a d=64 (chance a n_pairs 8 y 16-24; autopsia:
el write correctivo β=1 diverge — `‖S‖_F ~ 10^12` en una pasada — no es
receta: sweep lr×clip falla; con β=0.1 el mismo modelo entrena a 1.000).

## Baselines congelados (SHA256, clonados del lab viejo)

| Archivo | Rol | Suma EM (d=64 cpu_quick) | SHA256 |
|---|---|---|---|
| `prototypes/transformer/model.py` | smoke gate (GPT-2 fiel, Radford 2019) | 1.534 | `2E59EA0FF985BB87FB00716D649F3725546075F55BB22F0ED07B2A035B7EEA99` |
| `prototypes/transformer_lit/model.py` | régimen paper (Zoology, MQAR) | 0.7382 medio (sanity propio) | `D709D30E99FC334034F239A66C6DDE17506BEB02E303C02E1FD82E4083A17E8C` |
| `prototypes/kimi_kda/model.py` | calibración (arquitectura publicada Kimi K3-mini) | 1.52 (S79 heredado; gate NO recorrible en CPU, loop KDA Python) | `515F6CC87D405BC7817DE0734D07F3DEB6363490ECC089243D2EBCF77FBFABD9` |
| `prototypes/causal_state_lm/model.py` | tracer bullet (contrato G0-G3, GRU minimal) | — | `D318E26C4DF67813C127BE53E99A8CBE2357C13480EE3A21752E192E5FB59571` |

Los 4 baselines son 100% standalone (no importan `_pieces/`); mutar `_pieces/`
NO rompe sus hashes. `kimi_kda` redefine `RMSNorm`/`ShortConv`/`SwiGLUExpert`
inline (no consume `rmsnorm.py`/`swiglu.py`).

Verificación: `python tools/check_frozen.py --strict` por SHA256.
Hashes completos (post-normalización LF, `.gitattributes` con `eol=lf`):
identicos a los congelados en el lab viejo (clon bit-a-bit verificado).

El JSON `prototypes/_pieces/FROZEN_HASHES.json` cubre las piezas de
`_pieces/` clonadas. Las piezas se leen en binario puro (sin normalización);
el `.gitattributes` garantiza LF al checkout.

## Free-decode (ver PROTOCOL.md)

`em_free_id` OOD es techo UNIVERSAL (L-013). NO es gate de promoción, es
probe interno. Usar `loop_degenerate_rate` como probe de diagnóstico.