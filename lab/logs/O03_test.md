# O03_test.md - Pre-registro N1 5-task + auditoría O01d integrada

Template fijo. El runner escribe RESULTADO. El agente cierra VEREDICTO +
LECCION cuando la iteracion se cierra en `memory.md`.

## HIPOTESIS

Contra `paper/CLAIM.md` v4 (O01d). La auditoría de la especificación O03
detectó 3 defectos (verificado por script propio en
`outputs/audit/audit_o03_verification.json`, 2026-08-18):

1. **Ambigüedad del tipo de valor**: con v complejo, la fuga del canal Re
   es 0.555 ≈ ½ + (n−1)/(2D) (retiene la mitad imaginaria del ítem) y la
   predicción "wave re pierde" solo vale en ese régimen; con v REAL el
   canal complejo sigue exacto (0.000) y Re deja (n−1)/(2D) = 0.059
   (medido 0.068 @ D=128, n=16). Decisión: v real.
2. **Tres brazos, dos operadores**: wave_complex ≡ delta_forget (Clase A
   diff 0.0) → reclasificados como CONDICIÓN DE CONTROL con expectativa de
   identidad pre-registrada; el brazo que falta es la RE-PRESENTACIÓN del
   valor, que es el comparador falsable del claim de borrado.
3. **Maquinaria ρ sin poder discriminante** (NULL con claves i.i.d.) →
   reemplazada por Clase B′ = drift de fase δ, donde el proyector es
   EXACTO a todo δ (inmunidad por construcción: e^{iδ}w preserva el rango
   1-D de la clave) y la re-presentación degrada (n−1)/D + (2−2cosδ).

O03 ejecuta N1 (d=64 cpu_quick, 5-task, 3 seeds) en los brazos con v real
y valida la maquinaria de selectividad con la intervención oracle de
re-presentación sobre modelos entrenados. El gate 5-task NO aplica a P1
(ningún brazo tiene atención/RoPE; copy_reverse/mqar bajos esperados,
pre-registrado).

## PREDICCION

- **Probes sintéticos O03 (v real)**: canal complejo 0.0000; Re(·)
  (n−1)/(2D); orig_v (n−1)/D. Clase B′: re-lectura 0.0000 a todo δ ∈
  {0, 0.2, 0.4, 0.6}; re-presentación ≈ (n−1)/D + (2−2cosδ) (incrementos
  δ² exactos); delta real bajo rotación 2-D ≈ (2/D)sin²δ (robustez del
  control, no resultado). Ruido i.i.d. con FORMA CERRADA (reconciliada
  O01d, convención σ = fase/amplitud por elemento, potencia σ²):
  phase `reread = (1−e^{−σ²})² + (n−1)/D·(1−e^{−σ²})`,
  `represent = (n−1)/D + (1−e^{−σ²/2})²`;
  additive `reread = (σ²/(1+σ²))² + (n−1)/D·σ²/(1+σ²)`;
  cruce analítico σ* ≈ 0.64 rad (leak de amplitud ~35-39%) — fuera de
  régimen operativo; el probe matchea la fórmula en cada σ (no un cruce
  empírico). Fit C1: slope ≈ 1.0 con intercepto, excluyendo n/D ≥ 0.4, R²
  reportado.
- **N1 (5-task, 3 seeds, d=64)**: ambos brazos aprenden forget_retrieval
  (gate ~0.70) y state_tracking (~0.70); copy_reverse/mqar bajos en los
  DOS (sin atención/RoPE, esperado); control de identidad:
  selectividad_wave_complex ≈ selectividad_delta_forget dentro del ruido
  de seeds (TOST: |Δ| < 0.02 con IC 90%).
- **Potencia estadística (spec N1, auditoría O03)**: el contraste real es
  wave_re vs los otros dos con Δ = (n−1)/(2D) @ d=64. Spec:
  forget_retrieval con n_pairs ∈ [16, 24] (Δ = 0.117-0.180) y ≥300
  eventos FORGET evaluados por brazo/seed (potencia 90%: 160-320 eventos).
  n_pairs=8 (Δ 0.055 → ~830 eventos) NO alcanza: si la task da menos
  eventos, N1 corre pero no puede ver su efecto principal (null por
  diseño).
- **Decodificabilidad de fuga**: un decodificador lineal entrenado sobre
  M post-erase puede predecir v_borrado desde el crosstalk residual
  (head destado, L-015) → fuga-SISTEMA (EM entrenado) puede exceder la
  fuga-OPERADOR (fórmula); se mide y contrasta (figura del paper).
- **Selectividad sobre modelos entrenados** (intervención oracle):
  la maquinaria computa EM_vivo, fuga (queries a claves borradas) y
  selectividad con el borrado por re-lectura entrenado; la intervención
  re-presentación (erase con v_true inyectado) se implementa a nivel
  runner (snapshot de estado tras el token FORGET, erase algebraico
  externo, reanudación del prefill) — sin tocar el modelo.

Criterio de éxito O03: probes O03 consistentes con las predicciones de
O01d (incluyendo las formas cerradas i.i.d.) + N1 completo en los 3 brazos
con la spec de potencia (n_pairs ∈ [16,24], ≥300 eventos FORGET/brazo/
seed) + control TOST confirmado + probe de decodificabilidad ejecutado +
JSON en `outputs/` → O04 (Clase B′ δ sobre modelos entrenados) habilitado.
Si la identidad wave_complex vs delta NO se cumple (TOST falla) → la
comparación de operadores está confundida por optimización/inductive bias:
rediseñar el control antes de seguir.

## DISEÑO

- Modelo wave_mem a v REAL (O01d): `v_c = complex(v_proj(x), 0)` con
  `v_proj: D → D` (antes 2D→complejo). El resto de la arq no cambia
  (codebook fijo, escritura/erase estructurales, read_proj complex|re).
  delta_forget no cambia (ya es real).
- Probes actualizados en `tests/wave_mem_smoke.py`: `probe_canales` con
  v real y predicciones (n−1)/(2D) sin término ½; `probe_n_over_d` con
  fit log-log con intercepto excluyendo n/D ≥ 0.4 y R²; `probe_clase_b`
  (ρ) sustituida por `probe_clase_b_prime` (drift δ: re-lectura vs
  re-presentación + rotación 2-D del delta + sweep i.i.d. σ ∈
  {0.2, 0.3, 0.5}).
- N1: preset cpu_quick (d=64, n_layers=2, AdamW lr=1e-3 wd=0.01 clip=1.0),
  5-task (mqar_1hop, mqar_2hop, copy_reverse, state_tracking,
  forget_retrieval), 3 seeds (1, 2, 3), MISMA grilla para wave_complex,
  wave_re, delta_forget. Spec de potencia: forget_retrieval con n_pairs ∈
  [16, 24]; conteo de eventos FORGET por brazo/seed ≥ 300 (verificado en
  el runner antes de entrenar; si el preset da menos, subir épocas o
  batches hasta cumplir).
- Probe de selectividad O03 (runner de entrenamiento, no smoke): sobre
  modelos entrenados, dataset con queries a claves borradas y vivas;
  intervención oracle de re-presentación vía snapshot de estado + erase
  externo + reanudación (una línea de intervención, sin modificar el
  modelo); reportar EM_vivo, fuga, selectividad por seed y brazo.
- Control de identidad entrenado = TOST: |Δselectividad_wave_complex −
  Δselectividad_delta| < ε = 0.02 con IC 90% (reportar además el IC por
  seed). Si falla: atribuir a parametrización/optimización y reportar.
- Probe de decodificabilidad: regresión lineal (ridge) sobre el estado M
  post-erase (capturado en el token FORGET) → predice v_borrado; se
  contrasta con la fuga-OPERADOR de las fórmulas cerradas. Spec de
  implementación: features = Re/Im de M re-ordenado, target = v_i real;
  train/val split 80/20, sin escalamiento al head del modelo (es un
  decodificador independiente, figura de contraste).
- Presupuesto: CPU local ~30 min (N1) + probes sintéticos <2 min.
  Salidas: `outputs/wave_mem/probes.json` (regenerado) y
  `outputs/wave_mem/n1_<config>.json` (dedup por config hash).
- Pre-registrado: `recommend_kaggle` NO es regla de decisión de P1.

## RESULTADO (runner)

### Probes sintéticos (2026-08-18, `outputs/wave_mem/probes.json` regenerado)

- `probe_canales` (v real): complejo 0.000 / Re 0.0602 / orig_v 0.1049 @
  D=128, n=16 (pred 0 / 0.0586 / 0.1172). OK.
- `probe_n_over_d`: slope 1.054, R² = 0.959 (fit con intercepto, sin n/D ≥
  0.4). OK.
- `probe_clase_b_prime`: re-lectura 0.0000 a todo δ ∈ {0, 0.2, 0.4, 0.6};
  re-presentación incrementos δ² exactos (0.0196/0.0784/0.1764);
  represent δ=0 → 0.1172 (crosstalk); rotación 2-D delta 5e-5/2.3e-4/5.3e-4
  (= (2/D)sin²δ). OK.
- Sweep i.i.d. (σ por elemento, fase y aditivo): el probe matchea la
  fórmula cerrada en cada σ (phase σ=0.4: 0.0391 vs pred 0.0392; σ=0.8:
  0.2603 vs 0.2788). Cruce analítico fase σ* = 0.638 (fuga 0.151, leak
  amplitud 38.9%), aditivo σ* = 0.644 (fuga 0.12, leak 34.7%) — fuera de
  régimen operativo. Verificado por el auditor (tabla de 5 cantidades).
- Auditoría O03: **"Autorizo N1"** (2026-08-19) con 4 condiciones: (1)
  reportar conteo de eventos FORGET por brazo/seed en este log (≥300
  post-hoc); (2) reportar EM de entrenamiento por brazo junto a
  selectividad; (3) n_pairs ∈ [16,24]; (4) ridge sobre M post-erase y
  sobre el residual de lectura. Además: cerrar el fork de normalización
  del erase aditivo por escrito (hecho en CLAIM §0b: erase = proyección
  idempotente `(I − ww^H/‖w‖²)`; fase = modelo canónico O04).

### N1 (5-task cpu_quick, 3 seeds, d=64, n_pairs FR [16,24]) — completado
2026-08-19, `outputs/wave_mem/n1.json` (3230s CPU), runner
`tests/wave_mem_n1.py` + `n1_run.log`

**Entrenamiento (EM final, 3 seeds):**

| task | wave_complex | wave_re | delta_forget |
|---|---|---|---|
| mqar_1hop | 1.000 ×3 | 1.000 ×3 | 0.017-0.075 |
| mqar_2hop | 0.017-0.025 | 0.000-0.058 | 0.042 |
| copy_reverse | 0.000 ×3 | 0.000 ×3 | 0.000 ×3 |
| state_tracking | 0.307-0.340 | 0.307-0.407 | 0.320-0.407 |
| forget_retrieval | **1.000 ×3** | **1.000 ×3** | **0.000/0.020/0.000 (chance 0.025), loss NaN en s1 y s3** |

**Probe de selectividad (320 eventos FORGET por brazo/seed ≥ 300 ✓,
autocheck harness EM 1.0 = probe alive EM 1.0 en los 6 combos wave ✓):**

| brazo | sel_reread (EM_vivo − EM_fuga) | EM_fuga reread | fuga_power_reread | fuga_power_represent | sel_represent |
|---|---|---|---|---|---|
| wave_complex | 0.9719/0.975/0.9719 (mean 0.9729, sd 0.0015) | 0.025-0.0281 (≈chance) | 0.0000/0.0000 | **0.312-0.320 ≈ (n−1)/D** | 0.994-1.000 |
| wave_re | 0.9531/0.975/0.9719 (mean 0.9667, sd 0.0097) | 0.0281-0.0469 (≈chance) | 0.0000/0.0000 | **0.164-0.170 ≈ (n−1)/(2D), EXACTAMENTE la mitad del complejo** | 0.994-1.000 |
| delta_forget | — (FR no entrena) | — | — | — | — |

Δ(sel_re − sel_wc) = −0.0188/0.0/0.0 (mean −0.0063): NULL en EM (el piso
chance 0.025 domina). La diferencia de operador vive en POTENCIA:
fuga_represent del canal Re = la mitad del complejo ✓ (predicción
(n−1)/(2D) vs (n−1)/D del operador, confirmada en modelos entrenados).

**Ridge de decodificabilidad (condición 4 del auditor)**: cosine_test ≈ 0
(−0.03..+0.05) sobre M post-erase Y sobre el residual de lectura en los 6
combos wave: NINGÚN decodificador lineal recupera v_borrado → fuga-SISTEMA
= fuga-OPERADOR (sin efecto L-015 de head-que-decodifica-crosstalk a esta
dificultad). Delta s2 (no aprendió): mse_rel ~2e6 (garbage, documentado).

**Conteo de eventos (condición 1)**: 320 queries a claves borradas por
brazo/seed evaluadas post-hoc (probe 'erased'); ≥ 300 ✓ en los 6 combos
wave. **EM de entrenamiento por brazo (condición 2)**: tabla arriba
(copy_reverse 0.000 y mqar_2hop bajos en TODOS los brazos = sin
atención/RoPE, pre-registrado; state_tracking 0.31-0.41 < gate 0.70 en
todos = el gate NO es criterio de P1).

### Autopsia de delta_forget (condición bloqueante del auditor, 2026-08-19,
`tests/delta_autopsy.py` + `tests/delta_autopsy_beta.py`,
`outputs/wave_mem/delta_autopsy.json` + `delta_autopsy_beta.json`)

1. **Sweep lr × clip a n_pairs=8 (régimen más fácil)**: lr {1e-3, 5e-4,
   2.5e-4} × clip {1.0, None}, 40 épocas, 600 muestras. NINGUNA receta
   entrena FR: peak EM 0.047-0.073 en los 6 combos (= chance). Sin NaN a
   esta dificultad. → NO es "receta" (el veredicto "régimen" se mantiene).
2. **Normas de gradiente por componente**: el gradiente NO muere — k_proj
   crece 3.97 → 7.88 (época 1 → 20), v_proj 1.12 → 2.23, out_proj 1.25 →
   1.78; todos los componentes con señal sana toda la corrida. No es
   vanishing gradient.
3. **Norma de la memoria ‖S‖_F**: en el delta RANDOM (sin entrenar), UNA
   pasada de forward sobre 32 muestras → ‖S‖_F = **8×10¹¹ / 3.8×10¹²**
   (frente a ~1174 del wave entrenado ≈ n·|v|). El write correctivo
   `M += (v − k^T M)⊗k` a paso β=1 (LMS online con paso 1) DIVERGE desde
   el primer paso cuando las claves no son ortogonales: S explota a 10¹²,
   destruye el SNR del readout → el head aprende solo las marginales
   (loss 4.6 → 1.03, EM ≈ chance). El NaN a n_pairs 16-24/es 80 es el
   desbordamiento de este mismo mecanismo.
4. **Diagnóstico confirmatorio β=0.1** (mismo modelo, write correctivo a
   paso 0.1, 40 épocas, n_pairs=8): ‖S‖_F acotada (86 → 55) y **EM 1.000
   al final** (0.733 a ep 5). El punto de fallo ES la escritura (paso-1
   LMS), no la capacidad, no la receta de optimizador, no el erase.
5. **Tipo de escritura documentado en CLAIM §0** (sea cual sea el
   resultado, pidió el auditor): wave = superposición pura `M += w_pend⊗v`;
   delta = escritura CORRECTIVA `M += (v − k^T M)⊗k` (regla LMS/online-
   delta). La equivalencia Clase A (diff 0.0) cubre SOLO el erase; en N1
   wave y delta NO eran el mismo operador completo (compartían interfaz y
   erase, no la escritura).

Conclusión de la autopsia: **fallo estructural del write correctivo a
β=1** (divergencia de S), no de receta → el premio escondido del auditor
se documenta con mecanismo: mismo operador de erase, la instanciación
fasorial-compleja entrena y la real-delta no en este régimen, PORQUE la
parametrización (write superpuesto con fasores unitarios vs write LMS a
paso 1) gobierna la estabilidad de S y por tanto la optimizabilidad.
OPCIÓN B QUEDA DESCARTADA por la autopsia (el β=0.1 que sí entrena sería
un operador distinto = prototipo nuevo + re-pre-registro; el control TOST
NO resucita con β=1, que es lo pre-registrado).

### Held-out del 1.000 (ajuste 1 del auditor, 2026-08-19,
`tests/n1_heldout_check.py`, `outputs/wave_mem/n1_heldout.json`)

Eval de FR con claves/valores FRESCOS (mismo generador, seed de datos
7777) en los 6 ckpts wave: **EM = 1.000 en 6/6** (idéntico a la val
seed+100). El 1.000 NO es memorización de la distribución de eval: el
head generaliza el crosstalk decodificable (ítems vivos) — coherente con
el ridge ≈ 0 (el crosstalk es decodificable para ítems VIVOS, no para el
BORRADO).

### Gate d=128 reescrito (ajuste 2 del auditor, GATES.md)

EM saturado → los umbrales EM (0.113/0.30) están muertos como criterio en
d=128; el gate pasa a los probes (fuga_power reread 0.000 exacto; represent
(n−1)/D ± 0.02 con la mitad exacta en wave_re; ridge cosine ≈ 0; drift δ:
reread plano a todo δ ≤ 0.3, represent ≈ piso + δ²; EM en FR = sanity
check sin umbral). Tabla completa en GATES.md "Gate d=128 reescrito".

## VEREDICTO

O03 NO se cierra como estaba pre-registrado. Tres desviaciones + cierre
del bloqueo:

1. **REFUTADO (pre-registro)**: "ambos brazos aprenden forget_retrieval
   ~0.70". wave_mem lo aprende a 1.000 (mejor que el piso C1 previsto
   1−n/D ≈ 0.62: el head entrenado decodifica sobre el crosstalk; held-out
   con claves frescas: generaliza); el delta_forget NO lo aprende a d=64
   (EM ≈ chance en TODA dificultad, NaN en 2/3 seeds). Consistente con
   L-056/L-062 (regla delta con erase nativo ~0.05-0.11 en FR).
2. **CONTROL ROTO en FR + DECISIÓN DEL AUDITOR (2026-08-19)**: TOST
   wave_complex vs delta NO COMPUTABLE (delta no entrena FR). El auditor
   aprueba la **Opción A** con condición bloqueante (autopsia, RESUELTA
   arriba: fallo estructural del write β=1, no receta) y dos ajustes:
   held-out del 1.000 (✓ generaliza) y gate d=128 en probes (✓ reescrito
   en GATES.md). Contraste P1 = **wave_re vs wave_complex**. Opción B
   descartada por la autopsia (β=0.1 = otro operador, no resucita el TOST
   con β=1).
3. **CONFIRMADO (operador, en modelos entrenados)**: fuga_power del
   residual represent = (n−1)/D (complejo) vs (n−1)/(2D) (Re) — la mitad
   exacta, reproduciendo la predicción O01d a nivel entrenado; erase
   reread fuga 0.0000 en ambos canales (v real). El contraste EM de
   selectividad es NULL (piso chance domina) — la métrica que discrimina
   es fuga_power (figura del paper: predicción cerrada confirmada en
   sistema entrenado).
4. La intervención represent es MÁS selectiva que reread (0.994-1.000 vs
   0.973-0.975): el erase oracle elimina el ítem por completo; el reread
   entrenado deja un residuo que el head decodifica a nivel chance+ε.
5. **Premio escondido (contribución N1, mecanismo documentado)**: mismo
   operador de erase, la instanciación fasorial-compleja entrena y la
   real-delta no en este régimen — el operador es necesario pero no
   suficiente; la parametrización del write (superpuesto acotado vs LMS a
   paso 1 divergente: ‖S‖ 10¹² vs ~1200) es lo que lo hace optimizable.

Estado: **O04 AUTORIZADO** por el auditor ("en cuanto la autopsia y el
held-out check estén commiteados"). Pre-registro O04: fuga_power como
métrica primaria, δ inyectado como rotación de fase común en la clave al
momento del FORGET (modelo de reloj con drift), ley δ² pre-registrada para
represent, contraste P1 = wave_re vs wave_complex, EM = sanity check.

## LECCION

- La selectividad en EM es una métrica de piso: a fuga < umbral de
  decodificación del head, TODO operador "aprueba" (EM_fuga ≈ chance). La
  discriminación operador-específica vive en fuga_power (|r|²/|v|²), que
  SI reproduce (n−1)/D vs (n−1)/(2D) en modelos entrenados. Para O04:
  reportar EM y fuga_power, y diseñar el contraste sobre fuga_power.
- Un brazo que no entrena NO es un control: la expectativa de identidad
  (Clase A, diff 0.0) verificada en síntesis no garantiza que el brazo
  aprenda la task en el régimen del harness. Verificar entrenabilidad
  (EM de entrenamiento) del control ANTES de depender de él para un
  contraste (precedente: histórico L-056/L-062 ya lo advertía para FR).
- La intervención oracle (represent) es instrumento más fino que la
  comparación de brazos entrenados: misma arquitectura, una línea de
  diferencia, residual medible exactamente. Es el instrumento primario
  para O04.
- NaN en entrenamiento = registro de fallo, no excepción: el runner debe
  capturarlo (guard de NaN en ckpt + autocheck harness-vs-probe EM) para
  no producir selectividades fantasma (incidente: primera corrida de N1
  midió EM contra ANSWER_ID — target mal derivado del formato FR con
  valor en x — y dio sel=0.0 en todos los brazos; el autocheck lo habría
  detectado).
- Un "1.000" sospechosamente perfecto NO se reporta sin held-out: EM
  saturado exige claves/valores frescos (misma distribución, otra seed)
  para distinguir generalización de memorización de la distribución de
  eval. Coste: minutos con ckpt ya entrenado.
- Cuando un brazo no entrena una task, la autopsia (sweep de receta +
  normas de gradiente + norma del estado) resuelve "régimen vs receta"
  con evidencia, no con opinión: el gradiente sano + ‖S‖ 10¹² en UNA
  pasada delata el mecanismo (write LMS paso-1 divergente) que el sweep
  de lr/clip jamás habría encontrado (β=0.1 entrena a 1.000).

