# MEMORIA DEL LABORATORIO ONDA

> Tronco consolidado (≤300 líneas). Detalle: `logs/O##_test.md`; mapa
> 1-línea por iteración en `COMPENDIO.md`; protocolo/gates en
> `PROTOCOL.md`/`GATES.md`; estructura en `AGENTS.md`. Herencia: lab
> `../modelo-2` (S1-S110, cerrado con tag `lab-radioactivo-v1`).

## Síntesis (lo que dicta el siguiente experimento)

El lab viejo cerró con una ley que gobierna CUALQUIER propuesta ondulatoria:
**el direccionamiento es del operador, no del régimen** (L-056/L-064). El
régimen (32.7M tokens, d=128+) compra retrieval SOLO a la atención (0.9986);
la regla delta no NI con régimen (0.0033); la heterodina I/Q no da retrieval
(mqar 0.008); el oscilador coRNN da estabilidad de evaluación, no capacidad
(st techo 0.55, gate 0.70; acumulador ciego al orden). Dos familias
ondulatorias murieron (wave_iq S104, loopy_osc S109). ONDA nace con meta
distinta: **novedad paper-able, no mejorar el transformer**. El claim vive en
`paper/CLAIM.md`; toda iteración O## se diseña contra él.

## L-000 — Lección raíz (heredada)

TF y free decode son capacidades distintas: EM alto en gold no prueba
generación (v6 0.85 TF / 0.000 free; transformer con skill perfecto también
cae → cuello UNIVERSAL a d=64). **Implicación:** la arq nace causal con
contrato `prefill/decode_step`; lo que no toca la transición generativa es
parche.

## Ley del binding (L-056, L-064) — el muro de la familia ondulatoria

- **L-056**: a d=64, ningún operador de binding discrimina retrieval:
  heterodina I/Q 0.008, fase-codificada KDA 0.058, atención 0.058 en
  mqar_1hop; delta con erase nativo 1/5, st 0.713 (único PASS), mqar 0.075,
  forget 0.113. La separación se compra con dimensión/régimen (L-016), no
  con el operador.
- **L-064**: el MISMO régimen paper del transformer de literatura (32.7M
  tokens) con regla delta: EM 0.0033 vs 0.9986. El delta aprende lenguaje
  (loss 9.01→1.82) pero no retrieval. **El régimen compra retrieval solo a
  la atención — ley operador-específica.**

## O-001..O-005 — Auditorías O01b/O01c/O01d y prototipos O02 (verificados)

- **O-001** (O01b, verificado) La ley SNR ∝ D/(2n) de C1 SOLO vive en
  geometría matricial C^{D×D}: en vectorial la SNR es ~1/(n−1) y plana en D
  (0.35/0.36/0.34 para D=64/128/256, n=4); en matricial crece ∝ D (33/50/126).
  Fijar geometría ANTES del diseño: en C^D con `w∘v` el crosstalk crece con
  la señal a la misma tasa y la capacidad colapsa.
- **O-002** (O01b, verificado) "Borrar por antifase" == restar `w⊗v` a
  precisión de máquina (0.0 rel.). En geometría VECTORIAL el erase por
  re-lectura (`w_k∘(conj(w_k)∘M)`) borra TODA la memoria (residual 0.0);
  en matricial deja el ítem al nivel del crosstalk. Erase aprendido =
  puerta aprendida (auto-refutación). → erase α=1 analítico con v re-leído.
- **O-003** (O01b) El piso de crosstalk domina el borrado, no la fidelidad:
  el residual post-erase ≈ (n−1)/2 en unidades de señal por elemento.
  Métrica primaria para claims de borrado = SELECTIVIDAD
  (EM_vivo − tasa_de_fuga), EM absoluto secundario. Comparadores con la
  MISMA interfaz (señal FORGET_ID idéntica) o el resultado es artefacto de
  interfaz: el ancla delta 0.113 viene de régimen gated+aprendido; el brazo
  justo es delta+FORGET_ID con β=1.
- **O-004** (O01c + O02, verificado) La "diferencia ondulatoria" NO es el
  erase (con ‖w‖²=D, `M − w⊗(w^H M)/D ≡ (I − uu^H)M ≡` proyección delta
  β=1, diff 0.0 exacta — teorema de linaje). Vive en el CANAL DE CLAVE
  (fijo unitario vs aprendido) y la PROYECCIÓN DE LECTURA: Re(·) deja la
  mitad imaginaria del ítem (fuga 0.543 vs 0.000 complejo @ D=128 n=16).
  Ley del olvido: fuga(orig_v) ≈ (n−1)/D — función de n/D, no de D
  (0.099/0.103/0.135 @ n/D=0.125 en D=64/128/256; techo n/D=0.5 →
  0.51–0.66). Clase B sintética = NULL: wave inmune por construcción
  (codebook exacto, fuga 0.0000 a todo ρ), delta sensible 2º orden
  (~0.003); ambos agnósticos a ρ con claves i.i.d. → la sensibilidad a
  correlación exige estructura aprendida (Clase C / O04). Prototipos
  wave_mem (C^{D×D}, claves fijas, α=1, read_proj complex|re) y
  delta_forget (vanilla, β=1) con interfaz estructural idéntica: G0 y
  Fase 0-lite PASS en ambos (sufijo bit-exacto, full==rollout, grads).
  Erase del modelo = proyección idempotente: 2º erase no acumula.
  In-place en estado (setitem sobre M/r tensor) rompe el grafo en loops
  secuenciales → estado por listas + delta out-of-place (lección técnica).
- **O-005** (O01d, auditoría O03 verificada, `audit_o03_verification.json`)
  El tipo de valor es parte del claim: con v complejo el canal Re deja
  ½ + (n−1)/(2D) (0.56); con v REAL la tabla es limpia: complejo 0.000,
  Re (n−1)/(2D) = 0.059 (medido 0.068), orig_v (n−1)/D = 0.117 (medido
  0.128) @ D=128 n=16. Drift δ (Clase B′): re-lectura 0.0000 EXACTA a
  todo δ (e^{iδ}w preserva el rango 1-D del proyector); re-presentación
  = (n−1)/D + (2−2cosδ) (incrementos δ² exactos). Análogo real = rotación
  2-D (2/D)sin²δ ≈ 5e-5–5e-4; drift de fase complejo no existe en R → el
  delta se compara por corrupción de entrada (O04). Ruido i.i.d.: la
  inmunidad exacta es ESPECÍFICA del drift de fase; bajo i.i.d. el
  proyector gana solo en régimen moderado y PIERDE a σ=0.5 (cruce ≈
  S/(D+S)); formas cerradas verificadas: phase `reread = (1−e^{−σ²})² +
  (n−1)/D·(1−e^{−σ²})`, `represent = (n−1)/D + (1−e^{−σ²/2})²`; additive
  `reread = (σ²/(1+σ²))² + (n−1)/D·σ²/(1+σ²)`; cruce σ* ≈ 0.64 rad FUERA
  de régimen operativo (reloj real σ ≪ 0.1 rad). Decisiones integradas:
  v real, wave_represent = intervención oracle en eval, wave_complex vs
  delta_forget = control con expectativa de identidad, ρ retirada del
  sintético, P1 reformulado. Spec N1 (auditoría O03): contraste = wave_re
  vs los otros dos, Δ = (n−1)/(2D); FR n_pairs ∈ [16,24] y ≥300 eventos
  FORGET/brazo/seed; control por TOST (|Δselectividad| < 0.02, IC 90%);
  probe de decodificabilidad de fuga. Autorizada con 4 condiciones (n_pairs
  ∈ [16,24], conteo FORGET post-hoc, EM por brazo, ridge sobre M y
  residual). Fork de normalización del erase CERRADO (CLAIM §0b): erase =
  proyección idempotente `(I − ww^H/‖w‖²)`. N1 corrió en el tronco.

- **O-006** (O03 N1, 5-task cpu_quick, 3 seeds, d=64; `n1.json` +
  `logs/O03_test.md` RESULTADO) wave_mem (complex Y re) aprende
  forget_retrieval a **EM 1.000 en 6/6 combos** con n_pairs [16,24]
  (n/D ≤ 0.38 — por ENCIMA del techo EM 1−n/D≈0.62 previsto por C1: el
  head entrenado decodifica el crosstalk). El **delta_forget NO aprende
  FR a d=64** (EM ≈ chance 0.02-0.06 a CUALQUIER dificultad, NaN en 2/3
  seeds) — consistente con L-056/L-062. Selectividad (320 eventos
  FORGET/brazo/seed ≥ 300 ✓, autocheck harness-vs-probe 1.0 ✓):
  sel_reread wave_complex 0.9729±0.0015 vs wave_re 0.9667±0.0097 — NULL
  en EM (piso chance 0.025 domina); el contraste de operador vive en
  POTENCIA: fuga_power del residual represent = (n−1)/D ≈ 0.315 (complejo)
  vs (n−1)/(2D) ≈ 0.166 (Re) — la mitad EXACTA, predicción O01d
  reproducida en modelos entrenados; erase reread = 0.0000 en ambos
  canales (v real). Represent > reread en selectividad (0.994-1.0 vs
  0.973-0.975): el erase oracle elimina el ítem por completo. Ridge de
  decodificabilidad: cosine ≈ 0 (fuga-SISTEMA = fuga-OPERADOR, sin
  efecto L-015). **Control ROTO en FR**: TOST wave_complex vs delta NO
  COMPUTABLE (delta no entrena FR) → per el pre-registro, rediseño del
  control ANTES de O04 — escalado al auditor (opciones: A = delta FALLIDO
  en FR documentado, contraste = wave_re vs wave_complex, recomendada; B =
  delta con gate aprendido = prototipo nuevo + re-pre-registro; C = menor
  dificultad = inútil, no aprende ni a (5,9)). O04 BLOQUEADO hasta la
  decisión.

- **O-006b** (O04, `outputs/wave_mem/o04.json`, 3 seeds) Clase B′ drift δ
  sobre modelos entrenados (fuga pooled; la media de ratios infla ~0.026
  por cola pesada): reread-complex 0.0000 plano (P1 no dañado); represent-
  complex c + 2(1−cosδ) (0.549-0.562 vs 0.542 @ δ=0.5); **reread-re =
  sin⁴δ + c(1−cos2δ)/4 — la enmienda "reread plano" era falsa para Re**;
  represent-re c/2 + (1−cosδ)² (piso +10% elipticidad). Side arm
  delta_nlms: ENTRENA FR (1.000/0.993/0.993), ‖S‖ acotada (teorema
  β‖k‖² < 2); TOST casi-equivalente (Δ=0.0146 < ε en media, IC90 excede
  0.008). Detalle: COMPENDIO/GATES/logs/O04_test.md.

- **O-007** (O04b, `outputs/wave_mem/o04b.json`, 3 seeds, replay externo
  exacto + verificación sintética 2 grillas × 2 convenciones) Reconciliación
  2×2 del auditor (canal {complex,re} × drift {clave,estado}): **las 4
  leyes cerradas CONFIRMADAS en modelos entrenados, desvío < 0.01 en 6 δ ×
  3 seeds** — clave/complex 0.0000 plano (3/3 seeds, todo δ); estado/complex
  2(1−cosδ)(1+c) (0.3213-0.3261 vs 0.3175 @ δ=0.5); clave/re
  sin⁴δ + c(1−cos2δ)/4 (0.0876-0.0885 vs 0.0869); estado/re
  (1−cosδ)² + c(1−cosδ) (0.0499-0.0510 vs 0.0513). Semántica de estado
  fijada: solo el ítem olvidado con phasor en M; el erase lee con la clave
  drifteada (drift cancelado, wh = v + e^{−iδ}c̃/D) y remueve con la clave
  limpia. Convención del auditor (Re solo en wh, readout complejo):
  clave/re = sin²δ + c/2 EXACTA (0.2868 vs 0.2884); estado/re =
  2(1−cosδ) + c(3−2cosδ)/2 (ley nueva, derivada y verificada). **Framing
  corregido por el auditor (verificación independiente con decodificación):
  solo UNA celda tiene borrado verdadero — clave/complex (residual
  EXACTAMENTE cero); las otras tres degradan a copia-decodificable y la
  fuga en potencia subestima la fuga funcional en >1 orden de magnitud
  (clave/re decodifica 43% @ δ=0.2 con fuga 0.018). Teorema de garantía
  condicional: garantía absoluta bajo drift de reloj, copia-decodificable
  bajo drift de estado. En wave_mem v1 (codebook fijo) ningún drift ocurre
  naturalmente: el 2×2 es stress test, no modo de fallo operativo.**
  B-Re 0.0870 RETIRADO por el auditor (híbrido sin celda física limpia).
  Gate d=128 = 2×2 (GATES.md) + 3 adiciones del auditor para O05 (EM con
  guarda de residual nulo, C1 a c emparejada, TOST tal cual). EM
  kind='erased': a δ=0 la clave fue borrada → chance es lo correcto
  (sanity por fuga, no por EM). Backlog O07: drift de estado realista
  δ(t) = ω·lag (creciente por evento).

- **O-008** (O05, `outputs/onda_o05_kaggle/onda/outputs/wave_mem/o05.json`,
  kernel javierezcobar/onda-o05 v5, Kaggle 2xT4, 5 seeds, 19510s) **GATE
  d=128 CONFIRMADO**: 15/15 combos EM 1.000 (wave 60ep x 600 satura;
  delta_nlms necesitó la receta n1 150ep x 1200 — a 60ep x 600 quedó
  0.07-0.98). max|med−pred| de las 4 leyes = **0.0049** (criterio 0.02).
  **EM con guarda: clave/complex = residual nulo en 6 δ × 5 seeds**
  (borrado exacto = propiedad del proyector); clave/re decodifica
  creciente con δ (0.06→1.00) = copia-decodificable; estado ~0 (heads no
  decodifican estado, framing §0c). **C1 c emparejada (0.3047): 4/4 celdas
  diff ≤ 0.0064** — las 4 leyes son las mismas en dos escalas. **TOST
  casi-equivalente** (mean_delta −0.0031, IC90 [−0.0206, 0.0144], cruza
  ε=0.02 por 0.0006; selectividad wave_complex 0.972-0.991 vs delta_nlms
  0.963-0.997; delta EM_alive 1.000 en 5/5). Lecciones: el presupuesto
  justo se calibra POR FAMILIA de arquitectura (saturación temprana ≠
  para delta); bug de substring en filtros de path ('re' in p matchea
  'complex' → slot re cargó ckpt complex y rompió C1 en v4) → filtrar por
  segmento de ruta. **El lab tiene su paper** (eje del olvido
  operador-específico, 4 leyes cerradas en dos escalas, condición de
  estabilidad LMS, garantía condicional de borrado como teorema).

## Insights negativos heredados (una línea por lección, solo las activas)

- **L-001/002** TF != free decode; causalidad observable (sufijo futuro no
  cambia logits; forward == rollout); contrato antes de medir.
- **L-003..L-005** Features estáticas en head, SS parcial, repetition
  penalty: parches, no arreglos arquitecturales.
- **L-006** Memoria y emisión separables: recuperar bajo TF != actualizar
  estado con el token emitido.
- **L-007** Fugas de causalidad (norm/scan/chunk) inflan resultados; probe
  de sufijo + paridad obligatorios.
- **L-013** Free-decode OOD es techo UNIVERSAL a d=64; **diagnosticar el
  instrumento antes que la arquitectura**.
- **L-015** Baseline GPT-2 fiel (head SIN atar — tying degrada copy
  1.0→0.84); suma EM 1.534 @ d=64.
- **L-016** Fallo retrieval del transformer es régimen de datos, no bug:
  0.9986 EM en régimen paper.
- **L-017** Abortar pre-entrenamiento si `slope_act > 1.5` en Fase 0.
- **L-022/023** conv1d en readout sofoca la rama post-ANSWER; punto de
  injerto importa.
- **L-027** Loop Markoviano es atractor greedy, no divergencia: `argmax`
  colapsa a la modal; muestreo reduce.
- **L-028/029/030** signal_init falso positivo si grad 10x dispares;
  `beta_init=1.0` mata gradiente (usar 0.5); doble silu reduce señal.
- **L-031/032** El gate d=64 mide profundidad/capacidad, no tipo de
  atención; el transformer solo discrimina con copy_reverse (1.0).
- **L-035** loop_eval NO es depth generalization: degrada monótono; no usar
  loop_eval > loop_count como palanca.
- **L-046** fp16 en bloque > fp32 (regulariza; st 0.500 RÉCORD).
- **L-050** Feedback full-bandwidth CERRADO (replace colapsa 0.965→0.083);
  dual-bank + `h_fb = top UNFUSED` → neutral a d64.
- **L-052/053** VM latente CERRADA: la ejecución latente no emerge del
  gradiente end-to-end; weight-tying y VQ destruyen la compilación.
- **L-055** wave_iq CERRADO: SSM complejo 2 bandas (Kautz + PAC + I/Q +
  Householder) 0/5 gate; I/Q no da retrieval (mqar 0.008 < TF 0.058), st
  0.413 ~ banda lineal (loopy 0.500). G0/Fase 0 PASS (slope −1.915 = eje
  degenerado; real +0.031). Bankable: DualMuon, G0 multi-paso.
- **L-057/058/059** Loopy d384: attention-only NO escala (+3.60 ppl);
  ternario +1.46 ppl a escala; GDN-2 domina TinyStories a escala.
- **L-061** Curva de datos loopy d384 (5M→100M): 12.72→6.25 ppl sin meseta
  dura; beneficio marginal se comprime ~2x por duplicación.
- **L-062** Regla delta + erase nativo REFUTADA a d=128: mqar 0.025 / forget
  0.057 < anclas d=64 Y < transformer d=64; divergencia train→0/valid > ln(v).
- **L-063** loopy_osc CERRADO: coRNN 2º orden (γ=ε=1, dt=0.5) = estabilidad
  de EVALUACIÓN, no capacidad — st free C=8 no decae con K (0.42→0.55) vs
  fast (0.45→0.28); margen máx +0.22 single-seed (ruido ±0.10); st a techo
  recurrente 0.55 (gate 0.70); acumulador ciego al orden (copy 0.0 a TODA
  profundidad; st TF plana 0.39-0.42); control cuantizado C=8 iguala
  estabilidad en copy. Kaggle v2: imports `tests.*` fallan (shadowing) →
  flat imports + exec en-kernel.

## Piezas de código bankables (las que se rescatan)

- `prototypes/transformer/model.py` — baseline del smoke gate (suma EM 1.534
  @ d=64), congelado por hash.
- `prototypes/transformer_lit/` — port fiel Zoology, 0.9986 EM en régimen
  paper. Control "¿pruebas demasiado exigentes?".
- `prototypes/kimi_kda/` — porte fiel KDA. Control "arq publicada vs
  pruebas": 1/5 como transformer; st discrimina (0.70 vs 0.44).
- `prototypes/causal_state_lm/` — GRU minimal con contrato `prefill`/
  `decode_step`. Tracer bullet G0-G3.
- `prototypes/_pieces/` — piezas inmutables (rmsnorm, rope, swiglu,
  selective_s4d, entmax, gated_delta_rule, ...), congeladas por hash.
- `archive/wave_iq/` y `archive/loopy_osc/` — las dos familias ondulatorias
  caídas (model.py + logs S104/S109): punto de partida documentado de la
  línea que ONDA persigue.
- Herramientas del lab viejo NO clonadas (se portan cuando hagan falta):
  probes de diagnóstico (signal_init, early_slope, ood_transfer, emit_traj,
  lookahead_cut, ood_free_decode_v2), runners Kaggle TinyStories, batería VM.

## Baselines / Protocolo / Config (referencia, NO duplicados)

- Roles disjuntos: `transformer/` = gate smoke (1.534 @ d=64), `transformer_lit/`
  = régimen paper, `kimi_kda/` = calibración, `causal_state_lm/` = tracer
  bullet; nunca mezclar, nunca un solo baseline, nunca mutar;
  `check_frozen --strict`.
- G0-G3 (paridad `full==rollout`, overfit 32 seqs, delays 2/4/8/16), N0-N4,
  gasto 1→3→5-6 (single-seed ±0.10 = ruido), funnel 3 fases, gate 5-task
  (mqar_1hop 0.85 / mqar_2hop 0.80 / copy_reverse 0.90 / state_tracking 0.70
  / forget_retrieval 0.70): ver `PROTOCOL.md`/`GATES.md`.
- `d_model=64` (cpu_quick) / `128` (smoke/full), `n_layers=2`, `n_heads=4`,
  `ff_mult=2`, `forget_factor=0.98`; AdamW `lr=1e-3` `wd=0.01` `clip=1.0`;
  `ANSWER_ID=8`; `--emb_bits` {3=ternario, 4=e2m1, 8=e4m3, 16=e5m10}.

## Origen / Preguntas abiertas / Próximo paso

- Clonado de `../modelo-2` (S1-S110, cerrado). Iteraciones nuevas: O01+.
- **Claim vigente (O01 + O01b + O01c + O01d, `paper/CLAIM.md` v4)**: P1
  reformulado (O01d) — el borrado por CONSTRUCCIÓN (proyección sobre el
  subespacio de la clave) es exactamente robusto a drift de fase (reloj
  imperfecto); el borrado por RE-PRESENTACIÓN del valor no (deja
  2−2cosδ); la superioridad solo puede vivir en canal de clave +
  proyección de lectura: Clase B′ (drift δ sobre modelos entrenados, O04)
  o Clase C (claves aprendidas v2). Valor v REAL (canal imaginario nulo);
  wave_complex vs delta_forget = control con expectativa de identidad.
  Ley del olvido C1: fuga(orig_v) ≈ (n−1)/D, función de n/D no de D
  (verificada; fit con intercepto, excluyendo n/D ≥ 0.4, R²). Killer:
  selectividad_wave <= selectividad_delta @ d=128 en Clase B′ o C (ICs
  solapados) → Plan B (equivalencia + ley + residual, ya suficiente) / C
  (resultado negativo).
- Preguntas abiertas: ¿el delta entrenado desarrolla claves con estructura
  que la rotación/drift explota (Clase C)? ¿wave_mem aprende el patch de
  Re(·) en d=128 como sugiere el EM 0.91-0.97 del residual sin²δ·v? ¿la
  selectividad en Clase C es > delta? ¿la re-presentación oracle empeora la
  fuga en modelos entrenados como predice el operador?
- **Próximo paso (O05 CERRADO 2026-08-19)**: gate d=128 confirmado —
  el lab cierra su línea de investigación con el claim verificado en dos
  escalas (detalle: logs/O05_test.md, memory O-008). Siguiente: redactar
  el paper con la entrega al auditor (4 leyes, C1, TOST casi-equivalente,
  teorema de garantía condicional), tag de cierre `lab-radioactivo-v1` en
  el commit final, y O07 (drift δ(t)=ω·lag) queda como backlog para un
  futuro lab, no bloqueante.