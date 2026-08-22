# CLAIM.md — El claim del paper (O01, revisado O01b, O01c y O01d por auditorías)

> VIGENTE desde O01; **revisado O01b** (auditoría externa verificada,
> `outputs/audit/audit_o01b_verification.json`), **O01c** (segunda auditoría
> "killer test vacío", verificada por script propio,
> `outputs/audit/audit_o02_verification.json`), **O01d** (auditoría de la
> especificación de O03, verificada por script propio,
> `outputs/audit/audit_o03_verification.json`) y **O04b** (auditoría del 2×2
> canal×drift, `outputs/audit/audit_o04b_verification.json`; el auditor
> verificó las 4 leyes, retiró su B-Re 0.0870 y corrigió el framing: teorema
> de garantía condicional, §0c). Cambia SOLO al cerrar una iteración que lo
> refuta (PROTOCOL.md "Regla del claim").

## 0c. Lo que la auditoría O04b cambió (framing del 2×2, decidido 2026-08-19)

El 2×2 (canal {complex,re} × drift {clave,estado}) fue verificado por el
auditor (400 episodios/celda/δ, D=64, n=20): leyes de fuga exactas (0.000 /
0.318 / 0.087 / 0.051) + EM-decode (1.000 / 0.998 / 0.748). Corrección de
framing — **NO es "la cobertura exige ambos canales"; es que solo existe UNA
celda con borrado verdadero**:

| celda | fuga @ δ=0.5 | EM-decode @ δ=0.5 | EM @ δ=0.2 |
|---|---|---|---|
| clave/complex | 0.000 | n/a — residual EXACTAMENTE cero | n/a |
| estado/complex | 0.320 | 1.000 | 1.000 |
| clave/re | 0.088 | 0.998 | 0.432 |
| estado/re | 0.052 | 0.748 | 0.043 |

- La fuga en potencia SUBESTIMA la fuga funcional en >1 orden de magnitud
  (a δ=0.2 clave_re decodifica al 43% con fuga 0.018): el residual no es
  ruido informe, es una COPIA ESTRUCTURADA del valor borrado.
- **Teorema de garantía condicional (nuevo P1-bis, verificado)**: el borrado
  exacto de wave_mem es propiedad del PROYECTOR con clave fresca; la
  garantía es ABSOLUTA bajo drift de reloj (clave: residual exactamente
  cero, nada que decodificar) y se degrada a copia-decodificable bajo drift
  de estado. El head entrenado decodifica el residual en el canal Re
  (0.91-0.97 @ δ=0.5 en nuestros ckpts).
- Lectura arquitectónica honesta: en wave_mem v1 con codebook fijo NINGUNO
  de los dos drifts ocurre naturalmente (claves = lookup exacto, estado no
  oscila): el 2×2 es un STRESS TEST que delimita dónde vive la garantía, no
  un modo de fallo operativo. El drift de estado realista acumula con el
  lag escritura→olvido, δ(t) = ω·lag (O06, backlog).
- La reconciliación de números queda: A-Re = sin²δ + c/2 bajo la convención
  del auditor reproducida exacta (0.2868 vs 0.2884); B-estado-Re 0.0870 era
  un híbrido sin celda física limpia — RETIRADO por el auditor. Taxonomía
  2×2 = correcta; leyes de represent intactas (agnósticas a variante).

## 0d. Validación a d=128 en modelos entrenados (O05, cerrado 2026-08-19)

Gate O05 CONFIRMADO en Kaggle 2xT4 (kernel javierezcobar/onda-o05 v5,
5 seeds, 3 brazos wave_complex/wave_re/delta_nlms, FR d=128 n_pairs
[32,48] → n/D ∈ [0.25, 0.38], c = 0.3047; evidencia:
`outputs/onda_o05_kaggle/onda/outputs/wave_mem/o05.json`):

- **Las 4 leyes de la §0c se sostienen a d=128** (modelos entrenados a EM
  1.000): max|med−pred| = 0.0049 < 0.02 en 5 seeds × 4 celdas × 6 δ.
- **Borrado exacto**: clave/complex = "residual nulo" (‖r‖ ≤ ε‖v‖) en las
  6 δ × 5 seeds — la garantía condicional es ABSOLUTA bajo drift de reloj;
  clave/re decodifica creciente con δ (EM_erased 0.06 → 1.00) = copia
  decodificable; estado ~0 (los heads no decodifican estado, consistente
  con §0c).
- **Dos escalas, misma ley**: C1 a c emparejada (d=128 c=0.3047 vs d=64
  re-probe n_pairs (17,24) c=0.3047): |diff| ≤ 0.0064 < 0.02 en 4/4 celdas
  → las leyes son función de c = (n−1)/D, no de D (ley C1 del olvido
  extendida al régimen d=128).
- **Control delta**: delta_nlms entrena FR a d=128 (EM_alive 1.000 en 5/5
  con 150ep × 1200) y su selectividad es INDISTINGUIBLE de wave_complex
  (TOST: mean_delta −0.0031, IC90 [−0.0206, 0.0144] = casi-equivalente,
  cruza ε=0.02 por 0.0006, reportado sin redondear): la selectividad de
  borrado NO discrimina operadores a esta escala — el claim vive en la
  forma de la fuga (leyes), no en el EM de borrado.
- Presupuesto calibrado por familia (lección técnica): wave satura a
  60ep × 600 (~12 min/brazo), delta_nlms necesita 150ep × 1200 (~85
  min/brazo).
- **Estado del claim tras O05**: P1 (borrado por construcción exactamente
  robusto a drift de fase) verificado en dos escalas; 4 leyes cerradas;
  condición de estabilidad LMS (‖S‖ acotada, O04) vigente; teorema de
  garantía condicional validado con la guarda de residual nulo. El lab
  cierra con el paper: olvido operador-específico en el canal de clave,
  no en el operador de borrado (ver §1). O07 (drift δ(t) = ω·lag) queda
  como backlog de un lab futuro.

## 0b. Lo que la auditoría O03 cambió (O01d, decidido y verificado)

| Decisión | Antes (O01c) | Después (O01d) | Razón (verificada) |
|---|---|---|---|
| Tipo de valor v | complejo (emb 2D→complejo) | **v REAL: `v_c = complex(v_proj(x), 0)`** | Ambigüedad de tipo detectada: con v complejo el canal Re deja fuga 0.56 (retiene la mitad imaginaria: ½ + (n−1)/(2D)); con v real la predicción es limpia: complejo 0.000, Re (n−1)/(2D) = 0.059, orig_v (n−1)/D = 0.117 (medido 0.000/0.068/0.128 @ D=128, n=16). El claim fija el tipo de valor desde el día 1. |
| 4º brazo | no existía | **wave_represent = intervención ORACLE en eval** (erase con v_i VERDADERO inyectado por el evaluador, una línea en el probe, sobre modelos entrenados en el formato oficial) | El comparador que discrimina la hipótesis es borrado por CONSTRUCCIÓN (proyección) vs por RE-PRESENTACIÓN del valor. Se rechaza el brazo entrenado con formato modificado `FORGET k v`: rompe "misma interfaz" y abre atajo de diccionario de valores. |
| Clase B | ρ (correlación espacial del ruido) | **Clase B′ = drift de fase común δ** (reloj imperfecto: la clave de borrado llega con fase δ) | La maquinaria ρ fue NULL (agnóstica con claves i.i.d., O02). El drift de fase δ es el ruido físicamente motivado del canal complejo y discrimina EXACTO: re-lectura = fuga 0.0000 a todo δ (e^{iδ}w preserva el rango 1-D: proyector idéntico); re-presentación = (n−1)/D + (2−2cosδ) (medido: incrementos δ² exactos). Sube a O04 como benchmark con drift temporal. |
| wave_complex vs delta_forget | dos brazos comparativos | **CONDICIÓN DE CONTROL**: expectativa de IDENTIDAD pre-registrada (mismo operador, parametrización distinta; si difieren = diferencias de optimización/inductive bias, no de operador) | Verificado: erase idéntico (Clase A diff 0.0). No son dos hipótesis competidoras. |
| P1 (redacción) | "selectividad donde el delta no alcanza" | **"el borrado por construcción (proyección sobre el subespacio de la clave) es exactamente robusto a drift de fase; el borrado por re-presentación del valor no"** | Propiedad del operador, falsable, no formalizada en VSA/HRR: la novedad formalizable del paper. |
| C1 (fit) | slope log-log 1.125, sin excluir | **fit con intercepto, EXCLUIR n/D ≥ 0.4 (techo de saturación), reportar R²** (medido: slope 1.054, R² = 0.959, 9 puntos) | El techo n/D=0.5 distorsiona el fit. Regla permanente: ningún fit de ley sin excluir su techo y reportar R². |
| i.i.d. en la clave | asumido robusto | **formas cerradas `fuga_reread(σ)` para phase y additive; cruce analítico reconciliado** | Ruido de fase (ε_d i.i.d. N(0,σ) rad, reloj imperfecto): `reread = (1−e^{−σ²})² + (n−1)/D·(1−e^{−σ²})`, `represent = (n−1)/D + (1−e^{−σ²/2})²`. Aditivo (E\|ε_d\|² = σ²): `reread = (σ²/(1+σ²))² + (n−1)/D·σ²/(1+σ²)`. Cruce analítico phase σ* ≈ 0.64 rad y additive σ* ≈ 0.64 @ D=128 n=16 (fuga_reread 0.15/0.12, leak de amplitud ~35-39%) — FUERA de todo régimen operativo (reloj real σ ≪ 0.1 rad): la ventaja de reread cubre el régimen completo; la inversión es límite asintótico. La discrepancia previa (cruce en σ∈(0.3,0.5)) era convención de σ: por componente (Re e Im N(0,σ), potencia 2σ²) corre el cruce a ~0.51; por elemento (potencia σ²) a ~0.64-0.72. El probe matchea la FÓRMULA en cada σ (medido vs pred < 0.01 @ σ≤0.6), no un cruce empírico. |
| Drift del delta real | sin definir | **análogo = rotación 2-D de la clave real: fuga ≈ (2/D)sin²δ (protección dimensional, O(1/D))** | Verificado: 5e-5/2.3e-4/5.3e-4 para δ=0.2/0.4/0.6 @ D=128. El drift de fase complejo no existe en R; la comparación del delta se hace por corrupción de entrada (O04). |
| Normalización del erase aditivo | no declarada (fork: `/‖w′‖²` vs `/D`) | **erase = proyección idempotente: `(I − ww^H/‖w‖²)` (la variante sin normalizar `/D` NO es proyección; diff 2× en fuga_reread, verificada por el auditor)** | Fork cerrado por auditoría O03: la regla delta β=1 canónica normaliza por `‖w′‖²`. Declarado aquí para que el aditivo sea robustez secundaria y no una segunda teoría. |
| Modelo de ruido canónico O04 | — | **ruido de FASE (ε_d i.i.d. N(0,σ) rad en la clave de borrado) es el modelo canónico del canal de borrado; aditivo = robustez secundaria con la normalización fija** | Fase: físicamente motivado (reloj imperfecto), preserva ‖w‖=1, forma cerrada más limpia. Aditivo: `σ²/(1+σ²)` (no es proyección exacta, solo cercana) — se reporta como check de robustez, no como benchmark de O04. |

## 0. Lo que O01c cambió (decidido y verificado)

| Decisión | Antes (O01b) | Después (O01c) | Razón (verificada) |
|---|---|---|---|
| Canal de borrado v1 | re-lectura compleja | **complejo para Clase A, variante `read_proj='re'` para Clase B** | Con `‖w‖²=D`, `M − w⊗(w^H M)/D ≡ (I − uu^H)M ≡` proyección de la regla delta β=1: el erase por re-lectura compleja ES el operador delta (diff 0.0). La re-lectura compleja no puede ganar al delta por construcción → no es donde vive P1. La re-lectura `Re(·)` (onda "real": solo la parte real es física) SÍ difiere del canal complejo (fuga 0.558 vs 0.000 @ D=128, n=16: deja la mitad imaginaria del ítem). |
| Clases de experimento | una sola comparación | **Clase A (equivalencia, al paper como linaje) · Clase B (corrupción del canal de clave en el token de borrado: wave inmune por construcción — codebook exacto — vs delta con canal aprendido sensible; síntesis O02 = NULL test, probe real = O04 sobre modelos entrenados) · Clase C (claves aprendidas v2: portador primario de P1)** | La equivalencia A es un TEOREMA de linaje (wave_mem ≡ delta β=1): publicable como resultado de teoría, no como superioridad de mecanismo. Verificado en síntesis: con claves i.i.d. el operador es AGNÓSTICO a la forma del ruido (spread ρ < 0.03 en ambos brazos) → la sensibilidad a correlación espacial exige estructura aprendida en las claves (Clase C) o en el canal de entrada entrenado (Clase B en modelos entrenados, O04). |
| Métrica de fuga | sin definir formalmente | **`fuga = E\|r_j\|²/E\|v_j\|²`** con `r_j` = readout del ítem borrado en el canal de lectura del modelo | Pre-registrada; valores: complejo 0.0 (exacto), Re 0.5 + (n−1)/(2D) (retiene la mitad imaginaria), orig_v (n−1)/D (piso de crosstalk). El 0.062 del auditor para Re es la variante imag-crosstalk; se reporta como secundario. |
| Ley de carga (C1) | residual (n−1)/D en función de D | **fuga(orig_v) ≈ (n−1)/D: función de n/D, NO de D** (verificado: 0.099/0.103/0.135 para (64,8)/(128,16)/(256,32); n/D=0.5 → 0.51–0.66, techo para todos) | El probe C1 barre n/D, no D. Régimen operativo del harness: n/D ≤ 0.14 (n≤9, D=64) → techo de EM de TODO operador ≈ 1 − n/D; el claim es ORDENAMIENTO sobre el piso, no EM absoluto. |
| Killer test | selectividad_wave <= delta @ d=128 → Plan B | **P1 sobrevive SOLO si wave_mem (Re(·)) gana selectividad en Clase B o Clase C; la equivalencia A no puede sustentarlo** | En régimen ideal (Clase A) es imposible por construcción (diff 0.0). El resultado positivo posible está acotado a las variantes donde los operadores difieren. |
| Interfaz | FORGET_ID + β=1 | **escritura estructural (t≥9 y t−1≥9) en AMBOS brazos; erase estructural (t−1==FORGET_ID, t≥9); sin gates aprendidas en v1** | Apples-to-apples: misma interfaz de markers; el operador es la única diferencia. La regla `recommend_kaggle` NO es la regla de decisión de P1 (ningún brazo tiene atención/RoPE → copy_reverse y mqar quedan bajos en ambos; pre-registrado: no es evidencia contra P1). |
| Tipo de escritura (documentado por la autopsia O03-N1, auditoría) | no declarado | **wave: superposición pura `M += w_pend ⊗ v`; delta: escritura CORRECTIVA `M += (v − k^T M) ⊗ k` (regla LMS/online-delta); el erase proyectivo es el ÚNICO operador equivalente (Clase A diff 0.0)** | La equivalencia verificada cubre el ERASE (misma proyección `(I − uu^H)`); el WRITE difiere entre brazos: en N1 wave y delta NO eran el mismo operador completo — compartían interfaz y erase, no escritura. Consecuencia para el claim: la comparación N1 delta-vs-wave mezcla write y erase; la equivalencia de linaje (Clase A) se reafirma SOLO sobre el erase. El claim P1 usa la lectura/erase (operador compartido); el write correctivo del delta es su receta de optimización, no su erase. |

## 1. La afirmación novedosa

**Claim principal (P1, vigente):**
> A presupuesto igualado (params y estado, geometría matricial D²), el
> borrado asociativo es **operador-específico**: el borrado por construcción
> (proyección `(I − uu^H)M` sobre el subespacio de la clave de borrado) es
> EXACTAMENTE robusto al drift de fase de la clave (reloj imperfecto),
> mientras que el borrado por re-presentación del valor no lo es (deja
> `2 − 2cosδ` del ítem); el canal de clave (fijo unitario vs aprendido) y la
> proyección de lectura determinan la selectividad donde la regla delta con
> la MISMA interfaz FORGET_ID (β=1) no alcanza el mismo nivel (Clase B′/C),
> y donde la atención (0.067 @ d=64) no borra sin interfaz explícita.

**Sub-claims:**

- **C1 (teoría)**: la ley del olvido del operador: `fuga(orig_v) ≈ (n−1)/D`
  — función de la carga n/D, no de la dimensión (verificado O01c). Con
  `v` real (O01d): borrado por re-lectura compleja → fuga 0.0 exacta
  (≡ delta β=1); lectura Re(·) → (n−1)/(2D) (sin término ½: ese venía de
  valor complejo). Extensión no-formalizada de VSA/HRR (linaje citado:
  Plate 1995, Kanerva, Frady): el residual post-erase con claves
  aprendidas y borrado por re-lectura.
- **C2 (mecanismo)**: el erase es SELECTIVO: borrar k_i no degrada la
  recuperación de k_j más allá del piso `(n−1)/D`.
- **C3 (contrato)**: `forward == prefill/decode_step` O(1) en secuencia
  (estado D², coste O(D²)/token — declarado en el presupuesto igualado),
  paridad G0 bit-exacta, sin degeneración en free decode (probe G3).

## 2. El experimento decisivo (killer test, endurecido O01d)

**Killer test**: selectividad en **Clase B′** (O04: drift de fase δ de la
clave de borrado sobre modelos ENTRENADOS, comparando los canales de
borrado: re-lectura vs re-presentación oracle) y **Clase C** (claves
aprendidas v2), `forget_retrieval` @ d=128 (Kaggle), brazos con interfaz
IDÉNTICA: wave_mem(Re), delta+FORGET_ID (β=1), transformer (puede aprender
masking por contexto — se documenta, no se castra). Métrica primaria:
selectividad (EM_vivo − fuga). La síntesis O03 (probe Clase B′) fija los
números del operador: re-lectura fuga 0.0000 EXACTA a todo δ; re-presentación
≈ (n−1)/D + (2−2cosδ); el discriminador entre hipótesis de borrado se mide
sobre modelos entrenados recién en O04.

Si `selectividad_wave <= selectividad_delta` (ICs de 3 seeds solapados) →
**P1 muere** → Plan B (teoría + ley del olvido, ya con contenido
suficiente) / C (resultado negativo).

**Filtro local barato (antes de Kaggle)**: selectividad @ d=64 cpu_quick,
3 seeds, Clase B′ (δ) y re-presentación oracle. Si
`selectividad_wave <= selectividad_delta` en 2+ seeds → descarte inmediato.
Barra interna de señal: `selectividad_wave >= 0.30`.

**Anti-falsos positivos (obligatorios)**:
- G0 estricto: invariante de sufijo bit-exacto + `full == rollout` +
  gradientes conectados (sin G0 no se discute).
- Apples-to-apples: mismos seeds, preset cpu_quick, épocas; snapshot
  congelado `outputs/transformer/benchmark.json` como referencia.
- Presupuesto igualado DECLARADO por brazo: params, estado (D² vs D²),
  FLOPs/token O(D²); contra atención se declara la asimetría de estado.
- Misma interfaz FORGET_ID + escritura estructural idéntica (pre-registrado
  en §8).
- N2 mínimo (3 seeds) para "supera"; paper: ≥5 seeds + IC (reportar además
  cosine similarity del readout; EM es métrica de umbral).
- Pre-registrado: `recommend_kaggle` NO es la regla de decisión de P1.
  Ambos brazos son operadores puros sin atención/RoPE: copy_reverse y mqar
  bajos en los DOS son esperados y no tocan el claim (se reportan igual).

## 3. Nivel de evidencia objetivo

- **P1**: N3 — 3+ seeds d=128 smoke (Kaggle), Clase B y C + ablaciones
  (claves fijas vs aprendidas · borrado Re(·) vs complejo). El orden de
  operadores en selectividad bajo ruido es el resultado publicable.
- **C1**: probe independiente (fuga medida vs predicha `(n−1)/D`, barrido
  n/D, fit de exponente con intercepto, excluyendo n/D ≥ 0.4, con R²) —
  figura 1 del paper.
- **C3**: N1 (G0 + G3).
- Planes: O02 prototipos + probes; O03 N1 local; O04 N2 + Clase B′; O05
  Kaggle d=128; O06 manuscrito v0.

## 4. Planes de respaldo (si P1 muere)

- **Plan B — paper de teoría**: la equivalencia wave_mem ≡ delta β=1 como
  TEOREMA de linaje + la ley del olvido `(n−1)/D` (n/D, no D) + el análisis
  de residual post-erase con claves aprendidas → "el eje del olvido también
  es operador-específico" (extensión de L-064). Ya tiene contenido
  suficiente: la novedad es la medición, no el ganador.
- **Plan C — resultado negativo**: la ley L-056/064 extendida al eje de
  borrado, N3 completo. Honesto y defendible.

## 5. Lo que el claim NO afirma (límites explícitos)

- NO que la interferencia supere a la atención en retrieval (mqar): la ley
  L-056/064 se sostiene; la memoria por interferencia pierde por crosstalk
  a n_pairs grandes (eso es C1/C2).
- NO que wave_mem supere al delta en el régimen ideal (Clase A): es
  IMPOSIBLE por construcción (mismo operador, diff 0.0) — la equivalencia
  es el linaje del paper, no un resultado de superioridad.
- NO escala TinyStories (fuera de presupuesto). Claim a d=64/d=128 en el
  harness del lab, comparación controlada contra baselines congelados.
- NO es un mecanismo nuevo tipo VSA: es la **instanciación** del operador
  de interferencia con interfaz de borrado nativa, y el paper lo dice
  explícitamente (ver §7).
- NO pretensión biológica/física: "onda" = fasores complejos en C^{D×D}.

## 6. Naming y arquitectura semilla (O02)

Prototipo: `prototypes/wave_mem/` — **memoria por interferencia, geometría
MATRICIAL fija**:

- Estado: `M ∈ C^{D×D}` (un acumulador por bloque). Coste por token
  O(D²), O(1) en secuencia.
- Claves: `w_k ∈ C^D`, v1 FIJAS (codebook de fasores unitarios aleatorios
  por token id, buffer congelado, sin gradiente; v2 aprendidas = Clase C).
- Valores: `v_c = complex(v_proj(x), 0)` — valor REAL con forma compleja
  (O01d: el tipo del valor es parte del claim; canal imaginario nulo).
- Escritura (estructural, sin gates aprendidas): máquina de estado de
  alternancia k,v accionada por markers (BOS/SEP/FORGET/QUERY ⇒ sigue un
  KEY; tras un KEY ⇒ sigue un VALUE): en el VALUE se escribe
  `M ← M + w_{key_pendiente} ⊗ v_c(emb_t)` con la clave pendiente del KEY
  anterior. IDÉNTICA en ambos brazos.
- Erase (token t−1 == FORGET_ID): `M ← (I − uu^H)M` con `u = w_t/√D`
  (≡ `M − w_t ⊗ (w_t^H M)/D`, α = 1 fijo). Canal complejo (Clase A) o
  `Re(·)` (Clase B) según `read_proj`. El evaluador puede además inyectar
  el borrado por RE-PRESENTACIÓN oracle (`M − w_t ⊗ v_true`, comparador de
  O04) sin tocar el modelo.
- Lectura (token t−1 == QUERY_ID): `r = (w_t^H M)/D` (o `Re(·)`) → head
  lineal SIN atar (L-015); `r` persiste hasta el ANSWER_ID.
- Contrato: `forward` == `prefill`/`decode_step` (mismo loop secuencial,
  paridad por construcción), estado `{M_b, r_b, prev_tok}`.

Brazo comparador obligatorio O02: `prototypes/delta_forget/` — regla delta
secuencial VANILLA (sin conv1d/RoPE/gates del GDN: comparador limpio del
operador) con la MISMA interfaz estructural: erase `(I − kk^T/‖k‖²)S` con
β=1 forzado en el token de borrado, lectura con `q` aprendida que persiste.

## 7. Por qué esto NO es S104 (ni VSA-plagio)

- **S104 (wave_iq)** intentó binding de contenido por demodulación I/Q
  (leer fase entre bandas) para RETRIEVAL; falló porque la demodulación no
  discrimina contenido a d=64 (consistente con L-056). wave_mem usa
  matched-filter para LEER y cancelación para BORRAR; P1 no reclama
  retrieval — la lectura no compite con atención (la ley se sostiene), el
  claim es el EJE DE BORRADO. S104 no es evidencia en contra de P1.
- **VSA/HRR**: superposición fasorial + unbinding por conjugado es el
  formalismo de Plate (1995) / Kanerva / Frady (capacidad). El paper cita
  el linaje y NO reclama el mecanismo como nuevo; reclama (a) la
  formalización del residual post-erase con claves aprendidas y borrado
  por re-lectura (no está en VSA clásico), (b) la equivalencia exacta con
  la regla delta β=1 (puente formal VSA→delta rules, no publicado), y (c)
  la medición del eje de borrado a presupuesto igualado. Incluir baseline
  HRR literal en related work; si no lo citamos, lo hará el reviewer 2.

## 8. Pre-registro O02 (decisiones bloqueadas ANTES de escribir código)

1. Geometría: matricial C^{D×D}. (Verificada: única donde C1 vive.)
2. Claves v1: fijas, aleatorias, unitarias (codebook buffer). v2
   aprendidas = Clase C, nunca mezclar.
3. Erase: α = 1 fijo; SIN gates aprendidas (nada de amplitud aprendida).
   Escritura (alternancia k,v por markers, clave pendiente) y erase
   estructurales: interfaz IDÉNTICA en ambos brazos.
4. Métrica primaria: `selectividad = EM_vivo − tasa_de_fuga`, con
   `fuga = E|r_j|²/E|v_j|²` (r_j = readout del ítem borrado en el canal de
   lectura del modelo). EM absoluto secundario.
5. Brazos con interfaz idéntica: wave_mem (read_proj complejo y 're'),
   delta_forget (β=1 en borrado), transformer (sin castrar: puede aprender
   masking). wave_complex vs delta_forget = CONDICIÓN DE CONTROL con
   expectativa de IDENTIDAD (O01d).
6. Umbrales: filtro d=64 → selectividad_wave >= 0.30 y > delta en 2+ de 3
   seeds (Clase B′); killer d=128 → selectividad_wave > delta (ICs no
   solapados), Clase B′ o C.
7. Presupuesto igualado declarado por brazo (params / estado / FLOPs);
   `recommend_kaggle` NO es regla de decisión de P1.
8. Seeds: 3 local (N1/N2), ≥5 + IC para el paper.
9. Reportar EM, selectividad y cosine del readout (EM es métrica de
   umbral).
10. Probe C1 barre n/D (no D): fuga(orig_v) vs (n−1)/D, fit de exponente
    con intercepto, excluyendo n/D ≥ 0.4 (techo), reportando R².
11. Tipo del valor: **v REAL** (`v_c = complex(v_proj(x), 0)`); predicciones
    de fuga pre-registradas: complejo 0.0, Re (n−1)/(2D), orig_v (n−1)/D
    (O01d).
12. Clase B′ sintética (O03): drift de fase δ ∈ {0, 0.2, 0.4, 0.6}:
    re-lectura 0.0000 exacta (ruido que preserva el rango 1-D de la clave);
    re-presentación (n−1)/D + (2−2cosδ); delta real bajo rotación 2-D:
    (2/D)sin²δ (protegido dimensionalmente — robustez del control, no
    resultado). Ruido i.i.d.: formas cerradas (ver §0b): phase
    `(1−e^{−σ²})² + (n−1)/D(1−e^{−σ²})`, additive `(σ²/(1+σ²))² +
    (n−1)/D·σ²/(1+σ²)`; cruce analítico σ* ≈ 0.64 (leak ~35-39% de
    amplitud, fuera de régimen operativo); el probe reporta medido vs
    fórmula, no un cruce empírico.
13. Re-presentación = intervención oracle en eval (nunca brazo entrenado
    con formato modificado).
14. Potencia estadística de N1 (O03, auditoría): el contraste es
    wave_re vs {wave_complex, delta_forget} con déficit Δ = (n−1)/(2D) @
    d=64. Spec: `forget_retrieval` con n_pairs ∈ [16, 24] (Δ = 0.117-0.180;
    n_pairs 8 → 0.055, ~830 eventos necesarios: insuficiente; n_pairs 32 →
    régimen saturado) y **≥300 eventos FORGET evaluados por brazo por
    seed** (potencia 90%: 160-320 eventos según la tabla del auditor).
    Si la task da menos eventos, N1 corre pero no puede ver su efecto
    principal: null por diseño, no por evidencia.
15. Control de identidad (wave_complex ≡ delta_forget) en modelos
    ENTRENADOS = test de equivalencia TOST: `|Δselectividad| < ε = 0.02`
    con IC 90% → control confirmado. Si falla: la diferencia se atribuye a
    parametrización/optimización (heads destadas) y se reporta como tal.
16. Probe de decodificabilidad de fuga (figura del paper): ¿puede un
    decodificador lineal entrenado sobre M post-erase predecir el v_i
    borrado desde el residual? Contrasta fuga-OPERADOR (fórmulas §0b) vs
    fuga-SISTEMA (EM del modelo entrenado): el head destado puede haber
    aprendido a decodificar crosstalk ("trampa legítima del head").

## 9. Presupuesto estimado

- O02: wave_mem + delta_forget + G0 + Fase 0-lite + probes (Clase A,
  n/D, canales, Clase B′ sintética, plumbing de selectividad). CPU, <2h.
- O03: N1 d=64 cpu_quick 5-task AMBOS brazos en la misma grilla (~30 min
  CPU) + probe de selectividad sobre modelos entrenados (con la
  intervención oracle de re-presentación).
- O04: N2 (3 seeds) + Clase B′ (drift de fase δ) sobre modelos entrenados.
- O05: si señal: mini-smoke d=128 Kaggle (forget_retrieval +
  state_tracking + copy_reverse, ~30-45 min).
- O06: manuscrito v0 con C1 + P1 (o Plan B/C).
- O07 (backlog, anotado por el auditor en O04b): drift de estado realista
  δ(t) = ω·lag (lag escritura→olvido, δ creciente por evento) — la versión
  físicamente fiel; conecta con si el entrenamiento aprende a compensar
  drift lento.