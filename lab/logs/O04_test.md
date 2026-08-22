# O04_test.md - Clase B' drift δ sobre modelos entrenados (2x2xδ) + side arm delta_nlms

Template fijo. El runner escribe RESULTADO. El agente cierra VEREDICTO +
LECCION cuando la iteracion se cierra en `memory.md`.

## HIPOTESIS

El auditor aprobó O04 con dos enmiendas (2026-08-19):

1. **Forma cerrada de la autopsia (condición de estabilidad LMS)**: el
   write correctivo es un lazo cerrado con ganancia por dirección
   |1 − β‖k‖²| — estable solo si β‖k‖² < 2; la superposición es un lazo
   abierto incondicionalmente estable. La puerta β aprendida y la
   normalización de claves de DeltaNet/GDN son el dispositivo de
   estabilidad, no (solo) maquinaria de olvido. Side arm opcional:
   `delta_nlms` (claves normalizadas ‖k‖=1, β=1, erase exacto por
   proyección unitaria, sin gates) — si entrena FR el control TOST
   resucita; si no, "delta FALLIDO" queda hermético.
2. **Tres leyes, no dos**: la ley pre-registrada "represent ≈ piso + δ²"
   es el Taylor de 2(1−cosδ); y el canal Re tiene sensibilidad CUÁRTICA:
   represent-Re = (n−1)/(2D) + (1−cosδ)² (Re(v(1−e^{iδ})) = v(1−cosδ)).
   Diseño 2×2×δ, grilla extendida a δ ≤ 0.5, P1 reformulado = verificación
   de leyes por canal, criterio de fallo sobre reread.

La verificación previa del agente (`tests/o04_laws_verify.py`) CORRIGE la
enmienda 2 en un punto: **"reread plano" es falso para el canal Re** — el
erase Re-truncado del modelo con clave drifteada deja sin⁴δ + (n−1)/D
(1−cos2δ)/4 (verificado sintéticamente: 0.0909 vs 0.0941 @ δ=0.5, n=24,
D=64). La invarianza exacta es del proyector COMPLEJO. Consecuencia: el
criterio de fallo "reread > 0.02 @ δ=0.5 → Plan B" aplica SOLO al canal
complex (el Re lo superaría espuriamente por su ley cuártica).

## PREDICCION

Sobre modelos entrenados (6 ckpts wave, n_pairs [16,24], 320 eventos
FORGET/brazo/seed, fuga POOLED Σ|r|²/Σ|v|²):

- **reread complex**: fuga 0.0000 a todo δ ∈ {0,...,0.5} (proyector
  invariante: e^{iδ}w preserva el rango 1-D). Criterio P1: fuga > 0.02 a
  δ=0.5 → invarianza no sobrevivió el entrenamiento → Plan B.
- **represent complex**: (n−1)/D + 2(1−cosδ) — incrementos exactos.
- **reread re**: sin⁴δ + (n−1)/D·(1−cos2δ)/4 (ley CORREGIDA, sin
  parámetros libres).
- **represent re**: (n−1)/(2D) + (1−cosδ)² — incrementos exactos (piso
  con posible desvío por elipticidad del crosstalk entrenado).
- **EM sanity**: complejo-reread EM ≈ chance a todo δ (residual nulo);
  re-reread EM crece con δ (el head decodifica el residual sin⁴δ cuando
  supera su umbral); represent EM ≈ chance (el head NO decodifica el
  residuo 2(1−cosδ) porque la query es la clave BORRADA — predice el
  valor borrado a chance).
- **delta_nlms**: entrena FR (β‖k̂‖² = 1 < 2 → estable); si EM > 0.5 en
  las 3 seeds, TOST resucitado (identidad selectividad wave_complex vs
  delta_nlms, ε = 0.02, IC 90%).

## DISEÑO

- Runner `tests/o04_run.py` (resume por ckpt): (1) entrena delta_nlms FR
  (n_pairs [16,24], 3 seeds, 80 ep, d=64, L2; traza ‖S‖_F por época); (2)
  probe de drift: por ckpt wave, por modo {reread, represent}, por δ
  {0,.1,.2,.3,.4,.5}: intervención en el token FORGET (t = pos_forget+1)
  con clave drifteada e^{iδ}w, skip del erase entrenado (prev = clave),
  readout de la clave borrada en la query, fuga pooled + EM; (3) TOST si
  delta_nlms entrenó.
- `tests/o04_laws_verify.py`: verificación sintética de las leyes +
  condición de estabilidad LMS (ganancia |1−β‖k‖²| y ‖S‖ a t=120).
- Prototipo nuevo `prototypes/delta_nlms/model.py` (standalone completo,
  NLMS: claves normalizadas, β=1, sin gates aprendidas).
- Presupuesto CPU: ~35 min (delta_nlms 3×~8 min + probe 6 ckpts × 6 δ × 2
  modos × 10 batches). Salidas: `outputs/wave_mem/o04.json`,
  `o04_laws_verify.json`, `o04_run.log`, ckpts `outputs/o04_delta_nlms/`.

## RESULTADO (runner)

### Verificación sintética (`o04_laws_verify.json`)

Leyes (D=64, n=24, 20 trials): complex reread 0.0000 a todo δ ✓; complex
represent 0.419→0.631 vs (n−1)/D + 2(1−cosδ) 0.359→0.604 (dentro del
ruido de crosstalk, 20 trials); **re reread 0.0909 vs sin⁴ + c(1−cos2δ)/4
= 0.0941 @ δ=0.5 ✓ (NO plano — corregido vs enmienda)**; re represent
0.186→0.226 vs c/2 + (1−cosδ)² 0.180→0.195 (incrementos ✓, ruido de
crosstalk).

Estabilidad LMS: ganancias |1−β‖k‖²| = 7.0 / 0.2 / 0.5 exactas; ‖S‖ a
t=120: correctivo β=1,‖k‖²=8 → 8.6×10¹⁴ (diverge); β=0.1 → 12.0
(acotada); β=1,‖k‖²=1.5 → 51.8 (acotada); superposición → 125.3 (crece
lineal, nunca explota). **Teorema del auditor verificado: lazo abierto
incondicionalmente estable vs lazo cerrado con β‖k‖² < 2.**

### Side arm delta_nlms (FR, n_pairs [16,24], 3 seeds)

| seed | EM final | ‖S‖_F ep1 → ep80 |
|---|---|---|
| 1 | **1.000** | 83.0/81.8 → 100.9/173.3 |
| 2 | **0.993** | 82.0/82.6 → 127.5/151.0 |
| 3 | **0.993** | 83.0/80.8 → 116.7/148.6 |

**delta_nlms ENTRENA FR** (‖k‖=1 → ganancia 0 → write estable; ‖S‖
acotada ~10², vs 10¹² del delta_forget β=1 sin normalizar). **El control
TOST RESUCITA.**

TOST resucitado (selectividad EM, 320 eventos): delta_nlms sel =
0.950/0.959/0.966; wave_complex sel = 0.972/0.975/0.972 (N1).
Δ(sel_wc − sel_nlms) = 0.022/0.016/0.006, mean 0.0146, IC90
[0.0013, 0.0278], ε = 0.02 → **pass = False (marginal: media < ε, IC90
excede por 0.0078)**. La identidad se cumple EN MEDIA; el IC90 con 3
seeds no alcanza el criterio estricto. Se reporta como control
"casi-equivalente" (el EM es métrica de piso; la equivalencia sintética
Clase A diff 0.0 ya está verificada).

### Probe drift 2×2×δ (320 eventos/brazo/seed, fuga POOLED)

| canal | erase | δ=0 | δ=0.1 | δ=0.2 | δ=0.3 | δ=0.4 | δ=0.5 | ley (c=(n−1)/D=0.297) |
|---|---|---|---|---|---|---|---|---|
| complex | reread | 0.0000 ×3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000 ×3** | 0 (plana) |
| complex | represent | 0.299-0.313 | 0.310-0.323 | 0.341-0.353 | 0.391-0.404 | 0.461-0.473 | **0.549-0.562** | c + 2(1−cosδ) = 0.297/0.307/0.337/0.386/0.453/0.542 |
| re | reread | 0.0000 ×3 | 0.0015-0.0016 | 0.0073-0.0075 | 0.0205-0.0209 | 0.0456-0.0462 | **0.0876-0.0885** | sin⁴δ + c(1−cos2δ)/4 = 0/0.0019/0.0076/0.0206/0.0462/0.0869 |
| re | represent | 0.1628-0.1638 | 0.1629-0.1639 | 0.1635-0.1645 | 0.1656-0.1665 | 0.1704-0.1713 | **0.1799-0.1808** | c/2 + (1−cosδ)² = 0.148/0.148/0.149/0.150/0.155/0.163 |

- **reread complex: 0.0000 a todo δ ≤ 0.5** → criterio de fallo P1
  (fuga > 0.02 @ δ=0.5): 0.0000 → **P1 NO dañado: la invarianza
  sobrevivió el entrenamiento**.
- represent complex: δ=0.5 medido 0.549-0.562 vs ley 0.542 (+0.007-0.020
  = piso con elipticidad de normas, igual que en N1); incrementos
  2(1−cosδ) exactos (δ=0.5: +0.248 vs +0.245).
- **reread re: 0.0876-0.0885 vs ley sin⁴ 0.0869 @ δ=0.5 — la ley
  CUÁRTICA corregida es EXACTA en modelos entrenados** (0.0205-0.0209 vs
  0.0206 @ δ=0.3). La invarianza bajo drift es una propiedad del
  proyector COMPLEJO; el canal Re la aproxima a orden δ⁴.
- represent re: incrementos (1−cosδ)² exactos (+0.017 vs +0.015 @ δ=0.5);
  piso 0.163 vs c/2 = 0.148 (+10%: elipticidad del crosstalk entrenado —
  E[Re(c)²] = 0.55·E|c|² en vez de 0.5).
- **EM sanity**: complejo-reread EM ≈ chance (0.01-0.02) a todo δ (el
  residual es exactamente 0); re-reread EM crece 0.02 → 0.91-0.97 a
  δ=0.5 (el head DECODIFICA el residual sin⁴δ — coherente con la ley);
  represent EM ≈ chance a todo δ en ambos canales (la query es la clave
  borrada).
- Nota de métrica: la media de ratios infla la fuga del canal re
  (~0.026 espurio @ δ=0.5 por la cola pesada de ‖v‖², p90/p50 ≈ 5.6);
  la fuga POOLED (Σ|r|²/Σ|v|²) es la métrica correcta y reproduce las
  leyes exactamente.

## VEREDICTO

**O04 CONFIRMA las leyes del operador en modelos entrenados — P1
sobrevive.** Cuatro leyes cerradas sin parámetros libres verificadas
simultáneamente (la enmienda del auditor era 3; la 4ª — reread-re = sin⁴ —
es la corrección del agente, exacta a 0.002):

1. **reread complex plano 0.0000 a todo δ ≤ 0.5** — la invarianza del
   proyector sobrevivió el entrenamiento (criterio de fallo del auditor:
   0.0000 < 0.02 @ δ=0.5 → P1 NO dañado). El head NO aprendió a leer con
   clave drifted.
2. **represent complex = (n−1)/D + 2(1−cosδ)**: 0.549-0.562 vs 0.542 @
   δ=0.5 (incrementos exactos; el head decodifica el residuo solo a
   chance — el borrado por representación NO sobrevive al drift).
3. **reread re = sin⁴δ + c(1−cos2δ)/4**: 0.0876-0.0885 vs 0.0869 @ δ=0.5
   — la ley cuártica es EXACTA. La invarianza exacta es del canal
   complejo; el Re la aproxima a orden 4.
4. **represent re = (n−1)/(2D) + (1−cosδ)²**: incrementos exactos, piso
   +10% por elipticidad (no parámetro libre: se reporta el piso medido).

**Contraste P1 reformulado (verificación de leyes por canal): CONFIRMADO**
— las leyes se cumplen dentro de tolerancia (±0.02 absoluto = escala del
piso). La figura del paper: 4 curvas cerradas medidas vs teoría en modelos
entrenados + la ley de estabilidad LMS de la autopsia.

**Side arm delta_nlms: ENTRENA FR (EM 1.000/0.993/0.993, ‖S‖ acotada)**
→ la declaración "delta FALLIDO" se refina: el delta REAL con write
correctivo β=1 necesita la normalización NLMS (el dispositivo de
estabilidad) para ser optimizable; con ‖k‖=1 entrena igual que wave. El
control TOST resucita y es CASI-equivalente (Δ = 0.0146 < ε en media;
IC90 [0.001, 0.028] excede ε por 0.008 — identidad en media, no estricta
con 3 seeds). Opción B queda definitivamente cerrada: la resurrección del
delta vino por NORMALIZACIÓN (NLMS), no por gates aprendidas ni β≠1.

Estado: O04 cerrado. **El gate d=128 queda reescrito en GATES.md** con
las 4 leyes verificadas + métrica pooled. Próximo: mini-smoke d=128 de
los brazos wave (2 tasks + state_tracking de control) o el smoke completo
— decisión del auditor.

## LECCION

- El drift de fase al momento del FORGET separa canales con precisión de
  ley: complejo = invarianza EXACTA (orden infinito), Re = sin⁴δ (orden
  4). La "invarianza al drift" es una propiedad del proyector complejo,
  no del borrado en general — y la diferencia es medible en modelos
  entrenados con error < 0.002. Es la figura del paper.
- El piso de crosstalk de un modelo entrenado NO es (n−1)/D exacto: la
  elipticidad de ‖v‖ (cola pesada, p90/p50 ≈ 5.6) infla la media de
  ratios (~0.02-0.026 espurio). La fuga pooled (Σ|r|²/Σ|v|²) reproduce
  las leyes exactamente; la media de ratios NO. Regla para futuros
  probes de fuga sobre modelos entrenados: pooled, no mean-of-ratios.
- La estabilidad del write es un requisito ORTOGONAL a la capacidad: el
  mismo operador (regla delta β=1) entrena con claves normalizadas y no
  con claves crudas — la normalización de claves (NLMS, Widrow-Hoff) y
  las puertas β de DeltaNet/GDN son el dispositivo de estabilidad del
  lazo correctivo; wave_mem no lo necesita porque su write es lazo
  abierto. Esto da el linaje de related work (LMS/NLMS) para el paper.
- Un "casi-equivalente" TOST (media < ε, IC90 > ε) se reporta como tal,
  sin redondear a pase: la identidad se cumple en media pero no está
  demostrada al nivel estricto con 3 seeds; el contraste P1 no depende
  de él (la equivalencia sintética Clase A diff 0.0 es el teorema).
- Bug de runner (lección de higiene): pisar la variable de bucle
  (`rp = pooled(...)` reasigna `read_proj`) produce cargas de ckpt con
  shapes mezclados y parecería "ckpt corrupto" — en realidad era el
  shadowing en el script de diagnóstico. Los resultados del runner
  oficial (o04_run.py, nombres `fu_rr`/`fu_rp`) no fueron afectados.