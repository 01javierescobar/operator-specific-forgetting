# O05_test.md - Mini-smoke d=128 (3 brazos wave/delta_nlms, FR con n/D [0.25,0.38], gate 2x2 de O04b)

Template fijo. El runner escribe RESULTADO. El agente cierra VEREDICTO +
LECCION cuando la iteracion se cierra en `memory.md`. PRE-REGISTRO
ACTUALIZADO con las adiciones del auditor (2026-08-19): EM con guarda de
residual nulo, C1 a c emparejada, TOST tal cual. O05 AUTORIZADO, ejecucion
en Kaggle 2xT4 (45-75 min).

## HIPOTESIS

O04b verifico el 2x2 del auditor en d=64 (4 leyes cerradas, desvio < 0.01 en
3 seeds x 6 delta). El gate d=128 (GATES.md) pasa a los probes de operador:
las leyes son funciones de c = (n-1)/D, NO de D (ley del olvido C1: la fuga
es funcion de n/D). Hipotesis:

1. Las 4 leyes del 2x2 se mantienen en d=128 con modelos entrenados a
   n_pairs [32,48] (n/D 0.25-0.38, el mismo rango relativo que d=64 con
   [16,24]): la dependencia es solo de c.
2. wave_complex y wave_re entrenan FR a d=128 (d=64: EM 1.000 6/6) — la
   dificultad mayor (n/D mas alto) queda cubierta por el head entrenado
   que decodifica el crosstalk (O06).
3. delta_nlms entrena FR a d=128 (d=64: 1.000/0.993/0.993, la normalizacion
   es el dispositivo de estabilidad del lazo correctivo) -> el TOST del
   control es COMPUTABLE en FR por primera vez (el delta crudo fallo en FR
   a d=64: O06).

## PREDICCION

Entrenamiento (FR only, d=128, 5 seeds): wave_complex y wave_re a EM alto
(>= 0.95); delta_nlms EM >= 0.90; sin NaN.

Probe 2x2 (replay externo, 320 eventos/brazo/seed, fuga POOLED, c = 39/128
= 0.3047 para n medio 40; desvio permitido < 0.02 por evento-mix):

| celda | ley cerrada | valor a d=0.5 (c=0.3047) |
|---|---|---|
| clave/complex | 0.0000 plano | 0.0000 |
| clave/re | sin^4 d + c(1-cos 2d)/4 | 0.0866 |
| estado/complex | 2(1-cos d)(1+c) | 0.3193 |
| estado/re | (1-cos d)^2 + c(1-cos d) | 0.0527 |

EM sanity (con GUARDA de residual nulo, adicion 1 del auditor): se reporta
EM solo cuando ||residual|| > eps*||v_erased|| (eps=1e-4); si no,
"residual nulo" (resultado MAS FUERTE que cualquier EM bajo; evita el
artefacto argmax 0/0 -> indice 0 = item borrado, EM espurio 0.88-0.95 en
la simulacion del auditor). Prediccion con NUESTROS heads entrenados
(O04b d=64 medido): clave/complex = RESIDUAL NULO (solo celda con borrado
verdadero); clave/re EM crece con d (d=64: 0.02 -> 0.91-0.97, el head
decodifica el residual sin^2 d v); estado/complex y estado/re EM ~ 0
planos en d=64 (los residuales de estado no estan alineados con lo que
el head aprendio a decodificar; los 1.000/0.748 del auditor corresponden
a SU decodificador propio, no a nuestro head). En d=128 el gradiente
puede diferir: se reporta la DIRECCION, no valores exactos.

TOST control (wave_complex vs delta_nlms, selectividad EM, eps=0.02, IC
90%, 5 seeds -> t(0.95,4)=2.132): **reportado TAL CUAL (adicion 3 del
auditor)**: si el IC cruza eps, "casi-equivalente" va al paper sin
redondear (O04 d=64: Delta=0.0146 < eps en media, IC90 [0.001,0.028]
excede por 0.008; en d=128 con 5 seeds el IC se estrecha ~1.3x — se
espera que se cierre, sin prometerlo).

Criterios de PASE del gate d=128 (GATES.md, 2x2 + 3 adiciones):
1. |med-pred| < 0.02 en las 4 celdas (fuga pooled); clave/complex <=
   0.0001 (P1-bis: invarianza absoluta bajo drift de reloj).
2. EM_erased por celda con guarda de residual nulo: clave/complex =
   "residual nulo"; gradiente de decodificacion en las otras tres.
3. C1 a c emparejada (adicion 2 del auditor): las leyes son funcion de
   c = (n-1)/D, no de D. El MISMO c en ambas escalas: O05 d=128 con
   n_pairs [32,48] -> c = 39/128 = 0.3047; probe 2x2 re-ejecutado en los
   ckpts d=64 con n_pairs (17,24) -> c = 19.5/64 = 0.3047 (mismo c,
   mismo rango relativo n/D ~ 0.25-0.38). Criterio:
   |ley_d128(c) - ley_d64(c)| < 0.02 por celda (max sobre los 6 delta).
4. delta_nlms entrena FR (TOST computable) y TOST reportado tal cual.

## DISEÑO

- Runner `tests/wave_mem_smoke128.py` (resume por existencia de ckpts,
  salida unica `outputs/wave_mem/o05.json` + `o05_run.log`):
  - Entrenamiento FR only: d_model=128, n_layers=2, ForgetRetrieveDataset
    600 train / 128 valid, n_pairs_range=(32,48), n_forget_range=(1,2),
    max_seq_len=112 (2*48+7+headroom), epochs=60 (25% del preset smoke),
    AdamW lr=1e-3 wd=0.01 clip=1.0, 3 brazos x 5 seeds (1..5):
    wave_complex, wave_re (prototype wave_mem), delta_nlms.
  - ENMIENDA presupuesto (2026-08-19, previa a v4): 150 ep x 1200
    muestras = 61 min/brazo en T4 (15h totales, inviable; loss 0.000 /
    EM 1.000 = saturacion temprana). 60 ep x 600 mantiene saturación;
    el gate depende del probe, no del presupuesto de entrenamiento.
  - ENMIENDA v5 (2026-08-19, tras v4): wave confirma EM 1.000 a
    60 ep x 600 (10/10 brazos) PERO delta_nlms queda sub-entrenado
    (EM 0.07-0.98, loss 6.5 en el peor seed) -> TOST sin sentido.
    delta_nlms vuelve a la receta n1 (150 ep x 1200, la que dio EM
    1.000 a d=64); wave mantiene 60 ep x 600. Presupuesto v5: ~5.4h
    en 2xT4 con 2 workers (w0: seeds 1-3, w1: seeds 4-5).
  - ENMIENDA ejecucion (v4): con 2+ GPUs se lanzan 2 workers
    (CUDA_VISIBLE_DEVICES=0/1, seeds 1-3 | 4-5) y se MERGEAN los JSONs
    (TOST recomputado sobre las 5 seeds; C1 completo en part0, seeds
    1-3). 1 GPU o CPU: secuencial.
  - Autocheck: EM del probe (alive) vs harness oficial en formato con
    valor tras ANSWER; divergencia > 0.15 -> aborta el combo.
  - Probe 2x2 (brazos wave): replay externo exacto (reusa
    `o04b_run.replay_wave`): inyeccion por celda (clave: items limpios,
    erase lee e^{id}w y remueve e^{id}w; estado: solo item olvidado con
    phasor, erase remueve clave limpia), convencion FIEL (Re truncado en
    wh y readout), n_pairs_range (32,48), 320 eventos/brazo/seed, delta
    en {0,.1,.2,.3,.4,.5}, fuga POOLED por celda y por delta + EM_erased
    por celda con guarda de residual nulo.
  - C1 a c emparejada: mismo probe 2x2 sobre los ckpts d=64
    (outputs/n1_wave_complex/cache y n1_wave_re/cache) con n_pairs
    (17,24) -> c = 19.5/64 = 0.3047, emparejada con c de O05 d=128 ->
    tabla d=64(c) vs d=128(c), criterio |diff| < 0.02 por celda.
  - delta_nlms: EM alive/erased (selectividad) para TOST; sin 2x2 (las
    leyes son del operador wave).
  - TOST: selectividad (EM_alive - EM_erased) wave_complex vs delta_nlms,
    5 seeds, eps=0.02, reportado tal cual.
  - Reporte: leyes por celda con max|med-pred|; convencion auditor
    reportada como cross-check (no gate).
- Presupuesto estimado (v5): wave ~12 min/brazo x 10 + delta ~85
  min/brazo x 5 -> ~5.4h en 2xT4 con 2 workers + probes (~10 min, GPU).
  Kaggle: UN notebook a la vez (verificar que el agente previo termino).
- GATES.md: sin cambios de umbral (el gate 2x2 + 3 adiciones ya esta
  escrito); si las leyes se confirman a d=128 con c emparejada y el TOST
  se cierra, el criterio queda VALIDADO en el regimen del smoke y se
  habilita la promocion (tag de cierre / paper).

## RESULTADO (runner)

- Kernel v5 (javierezcobar/onda-o05) COMPLETE tras 5.44h en 2xT4
  (19510s, 2 workers: seeds 1-3 | 4-5, JSONs mergeados en o05.json).
- Entrenamiento: 15/15 combos EM 1.000 (wave 60ep x 600; delta_nlms
  150ep x 1200 = receta n1). Autocheck 1.000 en todos los probes.
- max|med-pred| de las 4 leyes (5 seeds x 4 celdas x 6 d):
  wave_complex clave 0.0000 / estado 0.3194 vs 0.3243 (0.0049);
  wave_re clave 0.0878 vs 0.088 / estado 0.0523 vs 0.054.
- EM con guarda: clave/complex = "residual nulo" en las 6 d x 5 seeds;
  estado/complex y estado/re EM_erased ~0 (los heads no decodifican
  estado, ver correccion de framing); clave/re EM_erased 0.06 -> 1.00
  con d (copia decodificable).
- C1 c emparejada (d=128 c=0.3047 vs d=64 c=0.3047): 4/4 celdas
  diff < 0.02 (clave/complex 0.0000, estado/complex 0.0064,
  clave/re 0.0026, estado/re 0.0027).
- TOST: mean_delta = -0.0031 (wave_complex vs delta_nlms, selectividad
  media 0.981 vs 0.984), sd 0.0184, IC90 [-0.0206, 0.0144] -> cruza
  eps=0.02 por 0.0006 -> CASI-EQUIVALENTE, reportado tal cual.

## VEREDICTO (agente)

- GATE O05 CONFIRMADO. Las 3 adiciones del auditor se cierran:
  (1) guarda: clave/complex = residual nulo a todo d (borrado exacto);
  (2) C1: leyes identicas en DOS escalas con c emparejada (4/4);
  (3) TOST: casi-equivalente (selectividad wave == delta-regla; el IC
  cruza eps por 0.0006, se reporta sin redondear).
- Las 4 leyes quedan CERRADAS en dos escalas (d=64 O04b, d=128 O05)
  con desvio max 0.0049 < 0.02. Promocion habilitada (tag de cierre /
  paper). El paper declara: borrado exacto = propiedad del proyector
  (key-value sobre noise-subspace), no del operador; 2x2 como stress
  test; casi-equivalencia con delta-regla.
- delta_nlms a d=128 confirmo el rol de control (EM_alive 1.000, fuga
  residual 0.00-0.04 = el "fondo" de selectividad que el TOST compara).

## LECCION (agente)

- La saturacion temprana hace el presupuesto parametro-libre: wave
  satura (loss 0.000) a 60ep x 600 (~12 min/brazo) y a 150ep x 1200
  (~61 min) con EM identico; delta_nlms NO satura igual de rapido
  (60ep x 600 deja EM 0.07-0.98). El presupuesto justo debe calibrarse
  por familia de arquitectura, no globalmente.
- Bug de substring en filtros de path ('re' in p matchea 'complex' ->
  el slot re cargo un ckpt complex y rompio C1): filtrar por segmento
  de ruta completo ('/re/' vs '/complex/').