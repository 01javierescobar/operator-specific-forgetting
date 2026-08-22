# COMPENDIO ONDA: una línea por iteración (agrupadas)

Mapa condensado de todo lo aprendido en el lab ONDA (proyecto del modelo
ondulatorio, meta paper). Herencia: lab `../modelo-2` (S1-S110, cerrado).
Detalle en `memory.md` y `logs/O##_test.md`.

## Era 0 — Clonación del lab viejo (O00)

- **O00 (lab ONDA)**: clon bit-a-bit del harness, baselines y piezas del
  lab viejo (hashes idénticos verificados); docs adaptados al propósito
  paper (regla del claim, numeración O##); familias ondulatorias caídas
  (wave_iq S104, loopy_osc S109) archivadas como evidencia; snapshot
  baseline regenerado (suma EM 1.534).

## Era 1 — Claim del paper (O01+)

- **O01 (claim firmado)**: el eje de BORRADO como novedad ondulatoria —
  memoria por interferencia de fase (fasores superpuestos + cancelación)
  contra atención (0.067) y delta-rule (0.113) en `forget_retrieval`, a
  presupuesto igualado. Claim P1 + sub-claims C1 (teoría SNR/crosstalk) +
  C3 (contrato causal). Killer test: EM <= 0.113 @ d=128 mata P1 (Plan B/C
  de teoría o resultado negativo). Prototipo semilla: `wave_mem`.
- **O01b (auditoría externa, verificada)**: 3 hallazgos → claim
  reformulado. (1) C1 solo vive en geometría MATRICIAL C^{D×D} (verificado:
  SNR vectorial plana en D ~1/(n−1); matricial ∝ D); el erase por re-lectura
  en vectorial borra TODA la memoria (0.0). (2) "Antifase" == resta (0.0
  rel.) → α=1 analítico, v re-leído; α aprendida prohibida en v1. (3) Piso
  de crosstalk domina → métrica primaria = SELECTIVIDAD, EM secundario.
  Re-framing: novedad real = eje de medición (ley del olvido, extensión de
  L-064) + teoría residual post-erase (extensión no-formalizada de
  VSA/HRR, citado como linaje); wave_mem = instanciación. Nuevo brazo
  crítico: `delta_forget` (misma interfaz FORGET_ID, β=1). Pre-registro O02
  en CLAIM §8. Evidencia: `outputs/audit/audit_o01b_verification.json`.
- **O01c (segunda auditoría, verificada)**: el "killer test endurecido" era
  VACÍO: con ‖w‖²=D el erase por re-lectura compleja ES el operador delta
  β=1 (diff 0.0) → la equivalencia es TEOREMA de linaje (Clase A, va al
  paper como resultado de teoría); P1 solo puede vivir donde los operadores
  difieren (canal de clave + proyección de lectura) → Clases B (corrupción
  del canal de clave, O04) y C (claves aprendidas). Ley del olvido C1
  verificada: fuga(orig_v) ≈ (n−1)/D, función de n/D NO de D (plana en D a
  n/D fijo; techo 0.51–0.66 en n/D=0.5). Decidido: v1 complejo para Clase
  A, Re(·) para Clase B; killer endurecido. Evidencia:
  `outputs/audit/audit_o02_verification.json`.
- **O02 (prototipos + probes, G0 PASS)**: `wave_mem` (C^{D×D}, codebook de
  claves fijas, α=1, read_proj complex|re, interfaz estructural por
  markers, sin gates aprendidas) y `delta_forget` (delta vanilla, β=1,
  MISMA interfaz). G0 y Fase 0-lite PASS en ambos (sufijo bit-exacto,
  full==rollout, grads conectados); Clase A 0.0; canales 0.000/0.543/0.106
  vs pred 0/0.559/0.117; C1 pendiente 1.125 (pred 1.0); Clase B sintética
  = NULL test (wave inmune por construcción, delta 2º orden ~0.003, ambos
  agnósticos a ρ) → maquinaria ρ pasa a O04. Probes:
  `outputs/wave_mem/probes.json`. O03 (N1 5-task ambos brazos) habilitado.
- **O01d (auditoría O03, verificada, `audit_o03_verification.json`)**: 3
  defectos del plan O03 detectados y corregidos: (1) tipo de valor v REAL
  (ambigüedad 9× resuelta: Re deja (n−1)/(2D)=0.059, no 0.56); (2) 4º
  brazo wave_represent = RE-PRESENTACIÓN oracle en eval (comparador
  falsable de P1; se rechaza brazo entrenado con formato modificado);
  (3) Clase B′ = drift de fase δ (re-lectura 0.0000 exacta a todo δ;
  re-presentación (n−1)/D + (2−2cosδ); delta real protegido
  dimensionalmente (2/D)sin²δ; i.i.d. = régimen con cruce ~σ=0.5).
  wave_complex vs delta_forget = CONTROL con expectativa de identidad.
  P1 reformulado ("borrado por construcción exactamente robusto a drift
  de fase; re-presentación no"). C1: fit con intercepto, excluyendo n/D ≥
  0.4, R² (medido: slope 1.054, R² 0.959).
- **O03-auditoría (reconciliación i.i.d. + potencia N1)**: el cruce
  i.i.d. reportado (σ∈(0.3,0.5)) era convención de σ (por componente,
  potencia 2σ²); reconciliado con FORMAS CERRADAS (σ por elemento,
  potencia σ²): phase `reread=(1−e^{−σ²})²+(n−1)/D(1−e^{−σ²})`, additive
  `reread=(σ²/(1+σ²))²+(n−1)/D·σ²/(1+σ²)`; probe matchea la fórmula en
  cada σ. Cruce analítico σ* ≈ 0.64 rad (leak ~35-39% de amplitud) —
  FUERA de régimen operativo: la ventaja de reread cubre el régimen
  completo, la inversión es límite asintótico. Spec N1: contraste =
  wave_re vs los otros dos (Δ = (n−1)/(2D)); forget_retrieval n_pairs ∈
  [16, 24] y ≥300 eventos FORGET/brazo/seed; control entrenado por TOST
  (ε=0.02, IC 90%); probe de decodificabilidad (ridge sobre M post-erase
  → v_borrado, contrasta fuga-operador vs fuga-sistema). Auditoría O03
  autorizó N1 (2026-08-19) con 4 condiciones (conteo FORGET ≥300 post-hoc,
  EM de entrenamiento por brazo, n_pairs [16,24], ridge sobre M y residual
  de lectura). Fork de normalización del erase aditivo CERRADO (CLAIM §0b):
  erase = proyección idempotente `(I − ww^H/‖w‖²)`; fase = canónico O04,
  aditivo = robustez secundaria.
- **O03-N1 (ejecutado 2026-08-19, `outputs/wave_mem/n1.json`)**: wave
  complex Y re aprenden forget_retrieval a EM 1.000 (6/6 combos, n_pairs
  [16,24]); delta_forget NO aprende FR a d=64 (chance + NaN 2/3) →
  control TOST (wave_complex vs delta) NO COMPUTABLE en FR → rediseño del
  control escalado al auditor (recomendada: delta FALLIDO documentado;
  contraste P1 = wave_re vs wave_complex). Selectividad EM NULL
  (0.973±0.002 vs 0.967±0.010, piso chance); el contraste de operador
  vive en fuga_power del residual represent: (n−1)/D ≈ 0.315 (complejo)
  vs (n−1)/(2D) ≈ 0.166 (Re), mitad exacta ✓; reread 0.0000 ambos.
  Represent más selectivo que reread (0.994-1.0 vs 0.973-0.975). Ridge
  decodificabilidad cosine ≈ 0 (fuga-sistema = fuga-operador). Auditor
  aprobó Opción A + autopsia: fallo ESTRUCTURAL del write correctivo β=1
  (sweep lr×clip falla 6/6; gradiente sano; ‖S‖_F 8×10¹¹ en UNA pasada vs
  ~1174 wave entrenado; β=0.1 diagnóstico entrena a 1.000) → delta FALLIDO
  en FR con mecanismo documentado; tipo de escritura en CLAIM §0
  (equivalencia Clase A = SOLO erase). Held-out ✓ (1.000 con claves
  frescas). Gate d=128 reescrito en GATES.md (probes, EM=sanity).
  **O04 AUTORIZADO**: δ = rotación de fase común al FORGET, fuga_power
  primaria, ley δ² para represent, P1 = wave_re vs wave_complex.
- **O04 (ejecutado 2026-08-19, `outputs/wave_mem/o04.json`)**: Clase B′
  drift δ sobre modelos entrenados, 2×2×δ, fuga POOLED (la media de
  ratios infla ~0.026 por cola pesada de ‖v‖²). **4 leyes cerradas
  CONFIRMADAS en modelos entrenados**: reread-complex 0.0000 plano a todo
  δ ≤ 0.5 (P1 NO dañado: criterio de fallo 0.0000 < 0.02 @ δ=0.5);
  represent-complex (n−1)/D + 2(1−cosδ) (0.549-0.562 vs 0.542 @ δ=0.5);
  reread-re = sin⁴δ + c(1−cos2δ)/4 (**la enmienda "reread plano" era
  falsa para Re: el erase Re-truncado deja término de orden 4; ley
  corregida exacta: 0.0876-0.0885 vs 0.0869 @ δ=0.5**); represent-re
  c/2 + (1−cosδ)² (incrementos exactos, piso +10% elipticidad). Side arm
  delta_nlms (‖k‖=1, β=1): **ENTRENA FR 1.000/0.993/0.993, ‖S‖ acotada
  ~10² vs 10¹² del delta crudo** → teorema de estabilidad LMS verificado
  (lazo abierto estable vs β‖k‖² < 2); TOST resucitado casi-equivalente
  (Δ=0.0146 < ε en media, IC90 excede por 0.008); Opción B cerrada
  definitivamente. Gate d=128 reescrito en GATES.md (4 leyes + pooled).
- **O04b (ejecutado 2026-08-19, `outputs/wave_mem/o04b.json` + `o04b_run.log`)**: reconciliación 2×2 del auditor (canal × drift), replay externo
  exacto + sintética (2 grillas × 2 convenciones, 16/16 bit-a-bit). **Las 4
  leyes CONFIRMADAS en modelos entrenados (desvío < 0.01 en 6 δ × 3 seeds)**:
  clave/complex 0.0000 plano; estado/complex 2(1−cosδ)(1+c); clave/re
  sin⁴δ + c(1−cos2δ)/4; estado/re (1−cosδ)² + c(1−cosδ). Semántica de
  estado: solo el ítem olvidado con phasor; erase lee con clave drifteada
  (drift cancelado) y remueve con clave limpia. Convención auditor (Re solo
  en wh): clave/re = sin²δ + c/2 exacta; su B-Re 0.0870 RETIRADO (híbrido
  sin celda física limpia). **Framing corregido por el auditor: solo UNA
  celda con borrado verdadero (clave/complex, residual exactamente cero);
  las otras tres degradan a copia-decodificable (fuga en potencia subestima
  la funcional en >1 orden de magnitud) → teorema de garantía condicional;
  el 2×2 es stress test, no modo de fallo operativo (codebook fijo: ningún
  drift natural).** Bug técnico: unsqueeze(-1)/unsqueeze(1) sobre vectores
  1-D = producto elementwise no outer (patrón "c-only"); el mismo patrón en
  3-D es correcto. Gate d=128 = 2×2 (GATES.md) + 3 adiciones del auditor
  para O05 (EM con guarda de residual nulo, C1 a c emparejada, TOST tal
  cual). **O05 AUTORIZADO**: mini-smoke d=128 en Kaggle (3 brazos × 5
  seeds, FR n_pairs [32,48], gate 2×2, TOST wave_complex vs delta_nlms).
- **O05 (ejecutado 2026-08-19, kernel javierezcobar/onda-o05 v5, Kaggle
  2xT4, `outputs/onda_o05_kaggle/onda/outputs/wave_mem/o05.json`)**: **GATE
  d=128 CONFIRMADO.** 15/15 combos EM 1.000 (wave 60ep×600 satura;
  delta_nlms 150ep×1200 = receta n1, a 60ep×600 quedaba 0.07-0.98).
  max|med−pred| 4 leyes = 0.0049; **clave/complex = residual nulo en 6 δ ×
  5 seeds** (guarda EM); C1 c emparejada 4/4 (diff ≤ 0.0064) → las 4 leyes
  son las mismas en dos escalas; TOST casi-equivalente (mean −0.0031, IC90
  [−0.0206, 0.0144] cruza ε por 0.0006, reportado tal cual). Autocheck
  1.000 en todos los probes. Lecciones: presupuesto por familia de arq
  (saturación temprana ≠ para delta); filtros de path por segmento, no por
  substring ('re' in p matchea 'complex'). **El lab cierra con paper:
  olvido operador-específico, 4 leyes en dos escalas, estabilidad LMS,
  garantía condicional como teorema.** Backlog O07 (drift δ(t)=ω·lag) para
  un lab futuro.
- **REGLA PERMANENTE (O01d, heredada de O02)**: estado recurrente = lista
  de tensores por bloque + deltas OUT-OF-PLACE; NUNCA setitem in-place
  sobre estado (M[b]=..., r_persist[b]=...) — rompe el grafo en loops
  secuenciales y produce drift silencioso de checkpoints. Aplica a todo
  prototipo nuevo del lab.