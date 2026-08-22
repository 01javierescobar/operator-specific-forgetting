# O04b_test.md - Reconciliacion 2x2 del auditor: canal {complex, re} x drift {clave, estado} (sin reentrenar)

Template fijo. El runner escribe RESULTADO. El agente cierra VEREDICTO +
LECCION cuando la iteracion se cierra en `memory.md`.

## HIPOTESIS

El auditor aprobo O04 "en lo cientifico" pero senalo un defecto de
construccion del probe: la tabla de 4 leyes mezcla DOS intervenciones
fisicas distintas entre brazos. Su analisis de variantes (delta=0.5,
D=128, n=16, c=15/128):

- A: clave drifteada en lectura Y en producto externo -> Re 0.289
  cuadratica (su forma cerrada); complejo 0.0000 (solo posible bajo A).
- B: drift solo en la estimacion v^ (M acumulo fase desde la escritura;
  la clave viene limpia), producto externo con clave limpia -> Re 0.0870
  cuartica (sus numeros: 0.0870 = (1-cos d)^2 + c/2 + c(1-cos 2d)/4).
- C: Re aplicado al sustractor completo -> 0.281, "plana rota".

Consecuencia: el complejo es exactamente invariante a drift de RELOJ
(clave) pero fuga CUADRATICA bajo drift de ESTADO (2(1-cos d)(1+c) ~ 0.27
a d=0.5); el Re fuga CUARTICAMENTE bajo clave y (segun el auditor) bajo
estado. Ningun canal es robusto a ambos drifts: los dos cubren al otro.
Teorema de no-free-lunch para canales de borrado bajo drift.

Reconciliacion del agente (derivacion analitica, verificada sinteticamente
en este runner):

- La celda estado/complex = 2(1-cos d)(1+c) fija la SEMANTICA de estado:
  solo el item olvidado queda con phasor rotado e^{id}w en M (los demas
  items limpios); el erase LEE con e^{id}w (lo almacenado: el drift se
  cancela en la estimacion, wh = v_j + e^{-id}c) y REMUEVE con la clave
  limpia w (reloj actual). La construccion alternativa (item limpio, erase
  con producto externo drifteado) da la misma ley compleja pero leyes Re
  distintas — la sintetica adjudica.
- Los numeros Re del auditor (0.289, 0.0870) NO corresponden a la
  convencion fiel al modelo entrenado (Re truncado en wh Y en el readout
  de la query): 0.289 = sin^2 d + c/2 implica readout COMPLEJO en la query;
  0.0870 = (1-cos d)^2 + c/2 + c(1-cos 2d)/4 no es reproducible por ninguna
  celda limpia (probable convencion de medicion distinta / crosstalk no
  removido). Se reportan ambas convenciones en la verificacion sintetica;
  el gate d=128 usa la convencion FIEL (lo que ve el head entrenado).

## PREDICCION

Sobre modelos entrenados (6 ckpts wave, n_pairs [16,24], 320 eventos
FORGET/brazo/seed, fuga POOLED, d=64, c=(n-1)/D ~ 0.297, delta en
{0,.1,.2,.3,.4,.5}), convencion FIEL (Re truncado en wh y readout):

| celda | ley cerrada | valor a d=0.5 (c~0.297) |
|---|---|---|
| clave/complex | 0.0000 plano (proyector invariante) | 0.0000 (reproduce O04) |
| clave/re | sin^4 d + c(1-cos 2d)/4 | 0.0869 (reproduce O04: 0.0876-0.0885) |
| estado/complex | 2(1-cos d)(1+c) | 0.32 (cuadratica; forma cerrada del auditor) |
| estado/re | (1-cos d)^2 + c(1-cos d) | 0.0513 (cuartica) |

- EM sanity: clave-complex EM ~ 1.000 a todo d (el head decodifica el
  residual nulo... el item NO se borra con drift: el readout de la clave
  borrada sigue intacto -> EM ~ 1.000, no chance: en O04 la EM de reread
  complex era ~1.000 a todo delta); estado-complex EM cae con d (el item
  se borra mal); delta=0: EM ~ 1.000 en las 4 celdas (el replay externo
  debe reproducir el forward entrenado EXACTO: sanity del replay).
- Verificacion sintetica: las 4 leyes arriba bit-a-bit en D=64/n=20 y
  D=128/n=16, ambas convenciones (fiel y auditor) — adjudica la discrepancia
  de numeros Re del auditor (0.289 vs sin^2+c/2 bajo su convencion; 0.0870
  no reproducible: se reporta el desvio).

## DISEÑO

- Runner `tests/o04b_run.py` (resume por existencia de JSON): (1)
  verificacion sintetica de las 4 celdas x 2 convenciones x 2 grillas
  (D=64/n=20, D=128/n=16) x 6 delta; (2) probe sobre modelos entrenados:
  REPLAY EXTERNO exacto del forward wave (memoria externa; el modelo NO
  ejecuta decode_step — solo embedding/v_proj/out_proj/head), con
  inyeccion por celda:
  - clave: items escritos limpios; en t=pos_forget+1 el erase lee
    conj(e^{id}w)·M/D (Re truncado en canal re) y remueve e^{id}w ⊗ wh.
  - estado: SOLO el item olvidado (pending_id == clave olvidada) se
    escribe con phasor e^{id}w (los demas limpios); el erase lee con
    conj(e^{id}w)·M/D (drift cancelado) y remueve w ⊗ wh (clave limpia).
  - readout de la query SIEMPRE con clave limpia (Re truncado en re);
    write espurio de K_q replicado (comportamiento real del modelo, termino
    ~1/D^2 despreciable).
  - Sanity delta=0: EM ~ 1.000 (replay fiel) + fuga 0.000 en las 4 celdas.
- Presupuesto CPU: ~5-10 min (72 combos x replay 320 eventos). Salida:
  `outputs/wave_mem/o04b.json` + `o04b_run.log`.
- GATES.md: sustituir las filas de drift por el 2x2 (leyes de represent
  quedan intactas: son agnosticas a la variante).

## RESULTADO (runner)

`outputs/wave_mem/o04b.json` (68.8s, 3 seeds x 6 delta x 4 celdas x 2 convenciones
+ 2 grillas sinteticas):

Verificacion sintetica (bit-a-bit contra formas cerradas, 320 eventos pooled,
desvio < 0.01 en las 16 celdas):

| grilla | convencion | celda | fuga med. | pred. |
|---|---|---|---|---|
| D64/n20 | fiel | clave/complex | 0.0000 | 0.0000 |
| D64/n20 | fiel | estado/complex | 0.3160 | 0.3175 |
| D64/n20 | fiel | clave/re | 0.0871 | 0.0869 |
| D64/n20 | fiel | estado/re | 0.0513 | 0.0513 |
| D64/n20 | auditor | clave/re | 0.3789 | 0.3783 |
| D64/n20 | auditor | estado/re | 0.4285 | 0.4296 |
| D128/n16 | fiel | estado/complex | 0.2732 | 0.2735 |
| D128/n16 | fiel | clave/re | 0.0659 | 0.0663 |
| D128/n16 | fiel | estado/re | 0.0287 | 0.0293 |
| D128/n16 | auditor | clave/re | 0.2868 | 0.2884 |
| D128/n16 | auditor | estado/re | 0.3151 | 0.3178 |

Modelos entrenados (replay externo, 3 seeds, max |med-pred| en 6 delta):

| celda | ley | desvio max |
|---|---|---|
| clave/complex | 0.0000 plano | 0.0000 (0.0000 en 3/3 seeds a todo d) |
| estado/complex | 2(1-cos d)(1+c) | 0.0086 (0.3213-0.3261 vs 0.3175 @ d=0.5) |
| clave/re | sin^4 d + c(1-cos 2d)/4 | 0.0016 (0.0876-0.0885 vs 0.0869 @ d=0.5) |
| estado/re | (1-cos d)^2 + c(1-cos d) | 0.0014 (0.0499-0.0510 vs 0.0513 @ d=0.5) |

EM sanity: el patron replica O04 exactamente — complejo chance plano (0.00-0.02)
a todo d en ambas celdas (la probe consulta la clave BORRADA: a d=0 el item esta
removido, EM ~ chance es lo correcto; la prediccion del pre-registro "EM ~ 1.000
a d=0" era un error del agente, el replay es fiel: las fugas matchean las leyes
a 0.01); re/clave crece con d (0.02 -> 0.91-0.97 @ d=0.5: el head decodifica el
residual sin^2 d v); re/estado plano ~0.00 (residual (1-cos d)^2 v bajo el
umbral de decodificacion del head).

Desvio del auditor: su B-Re 0.0870 = (1-cos d)^2 + c/2 + c(1-cos 2d)/4 NO
reproducible por ninguna celda x convencion limpia (estado/re fiel = 0.0293 en
su grilla; = 0.3178 bajo su convencion de readout) — probable crosstalk no
removido en su simulacion; se reporta el desvio, no se persigue el numero.

## VEREDICTO (agente)

O04b CONFIRMA el 2x2 en modelos entrenados: las 4 leyes cerradas sin parametros
libres reproducidas con desvio < 0.01 en 6 puntos de delta x 3 seeds (replay
externo exacto, fuga pooled). El defecto de construccion senalado por el auditor
queda corregido: la semantica de cada celda esta fijada por inyeccion explicita
y adjudicada por la sintetica. Teorema de no-free-lunch VERIFICADO: complejo
invariante a drift de clave (0.0000), fuga cuadratica bajo estado; Re fuga
cuartica bajo clave, residual (1-cos d)^2 v bajo estado — la cobertura exige
ambos canales. Los numeros del auditor bajo su convencion (Re solo en wh):
clave/re = sin^2 d + c/2 reproduccion EXACTA (0.2868 vs 0.2884); su B-Re 0.0870
no es reproducible (se reporta). El gate d=128 queda definido por el 2x2
(GATES.md); represent intacto (agnostico a variante).

## LECCION (agente)

1. El probe de drift debe inyectar la variante por celda, no derivarla: las
   celdas clave y estado son dos intervenciones fisicas con leyes distintas;
   mezclarlas (O04) produjo una tabla correcta en fuga pero sin semantica.
2. Los numeros Re del auditor corresponden a una convencion de medicion
   distinta (readout complejo en la query); las leyes son de la convencion
   MEDIDA, no del operador: reportar convencion + forma cerrada juntas.
3. Bug tecnico: el producto exterior en el sintetico usaba unsqueeze(-1)/un-
   squeeze(1) sobre vectores 1-D (ambos (D,1)) -> elemento a elemento, no
   outer; el patrón resultante era "c-only" (item terms ausentes). El mismo
   patron en 3-D (batch) es correcto: verificar shapes en codigo 1-D.
4. EM sanity de un replay kind='erased': a d=0 la clave consultada fue
   borrada -> EM ~ chance es lo correcto, no 1.000 (la prediccion del
   pre-registro era un error de razonamiento del agente: el erase a d=0
   remueve el item EXACTO; el replay se valida por fuga, no por EM).