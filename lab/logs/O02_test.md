# O02_test.md - Pre-registro prototipos wave_mem + delta_forget

Template fijo. El runner escribe RESULTADO. El agente cierra VEREDICTO +
LECCION cuando la iteracion se cierra en `memory.md`.

## HIPOTESIS

Contra `paper/CLAIM.md` v3 (O01c). C1 (ley del olvido `fuga(orig_v) ≈
(n−1)/D`, funcion de n/D no de D) y la equivalencia wave_mem ≡ delta β=1
(Clase A, teorema de linaje) ya tienen verificacion sintetica O01c; O02
construye los prototipos que los instancian y validan el contrato causal
(C3) + la maquinaria de medicion (probe de selectividad, Clase B sintetica).

## PREDICCION

- G0 (ambos brazos): invariante de sufijo bit-exacto, `full == rollout`
  bit-exacto, gradientes conectados a todos los params → PASS.
- Clase A (probe numerico): erase complejo wave_mem ≡ proyeccion delta
  β=1 con claves unitarias → diff rel < 1e-12. Fuga compleja 0.0;
  Re(·) ≈ 0.5 + (n−1)/(2D); orig_v ≈ (n−1)/D.
- C1 (probe n/D): fuga(orig_v) ≈ (n−1)/D plana en D a n/D fijo; techo en
  n/D = 0.5. Fit de exponente ≈ 1.0.
- Clase B sintetica: con ruido de clave correlado (rho=0.9), la fuga del
  ítem borrado difiere entre canal complejo y Re(·) — si es identica a
  precision de maquina, el probe no discrimina y hay que redisenar la
  intervencion (hallazgo O01c-2, no refutacion).
- Plumbing de selectividad: se computa EM_vivo, fuga y selectividad sobre
  modelos random-init (sin entrenar: N1 no es decision, solo validacion de
  la maquinaria; valores ~ chance, no se interpretan).

Criterio de exito O02: G0 PASS en ambos brazos + probes numericos
consistentes con C1/Clase A + JSON en `outputs/wave_mem/` → O03 (N1) tiene
luz verde. Si G0 falla: arreglar contrato antes de tocar el claim.

## DISEÑO

- `prototypes/wave_mem/model.py`: acumulador matricial C^{D×D} por bloque,
  codebook de claves fijas unitarias (buffer), escritura estructural
  (t>=9 y t-1>=9) `M += w_{t-1} (x) v_c(emb_t)`, erase estructural
  (t-1==FORGET_ID) `M <- M - w_t (x) (w_t^H M)/D` con `read_proj` en
  {complex, re}, lectura estructural (t-1==QUERY_ID) con `r` persistente
  hasta ANSWER, head sin atar (L-015). Contrato `prefill/decode_step`
  O(1), estado {M_b, r_b, prev_tok}. Sin gates aprendidas, sin RoPE, sin
  atencion.
- `prototypes/delta_forget/model.py`: regla delta secuencial vanilla
  (K/V/Q aprendidas, sin conv1d/RoPE/gating GDN), erase estructural con
  β=1 forzado `S <- S - k_t (k_t^T S)/||k_t||^2` (equivalente al erase
  wave), lectura con q aprendida persistente. MISMA interfaz de markers.
- `tests/wave_mem_smoke.py`: G0 (sufijo + paridad + gradientes), Fase
  0-lite (5 pasos AdamW en forget_retrieval, curva de loss sin NaN),
  probe Clase A (equivalencia numerica wave vs delta), probe n/D (grid
  D x n, fuga por canal, fit exponente), probe canales (complex/re/orig_v
  a D=128 n=16), probe Clase B sintetica (ruido correlado rho ∈
  {0,0.5,0.9}, fuga complejo vs Re), probe selectividad (plumbing sobre
  random-init; datasets con queries a claves BORRADAS y VIVAS).
- Presupuesto: CPU, d_model=64 para G0/Fase 0-lite, D∈{64,128,256} en
  probes sinteticos. Salidas: `outputs/wave_mem/probes.json` (unico,
  dedup por config hash; index.json no aplica: probes no se regeneran en
  loop).
- Seeds: fijas (7 para numericos, 42 para random-init). Sin entrenar en
  O02: el entrenamiento es O03 (N1 5-task ambos brazos, misma grilla).

## RESULTADO (runner)

`outputs/wave_mem/probes.json` (probes completas) — ejecutado
2026-08-18, d_model=64, n_layers=2, device=cpu, seed=7.

- **G0: PASS en los 3 modelos** (wave complex, wave re, delta_forget):
  sufijo bit-exacto, `full == rollout`, gradientes conectados a todos los
  params.
- **Fase 0-lite: PASS en los 3** (5 pasos AdamW sobre forget_retrieval:
  loss finita y decreciente, sin NaN).
- **Clase A (equivalencia)**: `M − w⊗(w^H M)/D ≡ M − w(w^H M)/‖w‖²` con
  ‖w‖²=D → max_abs_diff **0.0** (teorema de linaje confirmado).
- **Canales (D=128, n=16, 20 trials)**: complejo 0.0000 (pred 0.0) · Re(·)
  0.543 (pred 0.5 + (n−1)/2D = 0.5586) · orig_v 0.1057 (pred (n−1)/D =
  0.1172).
- **C1 (n/D)**: fuga(orig_v) ≈ (n−1)/D, plana en D a n/D fijo (0.099/
  0.103/0.135 para 64/128/256 @ n/D=0.125); pendiente log-log 1.125
  (pred 1.0); techo en n/D=0.5 (0.51–0.66).
- **Clase B sintética (NULL test)**: wave fuga_complex 0.0000 en los 3 ρ
  (inmune por construcción: el codebook no se corrompe); delta fuga
  ~0.0028 (2º orden, canal aprendido); spread ρ < 0.03 en ambos →
  agnosticismo de forma con claves i.i.d. → maquinaria ρ pasa a O04.
- **Selectividad (plumbing, random-init, sin interpretar)**: métricas
  computan (EM_vivo 0.03–0.22, fuga 0.0–0.03, selectividad 0.03–0.22).

## VEREDICTO

(O02 = construcción de instrumentos, N0. Los prototipos pasan el contrato
causal y los probes numéricos confirman C1 y la equivalencia Clase A. Sin
entrenamiento aún: P1 no se toca. O03 (N1 5-task ambos brazos) habilitado.)

## LECCION

(O02 abierta: la escribe el consolidor al cerrar la iteración.)