# O01_test.md - Definición del claim del paper

## HIPOTESIS

El lab viejo cerró con la ley del binding (L-056/064): el régimen compra
retrieval solo a la atención; ningún operador no-atencional discrimina
retrieval a d=64. La familia ondulatoria murió dos veces (wave_iq S104,
loopy_osc S109) persiguiendo retrieval/capacidad. O01 postula que el eje
que nadie midió con ondas es el **borrado asociativo** (`forget_retrieval`,
donde atención 0.067 y delta-rule 0.113 fallan): la cancelación de fase es
un erase POR CONSTRUCCIÓN de la onda, no una puerta aprendida.

## PREDICCION

- P1: memoria por interferencia (fasores superpuestos + cancelación de
  fase) supera 0.113 en `forget_retrieval` a d=64 y d=128 con presupuesto
  igualado; barra interna de señal: >= 0.30 @ d=64 (3x el mejor ancla).
- Si EM <= 0.113 @ d=64 (2+ seeds): descarte inmediato, Plan B/C.
- C1: SNR de crosstalk ∝ D/(2·n_pairs) verificable por probe (±10%).

## DISEÑO

Iteración de papel (no de código): `paper/CLAIM.md` firmado como claim
vigente. El prototipo `wave_mem` se diseña contra él en O02 (G0 + Fase
0-lite antes de entrenar). Baselines congelados como anclas: transformer
(0.067 forget), delta-rule (0.113, regenerable desde `_pieces/`).

## RESULTADO (runner)

N/A — iteración de definición, sin runner. Artefactos: `paper/CLAIM.md`.

## VEREDICTO

Claim definido y firmado como VIGENTE (N0 documental). Criterios de
muerte explícitos en la sección 2 del CLAIM. Presupuesto: O02-O06.

## LECCION

Primero el claim (afirmación + experimento que lo mata + evidencia
objetivo), después el prototipo. El diseño de wave_mem nace de la sección 6
del CLAIM, no al revés.

---

## ANEXO O01b — Auditoría externa (verificada por script propio)

Tres hallazgos críticos, confirmados con verificación numérica
independiente (`outputs/audit/audit_o01b_verification.json`):

1. **Geometría de C1**: el diseño decía `M ∈ C^D` y `w_k ⊗ v` a la vez.
   Verificado: en geometría vectorial la SNR es ~1/(n−1) y PLANA en D
   (0.35/0.36/0.34 para D=64/128/256 con n=4); en matricial crece ∝ D
   (33/50/126). La ley D/(2n) solo vive en matricial → CLAIM fija
   geometría matricial C^{D×D} en O02.
2. **"Antifase" == resta**: diff relativa 0.0 a precisión de máquina.
   Agravante propio: el erase por re-lectura en geometría VECTORIAL borra
   TODA la memoria (residual 0.0, porque `w_k∘(conj(w_k)∘M) = M`) —
   peor que lo reportado por la auditoría. → erase α=1 fijo con v re-leído
   en matricial; α aprendida prohibida en v1 (se auto-refuta: sería puerta
   aprendida).
3. **Piso de crosstalk domina el borrado**: el residual post-erase está en
   el nivel del crosstalk de los pares restantes, no en la fidelidad del
   borrado → métrica primaria = SELECTIVIDAD (EM_vivo − tasa_de_fuga), EM
   absoluto secundario.

Re-framing adoptado (ver CLAIM.md §0): la novedad real es el **eje de
medición** (el olvido como propiedad operador-específica a presupuesto
igualado, extensión de L-064) + **teoría del residual post-erase con
claves aprendidas y borrado por re-lectura** (extensión no-formalizada de
VSA/HRR). wave_mem es instanciación, no mecanismo nuevo. Se añade el brazo
`delta_forget` (misma interfaz FORGET_ID, β=1) como comparador crítico.

## VEREDICTO (revisado O01b)

GO condicionado. Claim reformulado con: geometría matricial fija, erase
analítico, métrica de selectividad, brazo comparador con interfaz idéntica,
pre-registro O02 (CLAIM.md §8), y VSA/HRR citado explícitamente como
linaje. Killer test intocado en esencia (sigue siendo falsable con umbral),
pero ahora contra el comparador justo.