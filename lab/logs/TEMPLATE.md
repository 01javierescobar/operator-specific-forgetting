# O##_test.md - TEMPLATE FIJO de entrada de experimento

Copia este template a `logs/O01_test.md` antes de correr nada. El runner
escribe la sección RESULTADO al final. El agente NO escribe el log; solo
cierra la entrada (VEREDICTO + LECCION) cuando la iteración se cierra en
`memory.md`.

## HIPOTESIS

(1-3 frases. Contra qué claim de `paper/CLAIM.md` va. Si no toca ningún
claim, marcarla explícitamente como exploratoria sin presupuesto de seeds.)

## PREDICCION

(Número concreto + criterio de éxito/fracaso ANTES de correr. Ej: "si
copy_reverse >= 0.90 y st >= 0.70 a d=64 cpu_quick seed 42, el mecanismo X
esquiva la ley L-056; si no, la ley se sostiene y la línea muere".)

## DISEÑO

(Variante, control, tareas, seeds, d_model, presupuesto según PROTOCOL.md
gasto 1→3→5-6. Mínimo: G0 antes de entrenar, Fase 0-lite del runner.)

## RESULTADO (runner)

(La escribe el runner: JSONs en `outputs/<variant>/`, tablas de EM por task,
taxonomía 3 ejes E1/E2/Δ. Sin JSON no hay resultado.)

## VEREDICTO

(¿Confirma, refuta o no toca el claim? Nivel de evidencia N0-N4. Un
single-seed es preliminar, NUNCA veredicto.)

## LECCION

(1 línea que entra comprimida a `memory.md`. Si no hay lección nueva, no
inventar una.)