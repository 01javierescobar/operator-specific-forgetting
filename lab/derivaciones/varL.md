# Derivación Var(L̂) — documento de trabajo

Objetivo: fórmula cerrada para sd(L) bajo el ensamble (claves iid uniformes +
valores iid gaussianos), validada contra MC, para dar a la tolerancia 0.02
interpretación kσ. Filosofía del lab: cada lema se verifica con un micro-MC
propio antes de usarlo.

## 1. Setup

- M = Σ_i w_i v_i^⊤, w_i ∈ C^D iid unit-modulus fases uniformes, v_i ~ N(0,I_D).
- State drift ítem j: almacenado (e^{iδ}w_j)v_j^⊤; re-read con w̃_j = e^{iδ}w_j;
  sustracción con w_j.
- x_j = γ Σ_{k≠j} β_jk v_k^⊤, γ = 1−e^{iδ}, β_jk = w_j^H w_k / D.
- Q = Σ_j ‖x_j‖², L = Q / Σ_k S_k, S_k = ‖v_k‖².
- Canal Re: truncamiento en la estimación y en la lectura.

## 2. Restricciones que la fórmula debe satisfacer (MC medido)

### 2a. Escalado en D (c ≈ 0.297 fijo, n ∝ D)
sd(estado/complex @ δ=0.5): D=32 → 0.01964; D=64 → 0.00965; D=128 → 0.00471.
⇒ sd ∝ |γ|²/D exacto. κ(c=0.2969) := sd·D/|γ|² ≈ 0.0394 (D=64).

### 2b. Escalado en n (D=64 fijo)
n = 10/20/40/80 ⇒ sd@δ=0.5 = 0.01157 / 0.00909 / 0.01164 / 0.01439.
U-shaped con mínimo cerca de c ≈ 0.30. Ni constante ni √n ni lineal.
⇒ hay al menos dos términos compitiendo con dependencias distintas en c.

### 2c. Descomposición por fuente (D=64, n=20, δ=0.5)
solo claves 0.00799; solo valores 0.00749; ambos 0.00874.
⇒ contribuciones comparables; NO vale derivar condicionando solo en valores.
Canal Re: dominancia de claves ~3.7×.

### 2d. Igualdades estructurales
sd(clave/re) ≈ sd(estado/re) a todo δ y D (ej D=64@0.5: 0.00355 vs 0.00338).
sd(clave/complex) = 0 literalmente (identidad proyectiva).
E[L] invariante entre modos A/B/C ✓.

## 3. Momentos exactos disponibles (verificados por counsel OpenAI)

- E[|w_j^H w_k|²] = D; E[|w_j^H w_k|⁴] = 2D² − D (exacto, ∀D finito).
- E[β_jk conj(β_jl)] = 0 (k≠l); E[|β_jk|²|β_jl|²] = 1/D² (k≠l, cond. en w_j).
- Valores: E[(v_k·v_l)²] = D para k≠l; E[S_k²] = D(D+2).

## 4. Plan de lemas (cada uno con micro-MC propio)

L1. E[Q] = |γ|²(n−1)/D · ΣS. [ya implícito en gates; re-verificar]
L2. Tabla de sextos/cuartos momentos cruzados de β necesarios para E[Q²]:
    patrones de emparejamiento de fases que sobreviven (consejero antigravity:
    términos recirculantes β_jj'β_j'j* = |β_jj'|² sobreviven).
L3. Términos mismo-j de E[Q²]: diagonal (E|β|⁴) vs cruzados (1/D²) ponderados
    por T_kl². Micro-MC: fijar W, muestrear solo V → compara Var_V(Y_j).
L4. Términos j≠j': covarianza vía columnas compartidas (v_k·v_m) y pares
    recirculantes. Micro-MC: Var(E[Y_j|W]) vs E[Cov] descomposición empírica.
L5. Ensamblar Var(L) = [Σ_j Var(Y_j) + Σ_{j≠j'} Cov(Y_j,Y_j')] / (ΣS)²,
    identificar qué término da el mínimo U-shaped en c ≈ 0.3.
L6. Análogo canal Re (doble truncamiento); explicar igualdad empírica 2d
    entre ambas celdas Re y la dominancia de claves (2c).
L7. Fórmula final sd(L) = |γ|²/D · g(c) con g cerrada; ajuste conjunto contra
    2a+2b (≥8 puntos, sin parámetros libres salvo constantes universales).

## 5. Notas de consejeros

- OpenAI: cuartos momentos exactos arriba; advertencia Cov(Y_j,Y_j') no
  despreciable; para Paper II: τ* cerrado existe para γ escalar determinista,
  decay coordenada-a-coordenada exige supuestos fuertes.
- Antigravity: estructura O(n³/D²) candidata CONTRADICHA por MC 2b a c fijo
  (predeciría √n/D ≡ 1/√D); su término recirculante O(n²/D²) sí es compatible
  con 1/D. Referencias: Magnus 1986 (moments of ratios of quadratic forms),
  Goodman Statistical Optics, Pitman/Geary ratio-independence.
- Discrepancia abierta: resolver con L4/L5 + barrero n extendido antes de
  escribir cualquier ecuación en el suplemento.


## 6. Hallazgos estructurales de sesion (delta-method, tests/varL_delta_method.py)

Identidad validada empiricamente (error < 1%):
sd^2(L) = [Var(Q) + E[L]^2 Var(SumS) - 2 E[L] Cov(Q,SumS)] / Sbar^2, Cov ~ 0.

Pieza ya cerrada: E[L]^2 Var(SumS)/Sbar^2 = 2|gamma|^4 (1+c)^2 / D.
Estado/complex @0.5 D=64: aporta sd-parcial 0.0127; Re: misma formula.

Hipotesis H (crosstalk auto-concentrado, Var(Qx)=0) REFUTADA:
prediccion |gamma|^2 sqrt(2nD)/Sbar = 0.00976 vs medido 0.00366 en clave/re.
=> Var(Qx) domina en canales Re y es material en complex.

Descomposicion correcta del numerador:
Q = |gamma|^2 [SumS (target determinista) + Qx (crosstalk)],
E[Qx]/Sbar = |gamma|^2 c verificado; queda derivar Var(Qx) conjunta sobre
claves y valores, mas Cov(SumS, Qx). Pistas: en Re la parte de claves
domina (~3.7x, secion 2c); el minimo U-shaped en c sugiere cancelacion
parcial entre terminos crecientes y decrecientes en c.

## 7. Estado abierto

- Grind completo de Var(Qx) con momentos seccion 3 + momentos gaussianos
  de T_kl (E[T^2]=D para k!=l, E[S_k^2]=D(D+2)); auditoria cruzada de
  consejeros antes de cerrar.
- Barrido extendido sugerido: variar sigma_v para separar contribuciones.


## 8. Lemas medidos (tests/varL_lemmas.py, estado/complex @ d=0.5, D=64)

- E[Y_j] = |g|^2 (n-1)/D verificado en n=10/20/40/80 (<5%: fluctuacion de
  normas de valores).
- Cov(Y_j,Y_i) promedio / Var(Y_j) = -0.101 / -0.050 / -0.025 / -0.012 para
  n=10/20/40/80 => Cov = -Var * 0.95/n con precision notable.
- Var(Y_j) superlineal: 2.09e-4 / 6.17e-4 / 1.80e-3 / 6.37e-3 (~n^1.55).
- Entonces Var(Qx) = n Var(Y) [1 - 0.95] aprox = 0.05 n Var(Y): la suma de
  varianzas individuales se cancela casi completa por la covarianza
  negativa. Este factor de cancelacion es el corazon algebraico restante.
- Siguiente grind cerrado: (i) derivar Var(Y_j) cerrada via Wishart-ish
  momentos condicionales; (ii) derivar el coeficiente -0.95/n desde los
  terminos recirculantes beta_ji beta_ij* (antigravity) + valores
  compartidos; (iii) ensamblar g(c) y validar contra secciones 2a+2b.


## 9. Sesion grind 1: reduccion rank-one y anclas

Reduccion exacta: x_j = (g/D) a_j^T V (a_j = vector alphas excluyendo j,
V = matriz de valores n x D sin v_j). Con CLT sobre D, a_j ~ CN aprox:
u = a^T V se escribe u = V^T p + i V^T q, p,q iid N(0,I_n).

- E[Y_j] = |g|^2 (n-1)/(D-2): clava MC al 0.1% (0.07503 vs 0.07495).
  La version con D-2 (inverse-chi2 del denominador) es la correcta.
- Ancla: CV(Y_j) = 0.33 a D=64,n=20,d=0.5.
- MURO detectado: Var(Z) via 'coordenadas gaussianas independientes'
  da 14x menos que MC. Causa: u = V^T p + i V^T q comparte V entre los
  dos terminos => E[u^4] tiene termino cruzado 2*(2(n-1)D)^2-ish por la
  norma compartida SumS. La cuenta correcta: E[||u||^4] con u = V^T p +
  i V^T q, p,q,V conjuntos:
    E[||u||^2] = 2(n-1)D (sin v_j)
    E[||u||^4] = 4 E[||V'||_F^4] + 2(2(n-1)D)^2-ish + terminos de exclusion
  con ||V'||_F^4 = (SumS)^2, E = (n-1)^2D^2 + 2(n-1)D.
- Pendiente cerrar: ensamblar Var(Z) exacta, dividir por D^4, multiplicar
  por |g|^4, validar contra sd(Y_j)=0.02483 medido; luego el coeficiente
  -0.95/n de Cov (mismo mecanismo de norma compartida entre items).
