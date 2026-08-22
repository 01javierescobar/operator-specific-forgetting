import random
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
QUERY_ID = 4
ASSIGN_ID = 5
PICK_ID = 6  # legacy (no usado en tests; src/ reason_smoke lo usa)
FORGET_ID = 6  # alias PICK_ID: id libre rehusado como marker FORGET
GOTO_ID = 7
ANSWER_ID = 8
RESERVED_SPECIAL = 9


VOCAB_MQAR_SIZE = 89
N_KEYS_MQAR = 40
N_VALS_MQAR = 40
K_OFFSET_MQAR = RESERVED_SPECIAL
V_OFFSET_MQAR = RESERVED_SPECIAL + N_KEYS_MQAR

VOCAB_COPY_SIZE = 19
N_DIGITS_COPY = 10
DIGIT_OFFSET = RESERVED_SPECIAL

VOCAB_ST_SIZE = 17
N_ENTITIES = 5
N_LOCATIONS = 3
ENT_OFFSET = RESERVED_SPECIAL
LOC_OFFSET = RESERVED_SPECIAL + N_ENTITIES

# Dyck-2 (S81): 2 tipos de parens (A/B). Tarea de cierre del top del stack
# (Suzgun et al. 2019): el prefijo es una secuencia parcialmente balanceada
# que termina con pila NO vacia; la respuesta es el cierre que toca
# (CLOSE del tipo del top). Anticorto de 'copiar el ultimo token': el
# ultimo token puede ser un CLOSE interior, no el par del top; y
# n_opened >= 2 fuerza a rastrear el stack completo, no solo el ultimo.
VOCAB_DYCK_SIZE = 16
N_DYCK_TYPES = 2
DYCK_OPEN_BASE = RESERVED_SPECIAL  # 9, 10 (OPEN A/B)
DYCK_CLOSE_BASE = RESERVED_SPECIAL + N_DYCK_TYPES  # 11, 12 (CLOSE A/B)

# forget_retrieval reusa el vocabulario MQAR (89 tokens). Mismas keys/vals:
# store de n_pairs pares k,v + instruccion FORGET k_i + QUERY k_j (j != i)
# + ANSWER v_j. Tests si el modelo sabe descartar par borrado y aun
# recuperar par presente (GDN acumula, no borra; s4d_carve tiene erase
# interno pero no integrado a una task).
VOCAB_FORGET_SIZE = VOCAB_MQAR_SIZE

# tiny_program (S61): vocabulario autocontenido, estilo state_tracking.
# Semantica: FACTS (FACT var val) + RULES (RULE rid lhs rhs) + query
# (QUERY v0 r1..rd ANSWER). El modelo recibe el prefijo SIN el valor de
# respuesta (prefix_answer=True: el target vive solo en y, nunca en x) y
# debe encadenar reglas POR ID y responder con el FACT del var final.
# depth = cantidad de rule-ids en la query; d2/d3 son buckets de la MISMA
# task (bucket_fn), no tasks separadas (S61: sin doble peso en la barra).
FACT_ID = 9
RULE_ID = 10
VOCAB_TINY_SIZE = 59
N_VARS_TINY = 16
N_VALS_TINY = 16
N_RULES_TINY = 16
VAR_OFFSET_TINY = 11
VAL_OFFSET_TINY = 27
RID_OFFSET_TINY = 43


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def gen_mqar_sample(rng: random.Random, n_pairs: int = 3, hops: int = 1) -> List[int]:
    if hops == 1:
        keys = rng.sample(range(N_KEYS_MQAR), n_pairs)
        vals = rng.sample(range(N_VALS_MQAR), n_pairs)
        store = []
        for k, v in zip(keys, vals):
            store.append(K_OFFSET_MQAR + k)
            store.append(V_OFFSET_MQAR + v)
        idx = rng.randrange(n_pairs)
        target_val = V_OFFSET_MQAR + vals[idx]
        return ([BOS_ID] + store +
                [SEP_ID, QUERY_ID, K_OFFSET_MQAR + keys[idx], ANSWER_ID, target_val])
    # hops == 2: genuine 2-hop chain via value-as-key.
    # Chain is always exactly 2 hops regardless of n_pairs:
    #   query k_a=ids[0] -hop1-> v_a=ids[1] (acts as key k_b for hop 2)
    #                                       -hop2-> v_b=ids[2]
    # Store contains the two chain pairs (ids[0]->ids[1], ids[1]->ids[2]) PLUS
    # (n_pairs-2) distractor pairs whose key/value are random ids NOT in the chain.
    # Query gives only k_a; model must retrieve ids[1] from store, then look up
    # ids[1] as a key in the store to find ids[2].
    ids = rng.sample(range(max(N_KEYS_MQAR, N_VALS_MQAR)), 3)
    chain_keys = [ids[0], ids[1]]
    chain_vals = [ids[1], ids[2]]
    k_query = chain_keys[0]
    target_val = chain_vals[-1]
    # Build distractor pairs from remaining ids not in the chain.
    pool = [i for i in range(max(N_KEYS_MQAR, N_VALS_MQAR)) if i not in ids]
    rng.shuffle(pool)
    n_dist = max(0, n_pairs - 2)
    distract_keys = pool[:n_dist]
    distract_vals = pool[n_dist:2 * n_dist] if 2 * n_dist <= len(pool) else pool[:n_dist]
    # Assemble pair list (chain + distractors), then shuffle pair order so the chain
    # is not always at the front of the store.
    pairs = list(zip(chain_keys, chain_vals)) + list(zip(distract_keys, distract_vals))
    rng.shuffle(pairs)
    store = []
    for k, v in pairs:
        store.append(K_OFFSET_MQAR + k)
        store.append(V_OFFSET_MQAR + v)
    # Query gives only k_a; the model must do hop 1 (retrieve v_a from store),
    # then hop 2 (treat v_a as key, retrieve v_b from store).
    return ([BOS_ID] + store +
            [SEP_ID, QUERY_ID, K_OFFSET_MQAR + k_query, ANSWER_ID, V_OFFSET_MQAR + target_val])


def gen_forget_retrieve_sample(rng: random.Random, n_pairs: int = 6,
                               n_forget: int = 1) -> List[int]:
    # Quinta task (S45): store + instruccion FORGET k_i + QUERY k_j (j != i)
    # + ANSWER v_j. Tests si el modelo descarta par borrado y aun recupera
    # par presente. GDN/KDA/Qwen3-Next acumulan, no borran; s4d_carve tiene
    # erase interno pero no ve instruccion FORGET explicita en tokens. Esta
    # task debil comun deberia revelar donde el proximo mecanismo debe
    # brillar (ver AGENTS.md "Las 5 sub-tasks").
    #
    # Formato:
    #   BOS [k1 v1 ... kn vn] SEP FORGET k_i1 FORGET k_i2 ... SEP
    #   QUERY k_j ANSWER v_j
    #   donde j no esta en {i_1, ..., i_m} (recupera par NO borrado).
    #
    # Anti-atajos:
    # - n_pairs >= 5 para que "v de cualquiera != v_i" no baste.
    # - pares (k, v) con permutacion random sobre max(N_KEYS, N_VALS);
    #   sin permutacion identity para evitar "v = k" como atajo.
    # - n_forget <= n_pairs - 2 (siempre queda par presente a recuperar).
    assert n_forget <= n_pairs - 2, 'need >=2 un-erased pairs for valid query'
    assert n_pairs >= 3, 'need >=3 pairs'
    ids_k = rng.sample(range(N_KEYS_MQAR), n_pairs)
    ids_v = rng.sample(range(N_VALS_MQAR), n_pairs)
    # Si un v_a == k_a por coincidencia de id, no es un atajo valido porque
    # el modelo no puede saber "k -> v" hasta leer el store (no hay forma
    # de derivar v de k por identidad cross-offset).
    erased_idx = rng.sample(list(range(n_pairs)), n_forget)
    erased_set = set(erased_idx)
    remaining = [i for i in range(n_pairs) if i not in erased_set]
    q_idx = rng.choice(remaining)
    target_val = ids_v[q_idx]
    # Construye store con orden de pares aleatorio (no revele erased vs alive).
    pair_order = list(range(n_pairs))
    rng.shuffle(pair_order)
    store = []
    for p in pair_order:
        store.append(K_OFFSET_MQAR + ids_k[p])
        store.append(V_OFFSET_MQAR + ids_v[p])
    # Orden de FORGET aleatorio (si multiples erased).
    forget_block = []
    for ei in erased_idx:
        forget_block.append(FORGET_ID)
        forget_block.append(K_OFFSET_MQAR + ids_k[ei])
    if len(forget_block) == 0:
        # n_forget=0: degenerado; insertamos un marker neutro para no romper
        # el parseo. (No usado en practica: n_forget>=1.)
        forget_block = [FORGET_ID, K_OFFSET_MQAR + ids_k[0]]
    # Forzar el q_idx NO erased ya garantizado por 'remaining'.
    return ([BOS_ID] + store +
            [SEP_ID] + forget_block +
            [SEP_ID, QUERY_ID, K_OFFSET_MQAR + ids_k[q_idx], ANSWER_ID,
             V_OFFSET_MQAR + target_val])


class ForgetRetrieveDataset(Dataset):
    def __init__(self, n_samples: int, seed: int = 0,
                 n_pairs_range: Tuple[int, int] = (5, 9),
                 n_forget_range: Tuple[int, int] = (1, 1)):
        self.samples = []
        rng = random.Random(seed)
        seen = set()
        attempts = 0
        while len(self.samples) < n_samples and attempts < n_samples * 4:
            attempts += 1
            n_pairs = rng.randint(*n_pairs_range)
            # n_forget en [1, n_pairs-2] para garantizar >=2 pares vivos.
            max_f = max(1, n_pairs - 2)
            lo_f, hi_f = n_forget_range
            lo_f = max(1, min(lo_f, max_f))
            hi_f = max(lo_f, min(hi_f, max_f))
            n_forget = rng.randint(lo_f, hi_f)
            seq = gen_forget_retrieve_sample(rng, n_pairs=n_pairs,
                                             n_forget=n_forget)
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            self.samples.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# tiny_program (S61): razonamiento composicional sobre reglas explicitas.
# ---------------------------------------------------------------------------

def tiny_program_build_seq(rng, facts, rules, query_var, query_rules):
    # facts: list[(var, val)], rules: list[(rid, lhs, rhs)]. Reordena
    # facts y rules internamente (anti-atajo de orden) y construye el
    # prefijo; termina en ANSWER_ID (el valor de respuesta NO va en x).
    f = list(facts)
    r = list(rules)
    rng.shuffle(f)
    rng.shuffle(r)
    seq = [BOS_ID]
    for var, val in f:
        seq += [FACT_ID, VAR_OFFSET_TINY + var, VAL_OFFSET_TINY + val]
    seq.append(SEP_ID)
    for rid, lhs, rhs in r:
        seq += [RULE_ID, RID_OFFSET_TINY + rid, VAR_OFFSET_TINY + lhs, VAR_OFFSET_TINY + rhs]
    seq.append(SEP_ID)
    seq += [QUERY_ID, VAR_OFFSET_TINY + query_var]
    for rid in query_rules:
        seq.append(RID_OFFSET_TINY + rid)
    seq.append(ANSWER_ID)
    return seq


def gen_tiny_program_sample(rng, depth=2, n_distract_facts=3, n_distract_rules=2,
                            confusable_prob=0.4):
    # Cadena: v0 --r1--> v1 --r2--> ... --rd--> vd. La query da v0 + ids de
    # regla EN ORDEN; la respuesta es el FACT del var final. Anti-atajos:
    # - vars/vals/ids re-sampleados por muestra (bindings frescos);
    # - distractores: facts de vars fuera de la cadena y rules con ids no
    #   consultados (cambiarlos no altera la respuesta);
    # - confusables (prob confusable_prob): rule distractor con MISMO lhs
    #   que una regla de la cadena y rhs distinto -> fuerza lookup por id.
    chain_vars = rng.sample(range(N_VARS_TINY), depth + 1)
    rule_ids = rng.sample(range(N_RULES_TINY), depth)
    chain_vals = [rng.randrange(N_VALS_TINY) for _ in range(depth + 1)]
    facts = list(zip(chain_vars, chain_vals))
    rules = [(rule_ids[i], chain_vars[i], chain_vars[i + 1]) for i in range(depth)]
    free_vars = [v for v in range(N_VARS_TINY) if v not in chain_vars]
    rng.shuffle(free_vars)
    for v in free_vars[:n_distract_facts]:
        facts.append((v, rng.randrange(N_VALS_TINY)))
    free_rids = [r for r in range(N_RULES_TINY) if r not in rule_ids]
    rng.shuffle(free_rids)
    for rid in free_rids[:n_distract_rules]:
        if rng.random() < confusable_prob:
            src = rng.choice(chain_vars[:-1])
            nxt = chain_vars[chain_vars.index(src) + 1]
            rhs = rng.choice([v for v in range(N_VARS_TINY) if v != nxt])
            rules.append((rid, src, rhs))
        else:
            lhs = rng.choice(free_vars) if free_vars else chain_vars[0]
            rhs = rng.choice([v for v in range(N_VARS_TINY) if v != lhs])
            rules.append((rid, lhs, rhs))
    return tiny_program_build_seq(rng, facts, rules, chain_vars[0], rule_ids)


def tiny_program_parse(x):
    # Parsea el prefijo a estructura (dict) o None si malformada. Fuente
    # unica de verdad para oracle y para las intervenciones del probe.
    toks = [int(t) for t in x]
    while toks and toks[-1] == PAD_ID:
        toks.pop()
    if not toks or toks[0] != BOS_ID:
        return None
    seps = [i for i, t in enumerate(toks) if t == SEP_ID]
    if len(seps) < 2:
        return None
    f_end, r_end = seps[0], seps[1]
    facts = {}
    i = 1
    while i < f_end:
        if i + 2 >= f_end or toks[i] != FACT_ID:
            return None
        var = toks[i + 1] - VAR_OFFSET_TINY
        val = toks[i + 2] - VAL_OFFSET_TINY
        if not (0 <= var < N_VARS_TINY and 0 <= val < N_VALS_TINY):
            return None
        facts[var] = val
        i += 3
    rules = {}
    i = f_end + 1
    while i < r_end:
        if i + 3 >= r_end or toks[i] != RULE_ID:
            return None
        rid = toks[i + 1] - RID_OFFSET_TINY
        lhs = toks[i + 2] - VAR_OFFSET_TINY
        rhs = toks[i + 3] - VAR_OFFSET_TINY
        if not (0 <= rid < N_RULES_TINY and 0 <= lhs < N_VARS_TINY
                and 0 <= rhs < N_VARS_TINY):
            return None
        rules[rid] = (lhs, rhs)
        i += 4
    q = toks[r_end + 1:]
    if not q or q[-1] != ANSWER_ID or q[0] != QUERY_ID:
        return None
    q = q[1:-1]
    if not q:
        return None
    q_var = q[0] - VAR_OFFSET_TINY
    if not (0 <= q_var < N_VARS_TINY):
        return None
    q_rules = []
    for t in q[1:]:
        rid = t - RID_OFFSET_TINY
        if not (0 <= rid < N_RULES_TINY):
            return None
        q_rules.append(rid)
    return {'facts': facts, 'rules': rules, 'query_var': q_var, 'query_rules': q_rules}


def tiny_program_oracle_target(x):
    # Resuelve la respuesta del prefijo como parser de referencia: encadena
    # reglas por id desde v0 y consulta el FACT del var final. None si la
    # secuencia es invalida. Usado por tests y por el probe (targets de
    # base e intervenciones).
    p = tiny_program_parse(x)
    if p is None:
        return None
    cur = p['query_var']
    for rid in p['query_rules']:
        r = p['rules'].get(rid)
        if r is None or r[0] != cur:
            return None
        cur = r[1]
    val = p['facts'].get(cur)
    if val is None:
        return None
    return VAL_OFFSET_TINY + val


def tiny_depth_bucket(x):
    # depth = cantidad de rule-ids en la query (d1/d2/d3; 'other' si
    # malformada). Separa EM_d2 (ID) de EM_d3 (OOD) dentro de la misma task.
    p = tiny_program_parse(x)
    if p is None:
        return 'other'
    d = len(p['query_rules'])
    return 'd%d' % d if d in (1, 2, 3) else 'other'


def gen_dyck2_sample(rng, n_opened=3, n_closed_before=2, n_trailing=1):
    """Prefijo Dyck-2 parcialmente balanceado con pila NO vacia al final.

    La respuesta (oracle) es el CLOSE del tipo del TOP de la pila
    (Suzgun et al. 2019, predict-closing-bracket).

    Estructura del prefijo (stack machine):
      1. n_closed_before pares balanceados de ruido (se cierran solos).
      2. n_opened opens (quedan SIN cerrar; su profundidad define el bucket).
      3. n_trailing pares balanceados interiores DESPUES de los opens:
         sus closes sacan solo opens interiores; el top del stack final
         sigue siendo opens[-1], un OPEN VIEJO.

    Anticortos:
    - 'copiar el ultimo token': con n_trailing>=1 el ultimo token del
      prefijo es un CLOSE interior (tipo distinto del top), nunca el par
      del top.
    - 'solo cuento la profundidad': con n_opened>=2 hay que recordar el
      tipo del mas interno abierto, no solo cuantos.
    Formato: BOS [t1...tk] ANSWER (prefix_answer=True, el CLOSE va solo
    en y; collate_fn de tiny_program ya maneja ese formato).
    """
    toks = []
    for _ in range(n_closed_before):
        t = rng.randrange(N_DYCK_TYPES)
        toks.append(DYCK_OPEN_BASE + t)
        toks.append(DYCK_CLOSE_BASE + t)
    opens = [rng.randrange(N_DYCK_TYPES) for _ in range(n_opened)]
    for t in opens:
        toks.append(DYCK_OPEN_BASE + t)
    for _ in range(n_trailing):
        t = rng.randrange(N_DYCK_TYPES)
        toks.append(DYCK_OPEN_BASE + t)
        toks.append(DYCK_CLOSE_BASE + t)
    seq = [BOS_ID] + toks + [ANSWER_ID]
    target = DYCK_CLOSE_BASE + opens[-1]
    return seq, target


def dyck2_oracle_target(seq):
    """CLOSE del top de la pila del prefijo, o None si malformada."""
    toks = [int(t) for t in seq]
    while toks and toks[-1] == PAD_ID:
        toks.pop()
    if not toks or toks[0] != BOS_ID or toks[-1] != ANSWER_ID:
        return None
    stack = []
    for t in toks[1:-1]:
        if DYCK_OPEN_BASE <= t < DYCK_OPEN_BASE + N_DYCK_TYPES:
            stack.append(t - DYCK_OPEN_BASE)
        elif DYCK_CLOSE_BASE <= t < DYCK_CLOSE_BASE + N_DYCK_TYPES:
            if not stack:
                return None
            stack.pop()
        else:
            return None
    if not stack:
        return None
    return DYCK_CLOSE_BASE + stack[-1]


def dyck2_depth_bucket(x):
    """Bucket por profundidad de la pila al final del prefijo (d1/d2/d3+)."""
    p = dyck2_oracle_target(x)
    if p is None:
        return 'other'
    toks = [int(t) for t in x]
    while toks and toks[-1] == PAD_ID:
        toks.pop()
    depth = 0
    for t in toks[1:-1]:
        if DYCK_OPEN_BASE <= t < DYCK_OPEN_BASE + N_DYCK_TYPES:
            depth += 1
        elif DYCK_CLOSE_BASE <= t < DYCK_CLOSE_BASE + N_DYCK_TYPES:
            depth -= 1
    if depth <= 2:
        return 'd%d' % depth
    return 'd3+'


class Dyck2Dataset(Dataset):
    # Tuplas (seq_prefijo, target): el CLOSE respuesta NO esta en x
    # (TaskSpec usa prefix_answer=True, como tiny_program).
    def __init__(self, n_samples, seed=0, n_opened_range=(2, 4),
                 n_closed_range=(1, 4), n_trailing_range=(1, 3)):
        self.samples = []
        rng = random.Random(seed)
        seen = set()
        attempts = 0
        while len(self.samples) < n_samples and attempts < n_samples * 8:
            attempts += 1
            n_opened = rng.randint(*n_opened_range)
            n_closed = rng.randint(*n_closed_range)
            n_trail = rng.randint(*n_trailing_range)
            seq, tgt = gen_dyck2_sample(rng, n_opened=n_opened,
                                        n_closed_before=n_closed,
                                        n_trailing=n_trail)
            if dyck2_oracle_target(seq) != tgt:
                continue
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            self.samples.append((torch.tensor(seq, dtype=torch.long),
                                 torch.tensor(tgt, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TinyProgramDataset(Dataset):
    # Entrega tuplas (seq_prefijo, target): el valor de respuesta NO esta
    # en x (el TaskSpec de la task usa prefix_answer=True; collate_fn lo
    # pone en y en la posicion del ANSWER_ID).
    def __init__(self, n_samples, seed=0, depth_range=(1, 2),
                 n_facts_range=(2, 4), n_rules_range=(2, 3),
                 confusable_prob=0.4):
        self.samples = []
        rng = random.Random(seed)
        seen = set()
        attempts = 0
        while len(self.samples) < n_samples and attempts < n_samples * 8:
            attempts += 1
            depth = rng.randint(*depth_range)
            seq = gen_tiny_program_sample(
                rng, depth=depth,
                n_distract_facts=rng.randint(*n_facts_range),
                n_distract_rules=rng.randint(*n_rules_range),
                confusable_prob=confusable_prob)
            tgt = tiny_program_oracle_target(seq)
            if tgt is None:
                continue
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            self.samples.append((torch.tensor(seq, dtype=torch.long),
                                 torch.tensor(tgt, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MQARDataset(Dataset):
    def __init__(self, n_samples: int, n_pairs_range: Tuple[int, int] = (2, 4),
                 hops: int = 1, seed: int = 0):
        self.samples = []
        rng = random.Random(seed)
        seen = set()
        attempts = 0
        while len(self.samples) < n_samples and attempts < n_samples * 4:
            attempts += 1
            n_pairs = rng.randint(*n_pairs_range)
            seq = gen_mqar_sample(rng, n_pairs=n_pairs, hops=hops)
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            self.samples.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def gen_copy_reverse_sample(rng: random.Random, min_len: int = 8, max_len: int = 12, shift_aug_max: int = 0) -> List[int]:
    n = rng.randint(min_len, max_len)
    digits = [DIGIT_OFFSET + rng.randrange(N_DIGITS_COPY) for _ in range(n)]
    rev = list(reversed(digits))
    seq = [BOS_ID] + digits + [SEP_ID, ANSWER_ID] + rev + [EOS_ID]
    if shift_aug_max > 0:
        offset = rng.randint(0, shift_aug_max)
        seq = [PAD_ID] * offset + seq
    return seq


class CopyReverseDataset(Dataset):
    def __init__(self, n_samples: int, seed: int = 0,
                 min_len: int = 8, max_len: int = 12,
                 shift_aug_max: int = 0):
        self.samples = []
        rng = random.Random(seed)
        seen = set()
        attempts = 0
        while len(self.samples) < n_samples and attempts < n_samples * 4:
            attempts += 1
            seq = gen_copy_reverse_sample(rng, min_len, max_len, shift_aug_max=shift_aug_max)
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            self.samples.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def gen_state_tracking_sample(rng: random.Random, n_events: int = 8) -> List[int]:
    ent_loc = {e: rng.randrange(N_LOCATIONS) for e in range(N_ENTITIES)}
    seq = [BOS_ID]
    for _ in range(n_events):
        et = rng.randint(0, N_ENTITIES - 1)
        op = rng.choice(['assign', 'goto'])
        loc = rng.randrange(N_LOCATIONS)
        ent_loc[et] = loc
        op_id = ASSIGN_ID if op == 'assign' else GOTO_ID
        seq += [ENT_OFFSET + et, op_id, LOC_OFFSET + loc, SEP_ID]
    qe = rng.randrange(N_ENTITIES)
    target_loc = ent_loc[qe]
    seq += [QUERY_ID, ENT_OFFSET + qe, ANSWER_ID, LOC_OFFSET + target_loc]
    return seq


class StateTrackingDataset(Dataset):
    def __init__(self, n_samples: int, seed: int = 0,
                 n_events_range: Tuple[int, int] = (6, 10)):
        self.samples = []
        rng = random.Random(seed)
        seen = set()
        attempts = 0
        while len(self.samples) < n_samples and attempts < n_samples * 4:
            attempts += 1
            n_events = rng.randint(*n_events_range)
            seq = gen_state_tracking_sample(rng, n_events=n_events)
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            self.samples.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_target_shifted(input_ids: torch.Tensor, pad_id: int = PAD_ID,
                        answer_marker_id: Optional[int] = None,
                        mark_after_marker: bool = False):
    B, T = input_ids.shape
    y = torch.full_like(input_ids, pad_id)
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    y[:, :-1] = input_ids[:, 1:]
    if answer_marker_id is None:
        target_mask[:, :-1] = input_ids[:, 1:] != pad_id
    elif not mark_after_marker:
        target_mask[:, :-1] = input_ids[:, :-1] == answer_marker_id
    else:
        marker_pos = (input_ids == answer_marker_id).int().argmax(dim=1)
        has_marker = (input_ids == answer_marker_id).any(dim=1)
        last_nonpad = (input_ids != pad_id).int().sum(dim=1) - 1
        for b in range(B):
            if not bool(has_marker[b]):
                continue
            s = int(marker_pos[b])
            e = int(last_nonpad[b].clamp(min=s))
            target_mask[b, s:e+1] = True
            if y[b, e] == pad_id:
                target_mask[b, e] = False
    return y, target_mask


def collate_fn(batch: List[torch.Tensor], pad_id: int = PAD_ID,
               answer_marker_id: Optional[int] = None, mark_after_marker: bool = False,
               prefix_answer: bool = False):
    # prefix_answer (S61): los items del batch son tuplas (seq, target);
    # seq termina en ANSWER (ultimo token real) y el target (el valor de
    # respuesta) vive SOLO en y, nunca en x. Aplica a train Y eval por
    # construccion: el modelo nunca ve tokens futuros dentro del chunk.
    if prefix_answer:
        assert answer_marker_id is not None, 'prefix_answer requires answer_marker_id'
        seqs = [b[0] for b in batch]
        targets = [int(b[1]) for b in batch]
        lengths = [len(s) for s in seqs]
        T = max(lengths)
        x = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = s
        y = torch.full_like(x, pad_id)
        target_mask = torch.zeros_like(x, dtype=torch.bool)
        for i in range(len(batch)):
            pos = (x[i] == answer_marker_id).nonzero(as_tuple=False)
            if len(pos) == 0:
                continue
            p = int(pos[0].item())
            y[i, p] = targets[i]
            target_mask[i, p] = True
        return x, y, target_mask
    lengths = [len(s) for s in batch]
    T = max(lengths)
    x = torch.full((len(batch), T), pad_id, dtype=torch.long)
    for i, s in enumerate(batch):
        x[i, :len(s)] = s
    y, target_mask = make_target_shifted(x, pad_id=pad_id,
                                          answer_marker_id=answer_marker_id,
                                          mark_after_marker=mark_after_marker)
    return x, y, target_mask


@dataclass
class TaskSpec:
    name: str
    vocab_size: int
    max_seq_len: int
    epochs: int
    train_samples: int
    valid_samples: int
    batch_size: int
    success_threshold: float
    success_metric: str
    make_train: Callable
    make_valid: Callable
    answer_marker_id: Optional[int] = None
    mark_after_marker: bool = False
    # S61 (tiny_program): el dataset entrega tuplas (seq, target) y el
    # target vive solo en y (nunca en x). bucket_fn etiqueta cada muestra
    # (ej. d2/d3) para reportar EM separado por bucket dentro de la task;
    # bucket_thresholds gatea recommend_kaggle: cada bucket debe pasar su
    # umbral (el gate real de extrapolacion es d3).
    prefix_answer: bool = False
    bucket_fn: Optional[Callable] = None
    bucket_thresholds: Optional[Dict[str, float]] = None


def make_task_specs(smoke: bool = True, include_forget: bool = True,
                    include_tiny_program: bool = False,
                    include_dyck2: bool = False) -> List[TaskSpec]:
    # include_forget=True desde 2026-08-03 (S45): la 5a task
    # `forget_retrieval` (store + FORGET k_i + QUERY k_j != i + ANSWER v_j)
    # entra al harness por defecto. Modo legacy 4-tasks:
    # `make_task_specs(smoke=..., include_forget=False)`. La regla
    # recommend_kaggle (passed >= 3 AND passes_baseline >= 2) reescala a
    # `passed >= n_total - 2` con n_total=5 -> >=4 (ver `_recommend_kaggle`).
    # include_tiny_program (S61): task 6ta `tiny_program` OP-TIN (default
    # False) para calibrar sin invalidar el baseline historico. El runner
    # la activa via `--tasks tiny_program` (run_smoke lo detecta) o con el
    # flag del runner. d2/d3 son buckets de la misma task (EM separado via
    # bucket_fn), con d3 como gate de extrapolacion (bucket_thresholds).
    # include_dyck2 (S81): task 7ma `dyck2` OP-TIN (default False), misma
    # mecanica que tiny_program. Mide memoria de stack / composicion
    # (Dyck-2 predict-closing-bracket); bucket d3+ gatea la profundidad.
    tiny = None
    dyck2 = None
    if smoke == 'cpu_quick':
        specs = [
            TaskSpec(
                name='mqar_1hop',
                vocab_size=VOCAB_MQAR_SIZE, max_seq_len=24, epochs=60,
                train_samples=400, valid_samples=120, batch_size=32,
                success_threshold=0.85, success_metric='exact_match',
                make_train=lambda seed: MQARDataset(400, seed=seed, hops=1, n_pairs_range=(5, 9)),
                make_valid=lambda seed: MQARDataset(120, seed=seed + 1, hops=1, n_pairs_range=(5, 9)),
                answer_marker_id=ANSWER_ID,
            ),
            TaskSpec(
                name='mqar_2hop',
                vocab_size=VOCAB_MQAR_SIZE, max_seq_len=24, epochs=60,
                train_samples=400, valid_samples=120, batch_size=32,
                success_threshold=0.80, success_metric='exact_match',
                make_train=lambda seed: MQARDataset(400, seed=seed, hops=2, n_pairs_range=(5, 9)),
                make_valid=lambda seed: MQARDataset(120, seed=seed + 1, hops=2, n_pairs_range=(5, 9)),
                answer_marker_id=ANSWER_ID,
            ),
            TaskSpec(
                name='copy_reverse',
                vocab_size=VOCAB_COPY_SIZE, max_seq_len=48, epochs=80,
                train_samples=800, valid_samples=200, batch_size=32,
                success_threshold=0.90, success_metric='exact_match',
                make_train=lambda seed: CopyReverseDataset(800, seed=seed, min_len=20, max_len=22),
                make_valid=lambda seed: CopyReverseDataset(200, seed=seed + 1, min_len=20, max_len=22),
                answer_marker_id=ANSWER_ID, mark_after_marker=True,
            ),
            TaskSpec(
                name='state_tracking',
                vocab_size=VOCAB_ST_SIZE, max_seq_len=80, epochs=80,
                train_samples=600, valid_samples=150, batch_size=32,
                success_threshold=0.70, success_metric='exact_match',
                make_train=lambda seed: StateTrackingDataset(600, seed=seed, n_events_range=(6, 10)),
                make_valid=lambda seed: StateTrackingDataset(150, seed=seed + 1, n_events_range=(6, 10)),
                answer_marker_id=ANSWER_ID,
            ),
            TaskSpec(
                name='forget_retrieval',
                vocab_size=VOCAB_FORGET_SIZE, max_seq_len=32, epochs=80,
                train_samples=600, valid_samples=150, batch_size=32,
                success_threshold=0.70, success_metric='exact_match',
                make_train=lambda seed: ForgetRetrieveDataset(600, seed=seed, n_pairs_range=(5, 9), n_forget_range=(1, 2)),
                make_valid=lambda seed: ForgetRetrieveDataset(150, seed=seed + 1, n_pairs_range=(5, 9), n_forget_range=(1, 2)),
                answer_marker_id=ANSWER_ID,
            ),
        ]
        tiny = TaskSpec(
            name='tiny_program',
            vocab_size=VOCAB_TINY_SIZE, max_seq_len=64, epochs=80,
            train_samples=600, valid_samples=200, batch_size=32,
            success_threshold=0.60, success_metric='exact_match',
            make_train=lambda seed: TinyProgramDataset(600, seed=seed, depth_range=(1, 2)),
            make_valid=lambda seed: TinyProgramDataset(200, seed=seed + 1, depth_range=(2, 3)),
            answer_marker_id=ANSWER_ID, prefix_answer=True,
            bucket_fn=tiny_depth_bucket,
            bucket_thresholds={'d3': 0.40},
        )
        dyck2 = TaskSpec(
            name='dyck2',
            vocab_size=VOCAB_DYCK_SIZE, max_seq_len=24, epochs=80,
            train_samples=600, valid_samples=200, batch_size=32,
            success_threshold=0.70, success_metric='exact_match',
            make_train=lambda seed: Dyck2Dataset(600, seed=seed),
            make_valid=lambda seed: Dyck2Dataset(200, seed=seed + 1),
            answer_marker_id=ANSWER_ID, prefix_answer=True,
            bucket_fn=dyck2_depth_bucket,
            bucket_thresholds={'d3+': 0.40},
        )
    elif smoke == 'smoke' or smoke is True:
        specs = [
            TaskSpec(
                name='mqar_1hop',
                vocab_size=VOCAB_MQAR_SIZE, max_seq_len=42, epochs=200,
                train_samples=800, valid_samples=200, batch_size=16,
                success_threshold=0.85, success_metric='exact_match',
                make_train=lambda seed: MQARDataset(800, seed=seed, hops=1, n_pairs_range=(12, 18)),
                make_valid=lambda seed: MQARDataset(200, seed=seed + 1, hops=1, n_pairs_range=(12, 18)),
                answer_marker_id=ANSWER_ID,
            ),
            TaskSpec(
                name='mqar_2hop',
                vocab_size=VOCAB_MQAR_SIZE, max_seq_len=42, epochs=250,
                train_samples=800, valid_samples=200, batch_size=16,
                success_threshold=0.80, success_metric='exact_match',
                make_train=lambda seed: MQARDataset(800, seed=seed, hops=2, n_pairs_range=(12, 18)),
                make_valid=lambda seed: MQARDataset(200, seed=seed + 1, hops=2, n_pairs_range=(12, 18)),
                answer_marker_id=ANSWER_ID,
            ),
            TaskSpec(
                name='copy_reverse',
                vocab_size=VOCAB_COPY_SIZE, max_seq_len=48, epochs=300,
                train_samples=1600, valid_samples=400, batch_size=16,
                success_threshold=0.90, success_metric='exact_match',
                make_train=lambda seed: CopyReverseDataset(1600, seed=seed, min_len=20, max_len=22),
                make_valid=lambda seed: CopyReverseDataset(400, seed=seed + 1, min_len=20, max_len=22),
                answer_marker_id=ANSWER_ID, mark_after_marker=True,
            ),
            TaskSpec(
                name='state_tracking',
                vocab_size=VOCAB_ST_SIZE, max_seq_len=80, epochs=400,
                train_samples=1200, valid_samples=300, batch_size=16,
                success_threshold=0.70, success_metric='exact_match',
                make_train=lambda seed: StateTrackingDataset(1200, seed=seed, n_events_range=(6, 10)),
                make_valid=lambda seed: StateTrackingDataset(300, seed=seed + 1, n_events_range=(6, 10)),
                answer_marker_id=ANSWER_ID,
            ),
            TaskSpec(
                name='forget_retrieval',
                vocab_size=VOCAB_FORGET_SIZE, max_seq_len=48, epochs=300,
                train_samples=1200, valid_samples=300, batch_size=16,
                success_threshold=0.70, success_metric='exact_match',
                make_train=lambda seed: ForgetRetrieveDataset(1200, seed=seed, n_pairs_range=(12, 18), n_forget_range=(1, 3)),
                make_valid=lambda seed: ForgetRetrieveDataset(300, seed=seed + 1, n_pairs_range=(12, 18), n_forget_range=(1, 3)),
                answer_marker_id=ANSWER_ID,
            ),
        ]
        tiny = TaskSpec(
            name='tiny_program',
            vocab_size=VOCAB_TINY_SIZE, max_seq_len=80, epochs=300,
            train_samples=1200, valid_samples=400, batch_size=16,
            success_threshold=0.60, success_metric='exact_match',
            make_train=lambda seed: TinyProgramDataset(1200, seed=seed, depth_range=(1, 2), n_facts_range=(4, 7), n_rules_range=(3, 6)),
            make_valid=lambda seed: TinyProgramDataset(400, seed=seed + 1, depth_range=(2, 3), n_facts_range=(4, 7), n_rules_range=(3, 6)),
            answer_marker_id=ANSWER_ID, prefix_answer=True,
            bucket_fn=tiny_depth_bucket,
            bucket_thresholds={'d3': 0.40},
        )
        dyck2 = TaskSpec(
            name='dyck2',
            vocab_size=VOCAB_DYCK_SIZE, max_seq_len=48, epochs=300,
            train_samples=1200, valid_samples=400, batch_size=16,
            success_threshold=0.70, success_metric='exact_match',
            make_train=lambda seed: Dyck2Dataset(1200, seed=seed, n_opened_range=(2, 5), n_closed_range=(2, 6), n_trailing_range=(2, 6)),
            make_valid=lambda seed: Dyck2Dataset(400, seed=seed + 1, n_opened_range=(2, 5), n_closed_range=(2, 6), n_trailing_range=(2, 6)),
            answer_marker_id=ANSWER_ID, prefix_answer=True,
            bucket_fn=dyck2_depth_bucket,
            bucket_thresholds={'d3+': 0.40},
        )
    else:
        specs = [
            TaskSpec(
                name='mqar_1hop', vocab_size=VOCAB_MQAR_SIZE, max_seq_len=54, epochs=400,
                train_samples=1000, valid_samples=200, batch_size=32,
                success_threshold=0.85, success_metric='exact_match',
                make_train=lambda seed: MQARDataset(1000, seed=seed, hops=1, n_pairs_range=(16, 24)),
                make_valid=lambda seed: MQARDataset(200, seed=seed + 1, hops=1, n_pairs_range=(16, 24)),
                answer_marker_id=ANSWER_ID,
            ),
            TaskSpec(
                name='mqar_2hop', vocab_size=VOCAB_MQAR_SIZE, max_seq_len=54, epochs=500,
                train_samples=1000, valid_samples=200, batch_size=32,
                success_threshold=0.80, success_metric='exact_match',
                make_train=lambda seed: MQARDataset(1000, seed=seed, hops=2, n_pairs_range=(16, 24)),
                make_valid=lambda seed: MQARDataset(200, seed=seed + 1, hops=2, n_pairs_range=(16, 24)),
                answer_marker_id=ANSWER_ID,
            ),
            TaskSpec(
                name='copy_reverse', vocab_size=VOCAB_COPY_SIZE, max_seq_len=48, epochs=500,
                train_samples=2000, valid_samples=400, batch_size=32,
                success_threshold=0.90, success_metric='exact_match',
                make_train=lambda seed: CopyReverseDataset(2000, seed=seed, min_len=20, max_len=22),
                make_valid=lambda seed: CopyReverseDataset(400, seed=seed + 1, min_len=20, max_len=22),
                answer_marker_id=ANSWER_ID, mark_after_marker=True,
            ),
            TaskSpec(
                name='state_tracking', vocab_size=VOCAB_ST_SIZE, max_seq_len=80, epochs=600,
                train_samples=1500, valid_samples=300, batch_size=32,
                success_threshold=0.70, success_metric='exact_match',
                make_train=lambda seed: StateTrackingDataset(1500, seed=seed, n_events_range=(6, 10)),
                make_valid=lambda seed: StateTrackingDataset(300, seed=seed + 1, n_events_range=(6, 10)),
                answer_marker_id=ANSWER_ID,
            ),
            TaskSpec(
                name='forget_retrieval', vocab_size=VOCAB_FORGET_SIZE, max_seq_len=54, epochs=600,
                train_samples=1500, valid_samples=300, batch_size=32,
                success_threshold=0.70, success_metric='exact_match',
                make_train=lambda seed: ForgetRetrieveDataset(1500, seed=seed, n_pairs_range=(16, 24), n_forget_range=(1, 4)),
                make_valid=lambda seed: ForgetRetrieveDataset(300, seed=seed + 1, n_pairs_range=(16, 24), n_forget_range=(1, 4)),
                answer_marker_id=ANSWER_ID,
            ),
        ]
        tiny = TaskSpec(
            name='tiny_program',
            vocab_size=VOCAB_TINY_SIZE, max_seq_len=96, epochs=600,
            train_samples=1500, valid_samples=500, batch_size=32,
            success_threshold=0.60, success_metric='exact_match',
            make_train=lambda seed: TinyProgramDataset(1500, seed=seed, depth_range=(1, 2), n_facts_range=(6, 10), n_rules_range=(4, 8)),
            make_valid=lambda seed: TinyProgramDataset(500, seed=seed + 1, depth_range=(2, 3), n_facts_range=(6, 10), n_rules_range=(4, 8)),
            answer_marker_id=ANSWER_ID, prefix_answer=True,
            bucket_fn=tiny_depth_bucket,
            bucket_thresholds={'d3': 0.40},
        )
        dyck2 = TaskSpec(
            name='dyck2',
            vocab_size=VOCAB_DYCK_SIZE, max_seq_len=64, epochs=500,
            train_samples=1500, valid_samples=500, batch_size=32,
            success_threshold=0.70, success_metric='exact_match',
            make_train=lambda seed: Dyck2Dataset(1500, seed=seed, n_opened_range=(2, 6), n_closed_range=(3, 8), n_trailing_range=(3, 8)),
            make_valid=lambda seed: Dyck2Dataset(500, seed=seed + 1, n_opened_range=(2, 6), n_closed_range=(3, 8), n_trailing_range=(3, 8)),
            answer_marker_id=ANSWER_ID, prefix_answer=True,
            bucket_fn=dyck2_depth_bucket,
            bucket_thresholds={'d3+': 0.40},
        )
    if include_tiny_program and tiny is not None:
        specs.append(tiny)
    if include_dyck2 and dyck2 is not None:
        specs.append(dyck2)
    if not include_forget:
        specs = [s for s in specs if s.name != 'forget_retrieval']
    return specs


def train_model(model, train_loader, valid_loader, device, epochs: int,
                lr: float, weight_decay: float, clip: float, vocab_size: int,
                optimizer: str = 'adamw', loss_fn: Optional[Callable] = None,
                bucket_fn: Optional[Callable] = None,
                ss_fn: Optional[Callable] = None):
    # loss_fn (opcional, default None): fn(model, logits, y_train, vocab_size)
    # -> scalar. Se usa para anadir terminos auxiliares (ej. L_homeo del
    # s4d_valence). None = CE puro (comportamiento historico identico).
    # bucket_fn (S61): etiqueta cada muestra (ej. tiny d2/d3) -> trackea
    # EM por bucket + max-in-window por bucket (precedente S30.11).
    from tests.muon_opt import make_optimizer
    model.to(device)
    opt = make_optimizer(model, optimizer, lr=lr, weight_decay=weight_decay)
    hist = []
    max_em_in_window = 0.0  # S30.11: blips transitorios -> reporta max
    max_token_acc_window = 0.0
    bucket_max = {}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_t0 = time.time()
        train_loss, train_tokens = 0.0, 0
        for x, y, target_mask in train_loader:
            x = x.to(device)
            y_train = torch.where(target_mask, y, torch.full_like(y, PAD_ID))
            y_train = y_train.to(device)
            opt.zero_grad(set_to_none=True)
            if ss_fn is not None:
                x = ss_fn(x, target_mask, model, ep)
            logits = model(x)
            if loss_fn is not None:
                loss = loss_fn(model, logits, y_train, vocab_size)
            else:
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_train.reshape(-1),
                                       ignore_index=PAD_ID)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            n_tok = (y_train != PAD_ID).sum().item()
            train_loss += loss.item() * n_tok
            train_tokens += n_tok
        val = evaluate(model, valid_loader, device, vocab_size, bucket_fn=bucket_fn)
        max_em_in_window = max(max_em_in_window, val['exact_match'])
        max_token_acc_window = max(max_token_acc_window, val['token_acc'])
        for k, v in val.get('exact_match_by_bucket', {}).items():
            bucket_max[k] = max(bucket_max.get(k, 0.0), v)
        hist.append({
            'epoch': ep,
            'train_loss': train_loss / max(train_tokens, 1),
            'valid_loss': val['loss'],
            'valid_token_acc': val['token_acc'],
            'valid_exact_match': val['exact_match'],
            'valid_max_em_in_window': max_em_in_window,
            'valid_max_token_acc_window': max_token_acc_window,
            'valid_exact_match_by_bucket': val.get('exact_match_by_bucket', {}),
            'epoch_seconds': time.time() - ep_t0,
        })
    final = hist[-1] if hist else {'valid_loss': float('nan'),
                                    'valid_token_acc': 0.0,
                                    'valid_exact_match': 0.0}
    final_meta = {
        'total_seconds': time.time() - t0,
        'final': final,
        'max_em_in_window': max_em_in_window,
        'max_token_acc_window': max_token_acc_window,
        'max_em_by_bucket_in_window': bucket_max,
    }
    return hist, final_meta


def evaluate(model, loader, device, vocab_size: int,
             bucket_fn: Optional[Callable] = None) -> Dict[str, float]:
    model.eval()
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    exact_match_ok, exact_match_n = 0, 0
    bucket_ok, bucket_n = {}, {}
    with torch.no_grad():
        for x, y, target_mask in loader:
            x = x.to(device)
            y_eval = torch.where(target_mask, y, torch.full_like(y, PAD_ID))
            y_eval = y_eval.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_eval.reshape(-1),
                                   ignore_index=PAD_ID, reduction='sum')
            n_tok = (y_eval != PAD_ID).sum().item()
            total_loss += loss.item()
            total_tokens += n_tok
            pred = logits.argmax(dim=-1)
            correct = (pred == y_eval) & (y_eval != PAD_ID)
            total_correct += correct.sum().item()
            for i in range(x.size(0)):
                m = target_mask[i]
                if m.sum().item() == 0:
                    continue
                exact_match_n += 1
                pos = torch.nonzero(m, as_tuple=False).squeeze(-1)
                matched = torch.equal(pred[i][pos], y_eval[i][pos])
                if matched:
                    exact_match_ok += 1
                if bucket_fn is not None:
                    lab = bucket_fn(x[i])
                    bucket_n[lab] = bucket_n.get(lab, 0) + 1
                    if matched:
                        bucket_ok[lab] = bucket_ok.get(lab, 0) + 1
    res = {
        'loss': total_loss / max(total_tokens, 1),
        'token_acc': total_correct / max(total_tokens, 1),
        'exact_match': exact_match_ok / max(exact_match_n, 1),
    }
    if bucket_fn is not None:
        res['exact_match_by_bucket'] = {
            k: bucket_ok.get(k, 0) / max(bucket_n.get(k, 0), 1)
            for k in bucket_n
        }
    return res


def run_one_task(task: TaskSpec, build_fn: Callable, device, seed: int,
                 lr: float, weight_decay: float, clip: float,
                 d_model: int, n_layers: int, variant: str = 'unknown',
                 compile_model: bool = False,
                 optimizer: str = 'adamw',
                 loss_fn: Optional[Callable] = None,
                 ss_fn: Optional[Callable] = None) -> Dict:
    from functools import partial
    if device.type == 'cuda':
        # Tecnicas GPU de bajo riesgo (DELIBERATION 30.15): TF32 en
        # matmuls+convs (T4) y cudnn.benchmark. El gate CPU local queda
        # en fp32 exacto (no afecta). bf16/autocast queda fuera (cambio
        # numerico grande -> re-validacion completa).
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    set_seed(seed)
    train_ds = task.make_train(seed=seed)
    valid_ds = task.make_valid(seed=seed + 100)
    col = partial(collate_fn, answer_marker_id=task.answer_marker_id,
                  mark_after_marker=task.mark_after_marker,
                  prefix_answer=task.prefix_answer)
    train_loader = DataLoader(train_ds, batch_size=task.batch_size, shuffle=True,
                              collate_fn=col, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=task.batch_size, shuffle=False,
                             collate_fn=col, num_workers=0)
    model = build_fn(vocab_size=task.vocab_size, max_len=task.max_seq_len,
                     d_model=d_model, n_layers=n_layers)
    if compile_model:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    hist, meta = train_model(model, train_loader, valid_loader, device,
                             epochs=task.epochs, lr=lr, weight_decay=weight_decay,
                             clip=clip, vocab_size=task.vocab_size,
                             optimizer=optimizer, loss_fn=loss_fn,
                             bucket_fn=task.bucket_fn, ss_fn=ss_fn)
    final = meta['final']
    metric = final['valid_' + task.success_metric]
    success = metric >= task.success_threshold
    ret = {
        'task': task.name,
        'n_params': n_params,
        'epochs': task.epochs,
        'train_samples': task.train_samples,
        'valid_samples': task.valid_samples,
        'final_loss': final['valid_loss'],
        'final_token_acc': final['valid_token_acc'],
        'final_exact_match': final['valid_exact_match'],
        'success_metric_name': task.success_metric,
        'success_metric_value': metric,
        'success_threshold': task.success_threshold,
        'success_pass': success,
        'max_em_in_window': meta.get('max_em_in_window', 0.0),
        'max_token_acc_window': meta.get('max_token_acc_window', 0.0),
        'exact_match_by_bucket': final.get('valid_exact_match_by_bucket', {}),
        'max_em_by_bucket_in_window': meta.get('max_em_by_bucket_in_window', {}),
        'seconds': time.time() - t0,
        'hist': hist,
    }
    # Save checkpoint for downstream probes (OOD transfer, etc)
    os.makedirs(f'outputs/{variant}/cache', exist_ok=True)
    ckpt_path = f'outputs/{variant}/cache/{task.name}_seed{seed}_dm{d_model}_L{n_layers}_ep{task.epochs}.pt'
    torch.save({'model': model.state_dict(), 'meta': meta}, ckpt_path)
    return ret


def _check_vs_baseline(summary: Dict, baseline_path: str,
                       epsilon: float = 0.0) -> Dict:
    """Compara el resumen del run actual contra el snapshot del transformer.

    Lee ``outputs/transformer/benchmark.json`` si existe; si no, devuelve
    ``baseline_present=False`` y el caller cae a la regla simple
    (passed >= 3/4).

    Para cada task, ``passes_baseline[task]`` es True si
    ``own_exact_match >= transformer_exact_match - epsilon`` (epsilon=0.0
    por default: el proto debe igualar o superar al transformer).

    ``recommend_kaggle`` final:
        passed >= 3/4 AND (baseline ausente OR passes_baseline en >=2 tasks)
    """
    import json as _json
    import os as _os
    if not _os.path.exists(baseline_path):
        return {
            'baseline_present': False,
            'baseline_path': baseline_path,
            'epsilon': epsilon,
            'passes_baseline': None,
            'n_passes_baseline': 0,
            'note': 'snapshot baseline no encontrado; cae a regla passed>=3/4',
        }
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = _json.load(f)
    passes = {}
    for tname, tr in summary['tasks'].items():
        own = tr['success_metric_value']
        base = baseline['tasks'].get(tname, {}).get('exact_match')
        if base is None:
            passes[tname] = None
        else:
            passes[tname] = bool(own >= base - epsilon)
    n_passes = sum(1 for v in passes.values() if v is True)
    n_total = sum(1 for v in passes.values() if v is not None)
    return {
        'baseline_present': True,
        'baseline_path': baseline_path,
        'epsilon': epsilon,
        'passes_baseline': passes,
        'n_passes_baseline': n_passes,
        'n_comparable_tasks': n_total,
        'baseline_variant': baseline.get('variant'),
        'baseline_smoke': baseline.get('smoke'),
        'baseline_seed': baseline.get('seed'),
        'baseline_d_model': baseline.get('d_model'),
        'baseline_n_layers': baseline.get('n_layers'),
    }


def run_smoke(variant: str, build_fn: Callable, device, seed: int = 42,
              lr: float = 1e-3, weight_decay: float = 0.01, clip: float = 1.0,
              d_model: int = 128, n_layers: int = 2, smoke=True,
              tasks: Optional[str] = None, epochs_override: int = -1,
              compile_model: bool = False,
              optimizer: str = 'adamw',
              include_forget: bool = True,
              include_tiny_program: bool = False,
              include_dyck2: bool = False,
              evidence_level: str = 'N1',
              loss_fn: Optional[Callable] = None,
              ss_fn: Optional[Callable] = None) -> Dict:
    # include_forget (S45): anade la 5a task `forget_retrieval` al harness.
    # Default True desde 2026-08-03; pasar False para modo legacy 4-tasks
    # (regla recommend_kaggle passed >= 3 exacto). Con 5 tasks, threshold
    # de passed sube a n_total - 2 (== 4/5) para preservar ratio 3/4.
    # include_tiny_program (S61): la 6ta task `tiny_program` es OP-TIN
    # (default False) para calibrar sin invalidar el baseline historico.
    # Ademas se auto-activa si `tasks` la nombra explicitamente
    # (--tasks tiny_program): sin runners modificados, sin inclusion
    # accidental en otros runs. d3 gatea recommend_kaggle via
    # bucket_thresholds del TaskSpec (ver mas abajo).
    # include_dyck2 (S81): la 7ma task `dyck2` es OP-TIN (default False),
    # misma mecanica que tiny_program (auto-activa si --tasks la nombra);
    # el bucket d3+ gatea la profundidad de stack (memoria de composicion).
    from dataclasses import replace
    tiny_requested = bool(tasks) and 'tiny_program' in {t.strip() for t in tasks.split(',')}
    dyck2_requested = bool(tasks) and 'dyck2' in {t.strip() for t in tasks.split(',')}
    specs = make_task_specs(smoke=smoke, include_forget=include_forget,
                            include_tiny_program=include_tiny_program or tiny_requested,
                            include_dyck2=include_dyck2 or dyck2_requested)
    if epochs_override > 0:
        specs = [replace(s, epochs=epochs_override) for s in specs]
    if tasks and tasks != 'all':
        wanted = set(t.strip() for t in tasks.split(','))
        specs = [s for s in specs if s.name in wanted]
    print(f"Reasoning smoke  |  variant={variant}  device={device}  smoke={smoke}  seed={seed}")
    print(f"Tasks: {[s.name for s in specs]}  d_model={d_model}  n_layers={n_layers}  opt={optimizer}")
    print("=" * 70)
    results = []
    for spec in specs:
        print(f"  - {spec.name:<16} epochs={spec.epochs} samples={spec.train_samples}/{spec.valid_samples}")
        res = run_one_task(spec, build_fn, device, seed=seed, lr=lr,
                           weight_decay=weight_decay, clip=clip,
                           d_model=d_model, n_layers=n_layers, variant=variant,
                           compile_model=compile_model, optimizer=optimizer,
                           loss_fn=loss_fn, ss_fn=ss_fn)
        results.append(res)
        print(f"    -> {res['success_metric_name']}={res['success_metric_value']:.3f} "
              f"target={spec.success_threshold:.2f} pass={res['success_pass']} "
              f"params={res['n_params']} t={res['seconds']:.1f}s")
    passed = sum(1 for r in results if r['success_pass'])
    # --- Comparacion contra baseline del transformer (snapshot) ---
    # Si el snapshot existe (lo genera transformer_smoke.py --save-baseline),
    # recommend_kaggle = passed>=3/4 AND passes_baseline en >=2 tasks.
    # Si no existe, cae a la regla simple passed>=3/4.
    # S45: con la 5a task activa, threshold de passed sube a n_total-1
    # (preserva "falla solo 1" como bar). Es decir: 4 tasks siguen
    # requiriendo 3 pasadas; 5 tasks requieren 4 pasadas (incluida o no
    # la critica forget_retrieval). Si include_forget=False o --tasks
    # omiten forget_retrieval, n_total=4 y se sigue el rule anterior
    # exacto (n_total-1 == 3).
    import os as _os
    repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    baseline_path = _os.path.join(repo_root, 'outputs', 'transformer',
                                  'benchmark.json')
    vs_base = _check_vs_baseline(
        {'tasks': {r['task']: {'success_metric_value': r['success_metric_value']}
                   for r in results}},
        baseline_path, epsilon=0.0)
    n_total = len(results)
    pass_threshold = max(3, n_total - 1)  # 4 -> 3, 5 -> 4, 6 -> 5
    # S61: gate por bucket (d3 de tiny_program). recommend_kaggle exige que
    # CADA bucket_threshold del spec se cumpla, ademas de la regla clasica:
    # fallar tiny_program (o su bucket d3) bloquea la promocion aunque otra
    # task secundaria falle.
    spec_by_name = {s.name: s for s in specs}
    thresholds_by_name = {s.name: s.bucket_thresholds for s in specs}
    bucket_gates = {}
    bucket_gates_ok = True
    for sname, spec in spec_by_name.items():
        if not spec.bucket_thresholds:
            continue
        res = next((r for r in results if r['task'] == sname), None)
        for k, thr in spec.bucket_thresholds.items():
            val = (res or {}).get('exact_match_by_bucket', {}).get(k, 0.0)
            ok = val >= thr
            bucket_gates['%s[%s]' % (sname, k)] = {
                'value': val, 'threshold': thr, 'pass': ok}
            bucket_gates_ok = bucket_gates_ok and ok
    if vs_base['baseline_present']:
        recommend = (passed >= pass_threshold
                     and vs_base['n_passes_baseline'] >= 2 and bucket_gates_ok)
    else:
        recommend = passed >= pass_threshold and bucket_gates_ok
    # Pin de entorno + git hash (S35.5: PyTorch upgrade silencioso
    # previene sorpresas; el JSON debe poder replicarse o diff-erse).
    import platform as _plat
    import subprocess as _sp
    try:
        env = {
            'python': _plat.python_version(),
            'torch': torch.__version__,
            'cuda_available': bool(torch.cuda.is_available()),
            'platform': _plat.platform(),
        }
    except Exception as _e:
        env = {'error': str(_e)}
    git_hash = None
    try:
        repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        out = _sp.run(['git', 'rev-parse', 'HEAD'], cwd=repo,
                      capture_output=True, text=True, timeout=5,
                      shell=True)
        if out.returncode == 0:
            git_hash = out.stdout.strip() or None
    except Exception:
        git_hash = None
    # Propaga max_em_in_window al summary por task
    return {
        'variant': variant, 'device': str(device), 'smoke': smoke, 'seed': seed,
        'lr': lr, 'weight_decay': weight_decay, 'clip': clip,
        'optimizer': optimizer,
        'd_model': d_model, 'n_layers': n_layers,
        'n_params': results[0]['n_params'] if results else 0,
        'env': env,
        'git_hash': git_hash,
        'evidence_level': evidence_level,
        'tasks': {r['task']: {
            'epochs': r['epochs'], 'train_samples': r['train_samples'],
            'valid_samples': r['valid_samples'],
            'final_loss': r['final_loss'], 'final_token_acc': r['final_token_acc'],
            'final_exact_match': r['final_exact_match'],
            'max_em_in_window': r.get('max_em_in_window', 0.0),
            'max_token_acc_window': r.get('max_token_acc_window', 0.0),
            'exact_match_by_bucket': r.get('exact_match_by_bucket', {}),
            'max_em_by_bucket_in_window': r.get('max_em_by_bucket_in_window', {}),
            'bucket_thresholds': thresholds_by_name.get(r['task']),
            'success_metric_name': r['success_metric_name'],
            'success_metric_value': r['success_metric_value'],
            'success_threshold': r['success_threshold'],
            'success_pass': r['success_pass'], 'seconds': r['seconds'],
            'hist': r['hist'],
        } for r in results},
        'passed_count': passed,
        'total_tasks': len(results),
        'recommend_kaggle': recommend,
        'bucket_gates': bucket_gates,
        'vs_baseline': vs_base,
    }


def print_summary(summary: Dict):
    print("\n" + "=" * 70)
    print(f"{'task':<18}{'metric':<12}{'value':>8}{'target':>8}{'pass':>6}{'time':>8}")
    print("-" * 70)
    for tname, tr in summary['tasks'].items():
        print(f"{tname:<18}{tr['success_metric_name']:<12}"
              f"{tr['success_metric_value']:8.3f}{tr['success_threshold']:8.2f}"
              f"{'  OK' if tr['success_pass'] else 'FAIL':>6}{tr['seconds']:7.1f}s")
        b = tr.get('exact_match_by_bucket') or {}
        if b:
            print("  buckets: " + " ".join(
                f"{k}={v:.3f}" for k, v in sorted(b.items())))
    bg = summary.get('bucket_gates') or {}
    if bg:
        print("-" * 70)
        for k, g in bg.items():
            print(f"  gate {k:<16} {g['value']:8.3f} >= {g['threshold']:.2f}"
                  f"{'  OK' if g['pass'] else 'FAIL'}")
    vsb = summary.get('vs_baseline') or {}
    if vsb.get('baseline_present'):
        print("\n" + "-" * 70)
        print(f"vs transformer baseline (epsilon={vsb['epsilon']:.2f}):")
        for tname, ok in vsb['passes_baseline'].items():
            mark = '==' if ok is True else ('--' if ok is None else '<<')
            print(f"  {tname:<18} {mark}")
        print(f"  -> {vsb['n_passes_baseline']}/{vsb['n_comparable_tasks']} >= baseline")
    else:
        print("\n" + "-" * 70)
        print(f"(sin baseline del transformer: {vsb.get('note', '')})")
    print(f"\n>> {summary['variant']}: {summary['passed_count']}/{summary['total_tasks']} "
          f"pasadas -> recommend_kaggle={summary['recommend_kaggle']}")
