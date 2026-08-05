"""
Task generators — from the easy case to the ones that actually differentiate.

Each task returns a list of "items":

    {
      "supports": [(depth, sentence), ...],  # sentences to plant in the haystack
      "question": str,
      "gold": str,                           # the value a correct answer contains
      "depth": float,                        # nominal depth (for the report)
      "entity": str (optional),              # enables distractor generation
      "topic":  str (optional),
    }

Tasks
-----
- single   : one lexically distinctive needle per depth. Easy; ~100% for lexical.
- multihop : RULER-style variable tracking. A chain (a0=value, a1=a0, ... aL=aL-1)
             is scattered through the haystack; the question names only aL. A
             single retrieval by aL surfaces the last *link*, not the value, so
             naive retrieval fails -- this exposes the multi-hop reasoning gap.
- nolima   : NoLiMa-style minimal lexical overlap. Needle and question share no
             content words (only a semantic link), so BM25 has nothing to match.
             Real dense embeddings should recover it; a lexical retriever cannot.
             This is the honesty check on whether we actually need dense/hybrid.
"""

from __future__ import annotations

CODEWORDS = (
    "zephyr quartz mirth vellum cinder gossamer thistle onyx marrow fathom "
    "brindle lattice pyre halcyon sable tundra cobalt ember plinth nimbus"
).split()

TOPICS = (
    "vault archive relay beacon ledger conduit cipher lattice manifold registry"
).split()

# needle phrasing  vs  question phrasing -- deliberately disjoint content words
NOLIMA_PAIRS = [
    ("the harbor lighthouse", "the coastal beacon tower"),
    ("the mountain cable car", "the alpine aerial gondola"),
    ("the desert solar array", "the arid photovoltaic farm"),
    ("the riverside grain mill", "the waterside flour works"),
    ("the underground metro depot", "the subterranean subway yard"),
    ("the orchard irrigation pump", "the fruit-grove watering engine"),
    ("the coastal shipping crane", "the seaside cargo hoist"),
    ("the northern wind turbine", "the polar breeze rotor"),
    ("the copper smelting furnace", "the bronze ore kiln"),
    ("the ancient stone aqueduct", "the old masonry water channel"),
    ("the glacier research cabin", "the icefield study lodge"),
    ("the savanna ranger outpost", "the grassland warden station"),
]


def _value(rng):
    return str(rng.randint(1000000, 9999999))


def task_single(depths, rng, **_):
    items, used = [], set()
    for d in depths:
        while True:
            ent = f"{rng.choice(CODEWORDS)}-{rng.randint(1000, 9999)}"
            if ent not in used:
                used.add(ent)
                break
        topic, val = rng.choice(TOPICS), _value(rng)
        s = (f"Important: the secret {topic} access code for entity {ent} "
             f"is {val}. Remember this exactly.")
        q = f"What is the secret {topic} access code for entity {ent}?"
        items.append({"supports": [(d, s)], "question": q, "gold": val,
                      "depth": d, "entity": ent, "topic": topic})
    return items


def task_multihop(depths, rng, hops=3, **_):
    items = []
    for i, d in enumerate(depths):
        val = _value(rng)
        names = [f"h{i}v{j}" for j in range(hops + 1)]
        spread = [min(0.99, max(0.0, d + (j - hops / 2) * 0.03))
                  for j in range(hops + 1)]
        supports = [(spread[0], f"Let variable {names[0]} be {val}.")]
        for j in range(1, hops + 1):
            supports.append(
                (spread[j], f"Let variable {names[j]} equal variable {names[j-1]}.")
            )
        q = f"What is the value of variable {names[hops]}?"
        items.append({"supports": supports, "question": q, "gold": val, "depth": d})
    return items


def task_nolima(depths, rng, **_):
    items, pairs = [], NOLIMA_PAIRS[:]
    rng.shuffle(pairs)
    for i, d in enumerate(depths):
        needle_phrase, query_phrase = pairs[i % len(pairs)]
        val = _value(rng)
        s = f"For the record, {needle_phrase} registration number is {val}."
        q = f"What is the identification code of {query_phrase}?"
        items.append({"supports": [(d, s)], "question": q, "gold": val, "depth": d})
    return items


TASKS = {"single": task_single, "multihop": task_multihop, "nolima": task_nolima}


def build_task(name, depths, rng, hops=3):
    if name not in TASKS:
        raise ValueError(f"unknown task '{name}'. choices: {list(TASKS)}")
    return TASKS[name](depths, rng, hops=hops)


def make_distractors(item, count, rng):
    """Confuser sentences sharing an item's topic + codeword prefix but a
    different entity number and value. Only for tasks with entity/topic.

    Retries on the rare draw that collides with the item's own entity, so it
    returns EXACTLY `count` distractors -- the old `continue` silently returned
    fewer than the harness then reported as distractors_per_needle."""
    if "entity" not in item:
        return []
    prefix = item["entity"].split("-")[0]
    out = []
    while len(out) < count:
        ent = f"{prefix}-{rng.randint(1000, 9999)}"
        if ent == item["entity"]:
            continue
        out.append(f"Note: the secret {item['topic']} access code for entity "
                   f"{ent} is {_value(rng)}.")
    return out
