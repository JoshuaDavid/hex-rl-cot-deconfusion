"""Induce a canonical grammar for the within-budget thinking-continuation set.

Pipeline (fully automated, exact for a finite string set):
  leaves -> word sequences -> trie -> minimal acyclic DFA (DAWG, suffix-merged)
  -> cut-point BNF readout (name only the states that branch or are shared;
     collapse forced linear chains into multi-word terminals).

The DAWG is the canonical (unique minimal) automaton for the finite language,
so the grammar is not "vibed": it is the minimal deterministic grammar whose
language is exactly the leaf set. We also verify by re-enumerating all accepted
strings and asserting set-equality.

Also compares the induced grammar across training tasks (Jaccard of leaf sets +
structural sizes) to test whether the grammar is task-specific.
"""
import argparse, itertools
from collections import defaultdict
import pandas as pd
from think_tree import load_model, build_context, enumerate_budget_tree


class Node:
    __slots__ = ("edges", "final")
    def __init__(self):
        self.edges = {}      # word -> Node
        self.final = False


def build_trie(word_seqs):
    root = Node()
    for words in word_seqs:
        n = root
        for w in words:
            n = n.edges.setdefault(w, Node())
        n.final = True
    return root


def minimize(root):
    """Suffix-merge the trie into the minimal acyclic DFA (Revuz/DAWG)."""
    registry = {}                # signature -> canonical Node
    done = {}                    # Node -> canonical Node

    def visit(n):
        if n in done:
            return done[n]
        new_edges = {w: visit(c) for w, c in sorted(n.edges.items())}
        sig = (n.final, tuple((w, id(c)) for w, c in new_edges.items()))
        if sig in registry:
            rep = registry[sig]
        else:
            n.edges = new_edges
            registry[sig] = n
            rep = n
        done[n] = rep
        return rep

    return visit(root)


def reachable(root):
    seen, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(n.edges.values())
    return seen


def degrees(nodes):
    indeg = {n: 0 for n in nodes}
    for n in nodes:
        for c in n.edges.values():
            indeg[c] += 1
    outdeg = {n: len(n.edges) for n in nodes}
    return indeg, outdeg


def induce_grammar(root):
    nodes = reachable(root)
    indeg, outdeg = degrees(nodes)

    def is_sink(n):
        return n.final and outdeg[n] == 0

    cut = set()
    for n in nodes:
        if is_sink(n):
            continue
        if n is root or indeg[n] > 1 or outdeg[n] > 1 or (n.final and outdeg[n] > 0):
            cut.add(n)
    cut.add(root)

    # name cut points; root = S
    names = {root: "S"}
    ctr = itertools.count(1)
    for n in sorted(cut, key=lambda x: (x is not root)):
        if n not in names:
            names[n] = f"N{next(ctr)}"

    def walk_phrase(w, ch):
        words, cur = [w], ch
        while cur not in cut and not is_sink(cur):
            (w2, nx), = cur.edges.items()
            words.append(w2)
            cur = nx
        return " ".join(words), cur

    productions = {}
    for node, name in names.items():
        alts = []
        if node.final and outdeg[node] > 0:
            alts.append("ε")
        for w, ch in sorted(node.edges.items()):
            phrase, target = walk_phrase(w, ch)
            term = f'"{phrase}"'
            if is_sink(target):
                alts.append(term)
            else:
                alts.append(f"{term} {names[target]}")
        productions[name] = alts
    order = ["S"] + sorted((v for v in names.values() if v != "S"),
                           key=lambda s: int(s[1:]))
    return productions, order, nodes, cut


def enumerate_language(root):
    """All accepted word-tuples (final at any point along a path)."""
    out = set()
    def dfs(n, acc):
        if n.final:
            out.add(tuple(acc))
        for w, c in n.edges.items():
            acc.append(w); dfs(c, acc); acc.pop()
    dfs(root, [])
    return out


def leaves_to_wordseqs(tok, leaves):
    seqs, probs = [], []
    for cb, seq, leaf, reason in leaves:
        s = tok.decode(list(seq))
        words = s.split(" ")
        assert " ".join(words) == s, f"whitespace round-trip failed: {s!r}"
        seqs.append(tuple(words))
        probs.append(2 ** (-cb))
    return seqs, probs


def grammar_for_task(tok, model, df, idx, budget, max_depth):
    user = df.iloc[idx]["prompt"][0]["content"]
    _, ctx_ids = build_context(tok, user)
    _, leaves, truncated = enumerate_budget_tree(model, tok, ctx_ids, budget, max_depth)
    seqs, probs = leaves_to_wordseqs(tok, leaves)
    records = []
    for rank, ((cb, seq, leaf, reason), words, p) in enumerate(zip(leaves, seqs, probs)):
        records.append(dict(task=idx, rank=rank, bits=round(cb, 4), prob=p,
                            reason=reason, text=tok.decode(list(seq)), words=list(words)))
    root = minimize(build_trie(seqs))
    prods, order, nodes, cut = induce_grammar(root)
    lang = enumerate_language(root)
    assert lang == set(seqs), "induced grammar language != leaf set!"
    return dict(idx=idx, seqs=set(seqs), root=root, prods=prods, order=order,
                n_states=len(nodes), n_edges=sum(len(n.edges) for n in nodes),
                n_leaves=len(seqs), truncated=truncated, records=records)


def trace_prefix(root, words):
    """Longest prefix of `words` that traces a valid path; (L, exact_accept)."""
    n, i = root, 0
    while i < len(words) and words[i] in n.edges:
        n = n.edges[words[i]]
        i += 1
    return i, (i == len(words) and n.final)


def leave_one_out(G, tasks):
    print("\n=== leave-one-out: grammar from all-but-one task vs held-out leaves ===")
    print("held  n_leaf  exact  full-trace   median_L  median_L/len   (train grammar leaves)")
    agg_exact = agg_full = agg_n = 0
    for h in tasks:
        train_seqs = set().union(*(G[t]["seqs"] for t in tasks if t != h))
        root = minimize(build_trie(train_seqs))
        held = G[h]["seqs"]
        Ls, fracs, exact, full = [], [], 0, 0
        for w in held:
            L, ex = trace_prefix(root, w)
            Ls.append(L)
            fracs.append(L / len(w))
            exact += ex
            full += (L == len(w))
        med = sorted(Ls)[len(Ls) // 2]
        medfrac = sorted(fracs)[len(fracs) // 2]
        print(f"{h:>4}  {len(held):>6}  {exact:>5}  {full:>10}   {med:>8}   {medfrac:>12.2f}   "
              f"{len(train_seqs)}")
        agg_exact += exact
        agg_full += full
        agg_n += len(held)
    print(f"\nTOTAL held-out leaves {agg_n}: exact-match {agg_exact} "
          f"({100*agg_exact/agg_n:.1f}%), full-path-traced {agg_full} "
          f"({100*agg_full/agg_n:.1f}%)")


def print_grammar(g):
    print(f"# induced grammar  (task {g['idx']}, verified exact)")
    print(f"# {g['n_leaves']} leaves | DAWG {g['n_states']} states, {g['n_edges']} edges "
          f"| {len(g['prods'])} nonterminals")
    for name in g["order"]:
        alts = g["prods"][name]
        head = f"{name} ::= "
        pad = " " * (len(head) - 2) + "| "
        print(head + alts[0])
        for a in alts[1:]:
            print(pad + a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/armD2_bok/hf_merged")
    ap.add_argument("--data", default="data/verl_witness_long/val.parquet")
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--show-task", type=int, default=0)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--max-depth", type=int, default=64)
    ap.add_argument("--dump-leaves", default="")
    args = ap.parse_args()

    tok, model = load_model(args.model)
    df = pd.read_parquet(args.data)
    tasks = [int(x) for x in args.tasks.split(",")]
    G = {i: grammar_for_task(tok, model, df, i, args.budget, args.max_depth) for i in tasks}

    if args.dump_leaves:
        import json
        with open(args.dump_leaves, "w") as f:
            for i in tasks:
                for r in G[i]["records"]:
                    f.write(json.dumps(r) + "\n")
        n = sum(len(G[i]["records"]) for i in tasks)
        print(f"[dumped {n} leaves across {len(tasks)} tasks -> {args.dump_leaves}]\n")

    print_grammar(G[args.show_task])

    print("\n=== structural sizes per task ===")
    print("task  leaves  states  edges  truncated")
    for i in tasks:
        g = G[i]
        print(f"{i:>4}  {g['n_leaves']:>6}  {g['n_states']:>6}  {g['n_edges']:>5}  {g['truncated']}")

    print("\n=== pairwise leaf-set Jaccard (does the grammar change between tasks?) ===")
    print("      " + "  ".join(f"t{j}" for j in tasks))
    for i in tasks:
        row = []
        for j in tasks:
            a, b = G[i]["seqs"], G[j]["seqs"]
            row.append(f"{len(a & b) / len(a | b):.2f}")
        print(f"  t{i}  " + "  ".join(f"{v:>4}" for v in row))

    # intersection / union sizes to characterise the shared core
    allsets = [G[i]["seqs"] for i in tasks]
    inter = set.intersection(*allsets)
    union = set.union(*allsets)
    print(f"\nleaves common to ALL {len(tasks)} tasks: {len(inter)} / union {len(union)}")

    pairs = [(i, j) for i in tasks for j in tasks if i < j]
    print("\n=== mean pairwise Jaccard of first-k-word leaf prefixes "
          "(template invariance vs depth) ===")
    maxk = max(len(s) for g in G.values() for s in g["seqs"])
    for k in list(range(1, 12)) + list(range(12, maxk + 1, 4)):
        sets = {i: {s[:k] for s in G[i]["seqs"]} for i in tasks}
        js = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i, j in pairs]
        print(f"  first {k:>2} words: mean J = {sum(js) / len(js):.2f}")

    vocab = {i: {w for s in G[i]["seqs"] for w in s} for i in tasks}
    vj = [len(vocab[i] & vocab[j]) / len(vocab[i] | vocab[j]) for i, j in pairs]
    print(f"\nmean pairwise VOCAB (unique-word) Jaccard: {sum(vj) / len(vj):.2f}")

    leave_one_out(G, tasks)


if __name__ == "__main__":
    main()
