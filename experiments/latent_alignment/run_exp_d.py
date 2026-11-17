"""Experiment D: Cross-Model Shared Latent Representation Alignment.

Extracts real hidden states from:
- Sender:   HuggingFaceTB/SmolLM2-135M-Instruct (LlamaForCausalLM, d=576)
- Receiver: Qwen/Qwen2.5-0.5B-Instruct (Qwen2ForCausalLM, d=896)

Evaluates Orthogonal Procrustes, Ridge Regression, 2-layer MLP, and Random
unlearned projectors on a strict 64-train / 24-eval disjoint context split.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from representations.latent import (  # noqa: E402
    LatentProjector,
    centered_kernel_alignment,
    cosine_similarity_mean,
    mse_loss,
)

SENDER_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
RECEIVER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# 64 training contexts across technical, reasoning, and code domains.
TRAIN_CONTEXTS: list[str] = [
    # Technical / Systems (22)
    "A B-tree keeps sorted data in balanced pages so lookups, insertions, and range scans run in logarithmic time.",
    "Write-ahead logging records every page modification sequentially on disk before applying changes to the main heap.",
    "Consistent hashing maps both servers and keys onto a unit ring so adding a node only moves a small fraction of keys.",
    "An inverted index maps each normalized term to a sorted list of document identifiers for fast full-text search.",
    "Copy-on-write snapshots share unchanged memory pages between parent and child processes until a write triggers duplication.",
    "TCP congestion control uses an additive-increase multiplicative-decrease window to avoid saturating bottleneck links.",
    "A Bloom filter tests set membership using k hash functions and a bit array, allowing false positives but zero false negatives.",
    "Virtual memory translates virtual addresses to physical frames through multi-level page tables cached in the TLB.",
    "Raft consensus elects a single leader per term and replicates log entries across a quorum before committing.",
    "LSM-trees buffer writes in a sorted memtable and flush immutable SSTables that are periodically merged by compaction.",
    "Read-copy-update allows readers to traverse lock-free data structures while writers publish new versions and reclaim old ones after a grace period.",
    "Zero-copy networking transfers data from page cache directly to the network socket buffer via sendfile without user-space copies.",
    "A ring buffer uses modular head and tail indices in a contiguous array to implement a lock-free single-producer queue.",
    "HTTP/3 runs over QUIC to eliminate head-of-line blocking across multiplexed streams and integrate TLS 1.3 handshake.",
    "Memory-mapped files bind a virtual address range directly to a file descriptor so page faults lazily load disk blocks.",
    "Vector clocks assign each node a logical counter vector to detect causal ordering and concurrent conflicts in distributed stores.",
    "A slab allocator pre-allocates fixed-size object caches to eliminate internal fragmentation and speed up kernel allocations.",
    "Adaptive radix trees compress sparse trie nodes into variable-capacity layouts to fit keys inside CPU cache lines.",
    "Two-phase commit coordinates distributed transactions using a prepare phase followed by a durable commit or abort decision.",
    "Gossip protocols disseminate cluster membership and failure heartbeats probabilistically with bounded per-node bandwidth.",
    "SIMD vectorization processes multiple packed data lanes in a single CPU instruction to accelerate matrix kernels.",
    "Backpressure signals upstream producers to slow emission rates whenever downstream consumer queues approach capacity.",
    # Reasoning / Science / Math (21)
    "If all prime numbers greater than two are odd, then the sum of any two primes greater than two must be even.",
    "Bayes theorem updates the prior probability of a hypothesis by multiplying by the likelihood and normalizing by the evidence.",
    "Entropy in information theory measures the minimum average number of bits required to encode symbols from a distribution.",
    "A convex function lies below every secant line connecting two points on its graph, ensuring any local minimum is global.",
    "By the pigeonhole principle, placing n plus one items into n containers guarantees at least one container holds two items.",
    "In a directed acyclic graph, topological sorting orders vertices so every directed edge points from earlier to later.",
    "The central limit theorem states that the normalized sum of independent identically distributed variables converges to a Gaussian.",
    "Dynamic programming solves overlapping subproblems once and stores their results to avoid exponential recomputation.",
    "Eigenvalues of a symmetric real matrix are always real, and eigenvectors for distinct eigenvalues are orthogonal.",
    "A Nash equilibrium occurs when no player can improve their expected payoff by unilaterally changing their strategy.",
    "Simpson's paradox arises when a trend appears in separate groups of data but reverses when the groups are aggregated.",
    "Gradient descent follows the negative gradient of a differentiable objective, with step size bounded by the Lipschitz constant.",
    "Markov chains satisfy the memoryless property because the next state distribution depends solely on the current state.",
    "An NP-complete problem belongs to NP and every problem in NP reduces to it in polynomial time.",
    "The Cauchy-Schwarz inequality bounds the absolute inner product of two vectors by the product of their Euclidean norms.",
    "Amortized analysis averages the cost of expensive occasional operations over a long sequence of cheap operations.",
    "A bijective function pairs every element of the domain with a unique element of the codomain, guaranteeing an inverse.",
    "In thermodynamics, reversible adiabatic expansion cools an ideal gas because internal energy is converted into work.",
    "Fourier transforms decompose time-domain signals into complex sinusoids indexed by frequency and phase.",
    "Proof by contradiction assumes the negation of the target claim and derives a logical inconsistency from accepted axioms.",
    "Singular value decomposition factors any real matrix into orthogonal left and right singular vectors scaled by non-negative singular values.",
    # Code / Software Engineering (21)
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs) // 2]\n    return quicksort([x for x in xs if x < pivot]) + [x for x in xs if x == pivot] + quicksort([x for x in xs if x > pivot])",
    "class LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n        self.data = {}\n    def get(self, k):\n        if k not in self.data: return -1\n        v = self.data.pop(k)\n        self.data[k] = v\n        return v",
    "SELECT department, COUNT(*) AS headcount FROM employees WHERE active = 1 GROUP BY department HAVING COUNT(*) > 5;",
    "async def fetch_all(urls, session):\n    tasks = [session.get(u) for u in urls]\n    return await asyncio.gather(*tasks)",
    "def fib_dp(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "def topological_sort(graph):\n    indeg = {u: 0 for u in graph}\n    for u in graph:\n        for v in graph[u]: indeg[v] += 1\n    q = [u for u, d in indeg.items() if d == 0]",
    "fn dot_product(a: &[f32], b: &[f32]) -> f32 {\n    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()\n}",
    "def merge_intervals(intervals):\n    intervals.sort(key=lambda x: x[0])\n    out = [intervals[0]]\n    for s, e in intervals[1:]:\n        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)\n        else: out.append([s, e])\n    return out",
    "template <typename T> T clamp(T val, T lo, T hi) { return std::max(lo, std::min(val, hi)); }",
    "def cosine_sim(u, v):\n    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))",
    "WITH ranked AS (SELECT id, score, ROW_NUMBER() OVER (ORDER BY score DESC) AS rk FROM scores) SELECT id FROM ranked WHERE rk <= 10;",
    "def bfs_shortest_path(adj, start, goal):\n    queue = deque([(start, 0)])\n    seen = {start}\n    while queue:\n        node, dist = queue.popleft()\n        if node == goal: return dist",
    "const debounce = (fn, ms) => { let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); }; };",
    "def union_find(parent, x):\n    while parent[x] != x:\n        parent[x] = parent[parent[x]]\n        x = parent[x]\n    return x",
    "def matrix_multiply(A, B):\n    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]",
    "grep -rn 'TODO' src/ | awk -F: '{print $1}' | sort | uniq -c | sort -nr",
    "def sliding_window_max(nums, k):\n    dq = deque()\n    res = []\n    for i, x in enumerate(nums):\n        while dq and nums[dq[-1]] <= x: dq.pop()",
    "CREATE INDEX CONCURRENTLY idx_orders_created_at ON orders (created_at DESC) WHERE status = 'pending';",
    "def power_mod(base, exp, mod):\n    res = 1\n    while exp > 0:\n        if exp & 1: res = (res * base) % mod\n        base = (base * base) % mod\n        exp >>= 1\n    return res",
    "def is_valid_bst(node, lo=float('-inf'), hi=float('inf')):\n    if not node: return True\n    if not (lo < node.val < hi): return False\n    return is_valid_bst(node.left, lo, node.val) and is_valid_bst(node.right, node.val, hi)",
]

# 24 unseen evaluation contexts (disjoint topics, entities, and formulations from train).
EVAL_CONTEXTS: list[str] = [
    # Technical / Systems (8)
    "A skip list layers probabilistic express pointers over a sorted linked list to achieve expected logarithmic search.",
    "HyperLogLog estimates the cardinality of massive multisets using the maximum leading-zero count of hashed elements.",
    "Speculative decoding drafts candidate tokens with a small model and verifies them in parallel with a larger target model.",
    "RoPE rotary position embeddings encode relative token offsets by rotating query and key pairs in two-dimensional subspaces.",
    "FlashAttention tiles attention matrices inside fast GPU SRAM to avoid materializing the quadratic score matrix in HBM.",
    "ephemeral Diffie-Hellman key exchange derives a shared session secret over an insecure channel using modular exponentiation.",
    "A token bucket rate limiter accumulates permits at a steady fill rate and allows short bursts up to bucket capacity.",
    "CRDT conflict-free replicated data types guarantee strong eventual consistency via commutative, associative, idempotent merges.",
    # Reasoning / Science / Math (8)
    "Kullback-Leibler divergence quantifies the extra bits needed when encoding samples from distribution P using a code optimized for Q.",
    "Jensen's inequality states that for a convex function f, f of the expectation is at most the expectation of f.",
    "A Hamiltonian cycle visits every vertex in an undirected graph exactly once before returning to the starting vertex.",
    "Lagrange multipliers find constrained extrema by setting the gradient of the objective proportional to the constraint gradient.",
    "In quantum mechanics, the Heisenberg uncertainty principle bounds the product of position and momentum standard deviations.",
    "Pumping lemma for regular languages shows that sufficiently long strings in a regular language contain a repeatable middle substring.",
    "Spectral clustering partitions a graph by computing eigenvectors of the normalized Laplacian and running k-means on the rows.",
    "A martingale is a stochastic process whose conditional expectation of the next value equals the present observed value.",
    # Code / Software Engineering (8)
    "def levenshtein(a, b):\n    dp = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        prev, dp[0] = dp[0], i\n        for j, cb in enumerate(b, 1):\n            prev, dp[j] = dp[j], prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])\n    return dp[-1]",
    "def reservoir_sample(stream, k):\n    sample = []\n    for i, item in enumerate(stream):\n        if i < k: sample.append(item)\n        elif (j := random.randint(0, i)) < k: sample[j] = item\n    return sample",
    "SELECT user_id, SUM(amount) OVER (PARTITION BY user_id ORDER BY ts ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) FROM txns;",
    "def trie_insert(root, word):\n    node = root\n    for ch in word:\n        node = node.setdefault(ch, {})\n    node['$'] = True",
    "async def retry_backoff(coro_fn, retries=3, base=0.5):\n    for attempt in range(retries):\n        try: return await coro_fn()\n        except Exception: await asyncio.sleep(base * (2 ** attempt))",
    "def kmp_prefix(pattern):\n    pi = [0] * len(pattern)\n    j = 0\n    for i in range(1, len(pattern)):\n        while j > 0 and pattern[i] != pattern[j]: j = pi[j - 1]\n        if pattern[i] == pattern[j]: j += 1\n        pi[i] = j\n    return pi",
    "type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };",
    "def manhattan_dist(p, q):\n    return sum(abs(a - b) for a, b in zip(p, q))",
]


@torch.no_grad()
def extract_hidden_states(
    model_name: str, texts: list[str], batch_size: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_pooled, last_token) numpy arrays of shape (N, hidden_size)."""
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    model.eval()

    mean_list: list[torch.Tensor] = []
    last_list: list[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1].float()  # (B, L, D)
        mask = enc["attention_mask"].float()  # (B, L)

        # Mean pooled over non-pad tokens
        masked_h = h * mask.unsqueeze(-1)
        mean_vec = masked_h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_list.append(mean_vec.cpu())

        # Last non-pad token
        lengths = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(h.size(0))
        last_vec = h[batch_idx, lengths]
        last_list.append(last_vec.cpu())

    return torch.cat(mean_list, dim=0).numpy(), torch.cat(last_list, dim=0).numpy()


def retrieval_accuracy(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Compute cross-model Top-1 and Top-5 retrieval accuracy via cosine similarity."""
    p_norm = F.normalize(pred, p=2, dim=-1)
    t_norm = F.normalize(target, p=2, dim=-1)
    sim = p_norm @ t_norm.T  # (N, N)
    n = sim.size(0)
    ranks = torch.argsort(sim, dim=-1, descending=True)
    targets = torch.arange(n).unsqueeze(1)
    top1 = float((ranks[:, :1] == targets).any(dim=1).float().mean().item())
    top5 = float((ranks[:, :5] == targets).any(dim=1).float().mean().item())
    return top1, top5


def main() -> None:
    assert len(TRAIN_CONTEXTS) == 64
    assert len(EVAL_CONTEXTS) == 24
    assert set(TRAIN_CONTEXTS).isdisjoint(set(EVAL_CONTEXTS)), "Train/Eval leakage detected!"

    print(f"Extracting hidden states from {SENDER_MODEL}...")
    t0 = time.perf_counter()
    src_train_mean, src_train_last = extract_hidden_states(SENDER_MODEL, TRAIN_CONTEXTS)
    src_eval_mean, src_eval_last = extract_hidden_states(SENDER_MODEL, EVAL_CONTEXTS)
    sender_extract_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Extracting hidden states from {RECEIVER_MODEL}...")
    t1 = time.perf_counter()
    dst_train_mean, dst_train_last = extract_hidden_states(RECEIVER_MODEL, TRAIN_CONTEXTS)
    dst_eval_mean, dst_eval_last = extract_hidden_states(RECEIVER_MODEL, EVAL_CONTEXTS)
    receiver_extract_ms = (time.perf_counter() - t1) * 1000.0

    d_src = src_train_mean.shape[1]
    d_dst = dst_train_mean.shape[1]
    print(f"Shapes: sender ({len(TRAIN_CONTEXTS)}+{len(EVAL_CONTEXTS)}, {d_src}), receiver ({len(TRAIN_CONTEXTS)}+{len(EVAL_CONTEXTS)}, {d_dst})")

    pooling_configs = [
        ("mean_pooled", src_train_mean, dst_train_mean, src_eval_mean, dst_eval_mean),
        ("last_token", src_train_last, dst_train_last, src_eval_last, dst_eval_last),
    ]
    methods = ["random", "procrustes", "ridge", "mlp"]

    rows: list[dict] = []
    for pooling_name, x_tr, y_tr, x_ev, y_ev in pooling_configs:
        cka_before = centered_kernel_alignment(x_ev, y_ev)
        y_ev_t = torch.as_tensor(y_ev, dtype=torch.float32)

        for method in methods:
            proj = LatentProjector(
                d_src=d_src,
                d_dst=d_dst,
                method=method,  # type: ignore[arg-type]
                ridge_alpha=100.0,
                mlp_hidden_dim=1024,
                mlp_epochs=300,
                mlp_lr=1e-3,
                mlp_weight_decay=1e-2,
                cosine_loss_weight=0.5,
                seed=42,
            )
            proj.fit(x_tr, y_tr)

            t_inf0 = time.perf_counter()
            pred_ev = proj.transform(x_ev)
            inf_ms = (time.perf_counter() - t_inf0) * 1000.0

            cka_after = centered_kernel_alignment(pred_ev, y_ev_t)
            ev_mse = mse_loss(pred_ev, y_ev_t)
            ev_cos = cosine_similarity_mean(pred_ev, y_ev_t)
            top1, top5 = retrieval_accuracy(pred_ev, y_ev_t)

            row = {
                "pooling": pooling_name,
                "method": method,
                "n_train": len(TRAIN_CONTEXTS),
                "n_eval": len(EVAL_CONTEXTS),
                "d_src": d_src,
                "d_dst": d_dst,
                "cka_before": round(cka_before, 6),
                "cka_after": round(cka_after, 6),
                "eval_mse": round(ev_mse, 6),
                "eval_cosine_sim": round(ev_cos, 6),
                "eval_top1_acc": round(top1, 6),
                "eval_top5_acc": round(top5, 6),
                "train_time_ms": round(proj.train_time_ms, 3),
                "eval_infer_time_ms": round(inf_ms, 3),
            }
            rows.append(row)
            print(
                f"[{pooling_name:11s} | {method:10s}] "
                f"CKA: {cka_before:.4f} -> {cka_after:.4f} | "
                f"Cos: {ev_cos:.4f} | MSE: {ev_mse:.4f} | "
                f"Top1: {top1:.3f} | Top5: {top5:.3f} | "
                f"Train: {proj.train_time_ms:.1f}ms"
            )

    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "exp_d_latent_alignment.json"
    csv_path = out_dir / "exp_d_latent_alignment.csv"

    payload = {
        "experiment": "Experiment D: Cross-Model Shared Latent Representation Alignment",
        "sender_model": SENDER_MODEL,
        "receiver_model": RECEIVER_MODEL,
        "n_train": len(TRAIN_CONTEXTS),
        "n_eval": len(EVAL_CONTEXTS),
        "sender_extract_ms": round(sender_extract_ms, 2),
        "receiver_extract_ms": round(receiver_extract_ms, 2),
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved CSV  to {csv_path}")


if __name__ == "__main__":
    main()
