"""Experiment E: Approximate Internal State / Soft-Prompt Transfer.

Tests whether compressed hidden states (K=8 chunk-pooled latent vectors) from
SmolLM2-135M-Instruct (d=576) can be projected via a trained adapter into
Qwen2.5-0.5B-Instruct's (d=896) input embedding space (inputs_embeds) to
initialize the frozen receiver on unseen QA tasks without feeding original
context tokens.

Compares 4 conditions on 20 unseen evaluation examples (disjoint from 48 train):
1. zero_context      : receiver sees only the question
2. random_projection : receiver sees K=8 embeddings from an unlearned adapter + question
3. learned_cmcp      : receiver sees K=8 embeddings from the trained adapter + question
4. full_text         : receiver sees the full original context + question
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SENDER_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
RECEIVER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_LATENTS = 8


@dataclass(frozen=True)
class QAExample:
    context: str
    question: str
    answer: str
    keywords: tuple[str, ...]


# 48 training examples (strictly disjoint entities/facts from eval)
TRAIN_DATA: list[QAExample] = [
    QAExample("Server Alpha-7 operates on port 8443 using TLS 1.3.", "Which port does Server Alpha-7 use?", " 8443.", ("8443",)),
    QAExample("The Zephyr database stores cold logs on NVMe tier-2.", "Where does Zephyr store cold logs?", " NVMe tier-2.", ("nvme", "tier-2")),
    QAExample("Project Orion has a memory budget of 64 gigabytes.", "What is Project Orion's memory budget?", " 64 gigabytes.", ("64", "gigabytes")),
    QAExample("The Cobalt compiler targets RISC-V 64-bit cores.", "Which architecture does Cobalt target?", " RISC-V 64-bit.", ("risc-v",)),
    QAExample("Sensor Node B4 samples temperature every 250 milliseconds.", "How often does Sensor Node B4 sample?", " 250 milliseconds.", ("250",)),
    QAExample("The Helix scheduler uses a time quantum of 12 milliseconds.", "What is the Helix time quantum?", " 12 milliseconds.", ("12",)),
    QAExample("Gateway West-9 routes traffic to subnet 10.44.0.0/16.", "Which subnet does Gateway West-9 route to?", " 10.44.0.0/16.", ("10.44.0.0/16",)),
    QAExample("The Nimbus cache evicts entries after 900 seconds.", "What is the Nimbus cache TTL?", " 900 seconds.", ("900",)),
    QAExample("Module Kestrel depends on libzstd version 1.5.5.", "Which library version does Kestrel require?", " libzstd 1.5.5.", ("libzstd", "1.5.5")),
    QAExample("Cluster Delta-3 contains 16 worker nodes.", "How many worker nodes are in Cluster Delta-3?", " 16.", ("16",)),
    QAExample("The Polaris queue enforces a max payload of 4 megabytes.", "What is the max payload for Polaris?", " 4 megabytes.", ("4", "megabytes")),
    QAExample("Backup job Vulcan runs daily at 03:15 UTC.", "When does backup job Vulcan run?", " 03:15 UTC.", ("03:15",)),
    QAExample("The Meridian index uses a B-tree with branching factor 128.", "What branching factor does Meridian use?", " 128.", ("128",)),
    QAExample("Worker pool Vega is capped at 32 concurrent threads.", "How many concurrent threads can Vega run?", " 32.", ("32",)),
    QAExample("The Titan ledger writes checkpoints every 5000 blocks.", "How often does Titan write checkpoints?", " 5000 blocks.", ("5000",)),
    QAExample("Device Atlas-2 communicates over UART at 115200 baud.", "What baud rate does Atlas-2 use?", " 115200.", ("115200",)),
    QAExample("The Argon tokenizer uses a vocabulary of 32000 subwords.", "What is the Argon vocabulary size?", " 32000.", ("32000",)),
    QAExample("Service Borealis listens on loopback address 127.0.0.9.", "Which address does Borealis listen on?", " 127.0.0.9.", ("127.0.0.9",)),
    QAExample("The Quartz profiler samples stack frames at 99 hertz.", "At what frequency does Quartz sample?", " 99 hertz.", ("99",)),
    QAExample("Pipeline Solaris processes Parquet files compressed with Snappy.", "Which compression does Solaris use?", " Snappy.", ("snappy",)),
    QAExample("Router Echo-5 drops packets when queue depth exceeds 2048.", "At what queue depth does Echo-5 drop packets?", " 2048.", ("2048",)),
    QAExample("The Stratos kernel reserves 256 megabytes for DMA buffers.", "How much memory does Stratos reserve for DMA?", " 256 megabytes.", ("256",)),
    QAExample("Agent Lyra authenticates requests with Ed25519 signatures.", "Which signature scheme does Lyra use?", " Ed25519.", ("ed25519",)),
    QAExample("The Cygnus broker replicates partitions with factor 3.", "What is the Cygnus replication factor?", " 3.", ("3",)),
    QAExample("Container Zenith mounts its config volume in read-only mode.", "How does Zenith mount its config volume?", " read-only mode.", ("read-only",)),
    QAExample("The Rigel solver converges within 40 iterations.", "How many iterations does Rigel need?", " 40.", ("40",)),
    QAExample("Endpoint Nova returns HTTP status 429 when rate-limited.", "Which HTTP status does Nova return when rate-limited?", " 429.", ("429",)),
    QAExample("The Draco filesystem aligns blocks to 4096 bytes.", "What block alignment does Draco use?", " 4096 bytes.", ("4096",)),
    QAExample("Replica Set Juno elects a primary within 5 seconds.", "How fast does Juno elect a primary?", " 5 seconds.", ("5",)),
    QAExample("The Siren monitor triggers an alert above 85 percent CPU.", "At what CPU threshold does Siren alert?", " 85 percent.", ("85",)),
    QAExample("Kernel module Hydra registers character device major 241.", "What major device number does Hydra register?", " 241.", ("241",)),
    QAExample("The Chronos clock synchronizes via PTP to within 2 microseconds.", "What precision does Chronos achieve?", " 2 microseconds.", ("2", "microseconds")),
    QAExample("Batch runner Castor allocates 8 GPU shards per job.", "How many GPU shards does Castor allocate?", " 8.", ("8",)),
    QAExample("The Mirage proxy buffers up to 512 kilobytes per stream.", "How much does Mirage buffer per stream?", " 512 kilobytes.", ("512",)),
    QAExample("Table Archive-9 is partitioned by calendar month.", "How is Archive-9 partitioned?", " calendar month.", ("month",)),
    QAExample("The Vela encoder quantizes weights to 4-bit integers.", "To what precision does Vela quantize weights?", " 4-bit integers.", ("4-bit",)),
    QAExample("Daemon Pulse checks disk health every 60 minutes.", "How often does Daemon Pulse check disks?", " 60 minutes.", ("60",)),
    QAExample("The Taurus ring buffer holds 1024 descriptors.", "How many descriptors does Taurus hold?", " 1024.", ("1024",)),
    QAExample("Firewall rule Amber blocks outbound UDP port 5353.", "Which UDP port does rule Amber block?", " 5353.", ("5353",)),
    QAExample("The Phoenix runtime compiles Lua scripts to bytecode.", "Which language does Phoenix compile?", " Lua.", ("lua",)),
    QAExample("Storage pool Crest uses RAID-10 across 8 disks.", "Which RAID level does Crest use?", " RAID-10.", ("raid-10",)),
    QAExample("The Aquila tracer samples 10 percent of incoming spans.", "What sampling rate does Aquila use?", " 10 percent.", ("10",)),
    QAExample("Lock manager Flint enforces a 30-second lease timeout.", "What is Flint's lease timeout?", " 30 seconds.", ("30",)),
    QAExample("The Corvus model uses 12 attention heads per layer.", "How many attention heads does Corvus have?", " 12.", ("12",)),
    QAExample("Exporter Finch exposes Prometheus metrics on path /metricsz.", "On which path does Finch expose metrics?", " /metricsz.", ("metricsz",)),
    QAExample("The Lynx parser validates schemas against Draft-2020-12.", "Which schema draft does Lynx validate against?", " Draft-2020-12.", ("2020-12",)),
    QAExample("Broker Pike persists offsets in an internal compacted topic.", "Where does Pike persist offsets?", " compacted topic.", ("compacted", "topic")),
    QAExample("The Dorado engine executes joins using hash partitioning.", "How does Dorado execute joins?", " hash partitioning.", ("hash",)),
]

# 20 unseen evaluation examples (completely disjoint entities, numbers, and domains)
EVAL_DATA: list[QAExample] = [
    QAExample("Satellite Beacon-11 transmits telemetry on frequency 437.5 megahertz.", "On what frequency does Beacon-11 transmit?", " 437.5 megahertz.", ("437.5",)),
    QAExample("The Obsidian vault rotates encryption keys every 14 days.", "How often does Obsidian rotate keys?", " 14 days.", ("14",)),
    QAExample("Turbine Unit-6 reaches peak efficiency at 3600 RPM.", "At what RPM does Turbine Unit-6 peak?", " 3600 RPM.", ("3600",)),
    QAExample("The Kestrel-X drone carries a lithium battery rated at 5200 mAh.", "What is the Kestrel-X battery capacity?", " 5200 mAh.", ("5200",)),
    QAExample("Library module Argon2id uses 3 iterations for password hashing.", "How many iterations does Argon2id use here?", " 3.", ("3",)),
    QAExample("The Cascadia bridge span measures 1420 meters in length.", "How long is the Cascadia bridge span?", " 1420 meters.", ("1420",)),
    QAExample("Sensor Array Tango logs barometric pressure in hectopascals.", "Which unit does Tango log pressure in?", " hectopascals.", ("hectopascals",)),
    QAExample("The Halcyon synthesizer features 8 polyphonic voices.", "How many polyphonic voices does Halcyon have?", " 8.", ("8",)),
    QAExample("Protocol Whisper-2 negotiates sessions using Curve25519.", "Which curve does Whisper-2 use?", " Curve25519.", ("curve25519",)),
    QAExample("The Boreal glacier retreated by 18 meters last summer.", "How far did the Boreal glacier retreat?", " 18 meters.", ("18",)),
    QAExample("Microcontroller Pico-9 includes 264 kilobytes of SRAM.", "How much SRAM does Pico-9 include?", " 264 kilobytes.", ("264",)),
    QAExample("The Saffron recipe calls for steeping leaves for 4 minutes.", "How long should the leaves steep?", " 4 minutes.", ("4",)),
    QAExample("Flight Corridor Foxtrot restricts altitude to 28000 feet.", "What is the altitude limit in Corridor Foxtrot?", " 28000 feet.", ("28000",)),
    QAExample("The Graphite renderer outputs frames at 120 FPS.", "At what frame rate does Graphite render?", " 120 FPS.", ("120",)),
    QAExample("Reactor Loop-4 circulates liquid sodium coolant.", "What coolant does Reactor Loop-4 circulate?", " liquid sodium.", ("sodium",)),
    QAExample("The Tempest modem supports a downlink of 150 megabits per second.", "What is Tempest's downlink speed?", " 150 megabits per second.", ("150",)),
    QAExample("Archive Box-19 contains microfilms from year 1974.", "From what year are the microfilms in Box-19?", " 1974.", ("1974",)),
    QAExample("The Prism spectrometer resolves wavelengths down to 0.2 nanometers.", "What resolution does Prism achieve?", " 0.2 nanometers.", ("0.2",)),
    QAExample("Warehouse Zone-C stores pallets at minus 20 degrees Celsius.", "At what temperature is Zone-C kept?", " minus 20 degrees Celsius.", ("20",)),
    QAExample("The Monarch butterfly migration covers roughly 4000 kilometers.", "How far is the Monarch migration?", " 4000 kilometers.", ("4000",)),
]


def chunk_pool_latents(hidden: torch.Tensor, n_latents: int = N_LATENTS) -> torch.Tensor:
    """Compress (L, D) hidden states into (n_latents, D) via adaptive chunk mean pooling."""
    seq_len, d = hidden.shape
    if seq_len >= n_latents:
        indices = torch.linspace(0, seq_len, steps=n_latents + 1).long()
        chunks = []
        for i in range(n_latents):
            start = int(indices[i].item())
            end = max(start + 1, int(indices[i + 1].item()))
            chunks.append(hidden[start:end].mean(dim=0))
        return torch.stack(chunks, dim=0)
    # Pad/repeat if shorter than n_latents
    reps = (n_latents + seq_len - 1) // seq_len
    tiled = hidden.repeat(reps, 1)[:n_latents]
    return tiled


@torch.no_grad()
def extract_sender_latents(
    examples: list[QAExample], n_latents: int = N_LATENTS
) -> tuple[torch.Tensor, float]:
    """Extract (N, n_latents, d_src) from SmolLM2-135M-Instruct."""
    tok = AutoTokenizer.from_pretrained(SENDER_MODEL, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(SENDER_MODEL, local_files_only=True)
    model.eval()

    t0 = time.perf_counter()
    latents = []
    for ex in examples:
        enc = tok(ex.context, return_tensors="pt", truncation=True, max_length=128)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0].float()  # (L, 576)
        latents.append(chunk_pool_latents(h, n_latents=n_latents))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return torch.stack(latents, dim=0), elapsed_ms


class SoftPromptAdapter(nn.Module):
    """Maps K compressed sender latents (B, K, d_src) -> receiver inputs_embeds (B, K, d_dst)."""

    def __init__(self, d_src: int, d_dst: int, n_latents: int = N_LATENTS, target_std: float = 0.025) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(d_src)
        self.proj = nn.Sequential(
            nn.Linear(d_src, d_src),
            nn.GELU(),
            nn.Linear(d_src, d_dst),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, n_latents, d_dst) * 0.01)
        self.norm_out = nn.LayerNorm(d_dst)
        self.out_scale = nn.Parameter(torch.tensor(target_std))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.norm_in(z)) + self.pos_emb
        return self.norm_out(h) * self.out_scale


def format_question(ex: QAExample) -> str:
    return f"Question: {ex.question}\nAnswer:"


def format_full_prompt(ex: QAExample) -> str:
    return f"Context: {ex.context}\nQuestion: {ex.question}\nAnswer:"


def train_adapter(
    receiver: AutoModelForCausalLM,
    tok: AutoTokenizer,
    train_latents: torch.Tensor,
    train_data: list[QAExample],
    d_src: int,
    d_dst: int,
    target_std: float,
    steps: int = 24,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[SoftPromptAdapter, float, list[float]]:
    torch.manual_seed(seed)
    adapter = SoftPromptAdapter(d_src=d_src, d_dst=d_dst, n_latents=N_LATENTS, target_std=target_std)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-2)
    embed_layer = receiver.get_input_embeddings()

    # Pre-tokenize question + answer pairs
    q_ids_list = [tok(format_question(ex), add_special_tokens=False, return_tensors="pt")["input_ids"][0] for ex in train_data]
    a_ids_list = [tok(ex.answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0] for ex in train_data]

    n = len(train_data)
    losses: list[float] = []
    t0 = time.perf_counter()
    adapter.train()

    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        opt.zero_grad()
        batch_loss = torch.tensor(0.0)

        for b_i in idx.tolist():
            z_b = train_latents[b_i : b_i + 1]  # (1, K, d_src)
            soft_prompt = adapter(z_b)  # (1, K, d_dst)

            q_ids = q_ids_list[b_i].unsqueeze(0)
            a_ids = a_ids_list[b_i].unsqueeze(0)
            with torch.no_grad():
                q_emb = embed_layer(q_ids)
                a_emb = embed_layer(a_ids)

            inputs_embeds = torch.cat([soft_prompt, q_emb, a_emb], dim=1)
            ignore_prefix = torch.full((1, N_LATENTS + q_ids.size(1)), -100, dtype=torch.long)
            labels = torch.cat([ignore_prefix, a_ids], dim=1)

            out = receiver(inputs_embeds=inputs_embeds, labels=labels)
            batch_loss = batch_loss + out.loss / batch_size

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        losses.append(float(batch_loss.item()))
        if (step + 1) % 6 == 0 or step == 0:
            print(f"  [Train step {step + 1:2d}/{steps}] loss = {losses[-1]:.4f}")

    train_ms = (time.perf_counter() - t0) * 1000.0
    adapter.eval()
    return adapter, train_ms, losses


@torch.no_grad()
def evaluate_condition(
    condition: str,
    receiver: AutoModelForCausalLM,
    tok: AutoTokenizer,
    eval_data: list[QAExample],
    eval_latents: torch.Tensor | None = None,
    adapter: SoftPromptAdapter | None = None,
) -> dict:
    embed_layer = receiver.get_input_embeddings()
    total_nll = 0.0
    total_ans_tokens = 0
    exact_matches = 0
    keyword_hits = 0
    total_keywords = 0
    total_input_tokens = 0
    latencies_ms: list[float] = []

    for i, ex in enumerate(eval_data):
        t0 = time.perf_counter()
        prompt_text = format_full_prompt(ex) if condition == "full_text" else format_question(ex)
        q_ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        a_ids = tok(ex.answer, add_special_tokens=False, return_tensors="pt")["input_ids"]

        q_emb = embed_layer(q_ids)
        a_emb = embed_layer(a_ids)

        if condition in ("learned_cmcp", "random_projection"):
            assert eval_latents is not None and adapter is not None
            soft_prompt = adapter(eval_latents[i : i + 1])  # (1, K, d_dst)
            prompt_embeds = torch.cat([soft_prompt, q_emb], dim=1)
            full_embeds = torch.cat([prompt_embeds, a_emb], dim=1)
            n_prompt_tokens = N_LATENTS + q_ids.size(1)
        else:
            prompt_embeds = q_emb
            full_embeds = torch.cat([q_emb, a_emb], dim=1)
            n_prompt_tokens = q_ids.size(1)

        ignore_prefix = torch.full((1, n_prompt_tokens), -100, dtype=torch.long)
        labels = torch.cat([ignore_prefix, a_ids], dim=1)
        out = receiver(inputs_embeds=full_embeds, labels=labels)
        n_ans = int(a_ids.size(1))
        total_nll += float(out.loss.item()) * n_ans
        total_ans_tokens += n_ans
        total_input_tokens += n_prompt_tokens

        # Greedy generation for exact match & keyword recall
        attn_mask = torch.ones((1, prompt_embeds.size(1)), dtype=torch.long)
        gen_ids = receiver.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attn_mask,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        # Transformers generate with inputs_embeds returns only newly generated tokens
        gen_text = tok.decode(gen_ids[0], skip_special_tokens=True).strip()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        target_clean = ex.answer.strip().rstrip(".").lower()
        gen_clean = gen_text.lower()
        if target_clean and target_clean in gen_clean:
            exact_matches += 1
        for kw in ex.keywords:
            total_keywords += 1
            if kw.lower() in gen_clean:
                keyword_hits += 1

    mean_nll = total_nll / max(1, total_ans_tokens)
    ppl = math.exp(min(mean_nll, 20.0))
    return {
        "condition": condition,
        "n_eval": len(eval_data),
        "mean_nll": round(mean_nll, 4),
        "perplexity": round(ppl, 4),
        "exact_match_rate": round(exact_matches / len(eval_data), 4),
        "keyword_recall": round(keyword_hits / max(1, total_keywords), 4),
        "mean_receiver_input_tokens": round(total_input_tokens / len(eval_data), 2),
        "mean_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
    }


def main() -> None:
    assert len(TRAIN_DATA) == 48
    assert len(EVAL_DATA) == 20
    assert {e.context for e in TRAIN_DATA}.isdisjoint({e.context for e in EVAL_DATA})

    print(f"Extracting sender latents (K={N_LATENTS}) from {SENDER_MODEL}...")
    train_latents, train_ext_ms = extract_sender_latents(TRAIN_DATA, n_latents=N_LATENTS)
    eval_latents, eval_ext_ms = extract_sender_latents(EVAL_DATA, n_latents=N_LATENTS)
    d_src = train_latents.shape[-1]

    print(f"Loading receiver {RECEIVER_MODEL} (frozen)...")
    tok = AutoTokenizer.from_pretrained(RECEIVER_MODEL, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(RECEIVER_MODEL, local_files_only=True).float()
    receiver.eval()
    for p in receiver.parameters():
        p.requires_grad_(False)

    d_dst = receiver.config.hidden_size
    emb_std = float(receiver.get_input_embeddings().weight.std().item())
    print(f"Receiver d_dst={d_dst}, token embedding std={emb_std:.5f}")

    # Random unlearned adapter baseline
    torch.manual_seed(999)
    random_adapter = SoftPromptAdapter(d_src=d_src, d_dst=d_dst, n_latents=N_LATENTS, target_std=emb_std)
    random_adapter.eval()

    # Train learned CMCP adapter
    print("Training SoftPromptAdapter (gradients flow through frozen Qwen2.5 inputs_embeds)...")
    learned_adapter, train_ms, train_losses = train_adapter(
        receiver=receiver,
        tok=tok,
        train_latents=train_latents,
        train_data=TRAIN_DATA,
        d_src=d_src,
        d_dst=d_dst,
        target_std=emb_std,
        steps=24,
        batch_size=8,
        lr=1e-3,
        seed=42,
    )

    print("\nEvaluating 4 conditions on 20 unseen examples...")
    results = []
    for cond, lat, adp in [
        ("zero_context", None, None),
        ("random_projection", eval_latents, random_adapter),
        ("learned_cmcp", eval_latents, learned_adapter),
        ("full_text", None, None),
    ]:
        res = evaluate_condition(cond, receiver, tok, EVAL_DATA, eval_latents=lat, adapter=adp)
        results.append(res)
        print(
            f"[{cond:18s}] NLL={res['mean_nll']:.4f} | PPL={res['perplexity']:.2f} | "
            f"EM={res['exact_match_rate']:.2f} | KW={res['keyword_recall']:.2f} | "
            f"Tokens={res['mean_receiver_input_tokens']:.1f} | Lat={res['mean_latency_ms']:.1f}ms"
        )

    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "exp_e_state_transfer.json"
    csv_path = out_dir / "exp_e_state_transfer.csv"

    payload = {
        "experiment": "Experiment E: Approximate Internal State / Soft-Prompt Transfer",
        "sender_model": SENDER_MODEL,
        "receiver_model": RECEIVER_MODEL,
        "n_latents": N_LATENTS,
        "n_train": len(TRAIN_DATA),
        "n_eval": len(EVAL_DATA),
        "sender_train_extract_ms": round(train_ext_ms, 2),
        "sender_eval_extract_ms": round(eval_ext_ms, 2),
        "adapter_train_ms": round(train_ms, 2),
        "train_losses": [round(x, 4) for x in train_losses],
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved CSV  to {csv_path}")


if __name__ == "__main__":
    main()
