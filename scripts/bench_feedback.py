"""Benchmark: Phase 3 visual feedback performance paths.

Measures on the real code path (VisualFeedback.update_eeg):
  1. Producer throughput with/without downsampling (events/s, us/event)
  2. Per-event serialization cost for K SSE clients
     (Phase 3 pre-serialized vs naive per-client json.dumps)
  3. Payload size reduction from downsampling

Run:  python scripts/bench_feedback.py
"""
import json
import os
import random
import sys
import time
from queue import Empty

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feedback.visual_feedback import VisualFeedback


def make_batch(n_samples, n_channels=8, seed=42):
    rnd = random.Random(seed)
    return [[round(rnd.uniform(-50.0, 50.0), 3) for _ in range(n_channels)]
            for _ in range(n_samples)]


def drain(queue):
    dropped = 0
    while True:
        try:
            queue.get_nowait()
            dropped += 1
        except Empty:
            return dropped


def bench_producer(server, batch, n_events=2000):
    """Push + consume loop through the real update_eeg path."""
    q = server._eeg_queue
    # warmup
    for _ in range(50):
        server.update_eeg(batch)
        drain(q)
    lat = []
    t0 = time.perf_counter()
    for _ in range(n_events):
        t1 = time.perf_counter()
        server.update_eeg(batch)
        drain(q)
        lat.append(time.perf_counter() - t1)
    dt = time.perf_counter() - t0
    lat.sort()
    return {
        "events_per_s": round(n_events / dt, 1),
        "us_per_event": round(dt / n_events * 1e6, 1),
        "p50_us": round(lat[len(lat) // 2] * 1e6, 1),
        "p99_us": round(lat[int(len(lat) * 0.99)] * 1e6, 1),
    }


def bench_clients(payload_dict, k_clients, n_events=2000):
    """Serialization cost per event for K concurrent SSE clients."""
    # naive: every client serializes the dict independently
    t0 = time.perf_counter()
    for _ in range(n_events):
        for _ in range(k_clients):
            json.dumps(payload_dict)
    naive = (time.perf_counter() - t0) / n_events * 1e6

    # Phase 3: serialize once, reuse the same string for all clients
    t0 = time.perf_counter()
    for _ in range(n_events):
        s = json.dumps(payload_dict)
        for _ in range(k_clients):
            _ = len(s)
    pre = (time.perf_counter() - t0) / n_events * 1e6
    return {"k_clients": k_clients, "naive_us": round(naive, 1),
            "pre_serialized_us": round(pre, 1),
            "saved_us": round(naive - pre, 1),
            "speedup": round(naive / pre, 2) if pre > 0 else None}


def main():
    print("=" * 64)
    print("NeuroDecode Phase 3 feedback pipeline benchmark")
    print("=" * 64)

    results = {"producer": {}, "clients": {}, "payload": {}}

    # --- 1. producer throughput: batch size x downsample factor ---
    for n_samples in (32, 256, 512):
        batch = make_batch(n_samples)
        for factor in (1, 4):
            srv = VisualFeedback(eeg_downsample=factor)
            r = bench_producer(srv, batch)
            results["producer"][f"samples={n_samples},factor={factor}"] = r
            print(f"[producer] window={n_samples:>3}x8ch  downsample={factor}  "
                  f"->  {r['events_per_s']:>10,.0f} ev/s   "
                  f"({r['us_per_event']:>7.1f} us/ev, p99={r['p99_us']:.1f} us)")
            srv.stop()

    # --- 2. multi-client serialization: pre-serialized vs naive ---
    batch = make_batch(256)
    payload = {"type": "eeg_batch", "data": batch}
    for k in (1, 2, 5):
        r = bench_clients(payload, k)
        results["clients"][f"k={k}"] = r
        print(f"[clients]   k={k}  naive={r['naive_us']:>8.1f} us/ev  "
              f"pre-serialized={r['pre_serialized_us']:>8.1f} us/ev  "
              f"speedup={r['speedup']}x")

    # --- 3. payload size ---
    for n_samples in (256, 512):
        full = json.dumps({"type": "eeg_batch", "data": make_batch(n_samples)})
        ds = json.dumps({"type": "eeg_batch",
                         "data": make_batch(n_samples)[::4]})
        results["payload"][f"samples={n_samples}"] = {
            "full_bytes": len(full), "downsampled_bytes": len(ds),
            "reduction_pct": round((1 - len(ds) / len(full)) * 100, 1)}
        print(f"[payload]   window={n_samples:>3}  {len(full):>7,} B -> "
              f"{len(ds):>6,} B  "
              f"({results['payload'][f'samples={n_samples}']['reduction_pct']}% smaller)")

    with open("bench_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to bench_results.json")


if __name__ == "__main__":
    main()
