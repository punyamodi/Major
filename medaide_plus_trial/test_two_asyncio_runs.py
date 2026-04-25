"""
Reproduce the benchmark hang: two sequential asyncio.run() calls
using the same OllamaProvider instance (as in benchmark_runner.py).
"""
import asyncio
import sys
import time
sys.path.insert(0, ".")
from medaide_plus.utils.llm_provider import create_provider

cfg = {"provider": "ollama", "model": "gemma3:4b", "base_url": "http://localhost:11434",
       "max_tokens": 512, "temperature": 0.0}
provider = create_provider(cfg)

msg = [{"role": "user", "content": "Write a detailed 600-word medical essay about diabetes management."}]

# Simulate the benchmark: first asyncio.run() = baseline MedAide
print(f"[{time.strftime('%H:%M:%S')}] Call 1 (baseline)...", flush=True)
t0 = time.time()
resp1 = asyncio.run(provider.chat(msg, max_tokens=512))
print(f"[{time.strftime('%H:%M:%S')}] Call 1 done in {time.time()-t0:.1f}s, {len(resp1.text)} chars", flush=True)

# Simulate second asyncio.run() = MedAide+
print(f"[{time.strftime('%H:%M:%S')}] Call 2 (MedAide+)...", flush=True)
t0 = time.time()
resp2 = asyncio.run(provider.chat(msg, max_tokens=512))
elapsed = time.time() - t0
print(f"[{time.strftime('%H:%M:%S')}] Call 2 done in {elapsed:.1f}s, {len(resp2.text)} chars", flush=True)

if elapsed < 200:
    print("TEST PASSED — no hang")
else:
    print("TEST FAILED — hung!")
