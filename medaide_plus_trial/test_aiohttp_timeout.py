"""Test that aiohttp total timeout fires correctly for Ollama streaming."""
import asyncio
import time
import json
import aiohttp


async def test_timeout():
    t0 = time.strftime("%H:%M:%S")
    print(f"Starting aiohttp request at {t0}", flush=True)
    timeout = aiohttp.ClientTimeout(total=10.0, connect=5.0)
    collected = []
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "gemma3:4b",
                    "messages": [{"role": "user", "content": "Write a very long 2000-word essay about medicine."}],
                    "stream": True,
                    "keep_alive": -1,
                    "options": {"num_predict": 2000, "temperature": 0.0},
                },
            ) as resp:
                print(f"Headers received at {time.strftime('%H:%M:%S')}", flush=True)
                async for raw in resp.content:
                    txt = raw.decode("utf-8", errors="ignore").strip()
                    if not txt:
                        continue
                    for line in txt.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            tok = chunk.get("message", {}).get("content", "")
                            if tok:
                                collected.append(tok)
                        except Exception:
                            pass
    except asyncio.TimeoutError:
        t1 = time.strftime("%H:%M:%S")
        print(f">>> asyncio.TimeoutError fired at {t1} ({len(collected)} tokens collected)", flush=True)
    except aiohttp.ServerTimeoutError as e:
        print(f">>> ServerTimeoutError: {e}", flush=True)
    except Exception as e:
        print(f">>> Exception: {type(e).__name__}: {e}", flush=True)
    print(f"Done at {time.strftime('%H:%M:%S')}. Total tokens: {len(collected)}", flush=True)


asyncio.run(test_timeout())
