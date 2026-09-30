"""
Automated UI Health Check for Streaming Live RAG.
Uses Playwright to drive a headless Chromium browser against http://localhost:8000.
Captures screenshots, WebSocket events, console logs, and timing measurements.
"""
import asyncio, json, time, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from playwright.async_api import async_playwright

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'ui_audit')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def ss(name):
    return os.path.join(ARTIFACT_DIR, f"{name}.png")

async def main():
    report = {}
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await ctx.new_page()

        # â”€â”€ Collectors â”€â”€
        console_logs = []
        page.on("console", lambda msg: console_logs.append({"type": msg.type, "text": msg.text}))
        
        page_errors = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        ws_frames = []   # list of {"dir": "send"|"recv", "data": str, "ts": float}
        def on_ws(ws_obj):
            ws_obj.on("framesent",    lambda payload: ws_frames.append({"dir":"send","data":payload,"ts":time.time()}))
            ws_obj.on("framereceived",lambda payload: ws_frames.append({"dir":"recv","data":payload,"ts":time.time()}))
        page.on("websocket", on_ws)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 1. HEALTH ENDPOINT
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("â•â•â• 1. Health endpoint â•â•â•")
        t0 = time.time()
        health_resp = await page.request.get("http://localhost:8000/health")
        health_ms = (time.time() - t0) * 1000
        health_json = await health_resp.json()
        report["health"] = {"status": health_resp.status, "body": health_json, "latency_ms": round(health_ms,1)}
        print(f"  HTTP {health_resp.status}  {health_ms:.0f}ms  model={health_json.get('model')}  chunks={health_json['corpus_stats']['total_chunks']}  sessions={health_json['active_sessions']}")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 2. INITIAL PAGE LOAD
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 2. Initial page load â•â•â•")
        t0 = time.time()
        resp = await page.goto("http://localhost:8000", wait_until="networkidle")
        load_ms = (time.time() - t0) * 1000
        report["page_load"] = {"status": resp.status, "latency_ms": round(load_ms,1)}
        print(f"  HTTP {resp.status}  load={load_ms:.0f}ms")

        # Wait for WS connection indicator
        try:
            await page.wait_for_function("() => document.getElementById('wsStatusText').innerText === 'Connected'", timeout=8000)
            ws_connected = True
        except:
            ws_connected = False
        report["ws_connected"] = ws_connected
        print(f"  WebSocket connected: {ws_connected}")

        await page.screenshot(path=ss("01_initial_load"))
        print(f"  Screenshot: 01_initial_load.png")

        # â”€â”€ Helper: type a query and press Enter â”€â”€
        async def send_query(query, label=""):
            """Type query into textarea (auto-sends tokens on space), then press Enter to end utterance."""
            ws_frames.clear()
            # Type word-by-word with trailing space so the input handler fires
            await page.fill("#speechInput", "")
            await page.focus("#speechInput")
            for word in query.split():
                await page.type("#speechInput", word + " ", delay=30)
                await asyncio.sleep(0.05)
            # Press Enter to trigger end_of_utterance
            await page.press("#speechInput", "Enter")

        async def wait_for_done(timeout=120):
            """Wait until the cursor disappears (cursor display=none means 'done' was received)."""
            try:
                await page.wait_for_function(
                    "() => document.getElementById('cursor').style.display === 'none'",
                    timeout=timeout*1000
                )
                return True
            except:
                return False

        async def get_ui_state():
            return await page.evaluate("""() => ({
                action: document.getElementById('actionBadge').innerText,
                turns: document.getElementById('turnCount').innerText,
                chunks: document.getElementById('chunkCount').innerText,
                answer: document.getElementById('answerContent').innerText,
                citations: document.getElementById('citations').innerText,
                insufficient: document.getElementById('alertInsufficient').style.display,
                latency: document.getElementById('latencyIndicator').innerText,
                constraints: document.getElementById('constraintsList').innerText,
                subqueries: document.getElementById('subQueriesList').innerText,
                cursorVisible: document.getElementById('cursor').style.display !== 'none',
            })""")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 3. BASIC SINGLE-QUERY TEST
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 3. Single-query: 'What is retrieval augmented generation?' â•â•â•")
        t_q = time.time()
        await send_query("What is retrieval augmented generation?")
        done = await wait_for_done(timeout=120)
        q1_ms = (time.time() - t_q) * 1000
        ui = await get_ui_state()
        await page.screenshot(path=ss("02_single_query"))
        report["single_query"] = {
            "done": done,
            "total_ms": round(q1_ms,1),
            "ui": ui,
            "ws_event_count": len(ws_frames),
        }
        # Extract the event types from recv frames
        recv_types = []
        for f in ws_frames:
            if f["dir"] == "recv":
                try:
                    d = json.loads(f["data"])
                    recv_types.append(d.get("type","?"))
                except: pass
        report["single_query"]["event_sequence"] = recv_types
        print(f"  done={done}  total={q1_ms:.0f}ms  action={ui['action']}  turns={ui['turns']}  chunks={ui['chunks']}  citations present={bool(ui['citations'].strip())}")
        print(f"  answer preview: {ui['answer'][:120]}...")
        print(f"  WS event sequence: {recv_types}")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 4. CONCISE QUERY TEST
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 4. Concise query: 'Who was Vaswani?' â•â•â•")
        await page.click("#btnReset")
        await asyncio.sleep(1)
        await send_query("Who was Vaswani?")
        done = await wait_for_done(timeout=120)
        ui = await get_ui_state()
        await page.screenshot(path=ss("03_concise_query"))
        report["concise_query"] = {"done": done, "ui": ui}
        print(f"  done={done}  action={ui['action']}  answer_len={len(ui['answer'])}  turns={ui['turns']}")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 5. WAIT BEHAVIOR
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 5. WAIT: 'um uh well so' â•â•â•")
        await page.click("#btnReset")
        await asyncio.sleep(1)
        await send_query("um uh well so")
        await asyncio.sleep(3)
        ui = await get_ui_state()
        await page.screenshot(path=ss("04_wait"))
        report["wait_test"] = {"ui": ui}
        print(f"  action={ui['action']}  turns={ui['turns']}  answer_len={len(ui['answer'])}")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 6. SUPPRESS BEHAVIOR
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 6. SUPPRESS: 'next slide please' â•â•â•")
        await page.click("#btnReset")
        await asyncio.sleep(1)
        await send_query("next slide please")
        await asyncio.sleep(3)
        ui = await get_ui_state()
        await page.screenshot(path=ss("05_suppress"))
        report["suppress_test"] = {"ui": ui}
        print(f"  action={ui['action']}  turns={ui['turns']}  answer_len={len(ui['answer'])}")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 7. MULTI-INTENT TEST
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 7. Multi-intent â•â•â•")
        await page.click("#btnReset")
        await asyncio.sleep(1)
        ws_frames.clear()
        await send_query("What is retrieval augmented generation and what are the main causes of climate change?")
        done = await wait_for_done(timeout=120)
        ui = await get_ui_state()
        await page.screenshot(path=ss("06_multi_intent"))
        recv_types_m = []
        for f in ws_frames:
            if f["dir"] == "recv":
                try: recv_types_m.append(json.loads(f["data"]).get("type","?"))
                except: pass
        report["multi_intent"] = {"done": done, "ui": ui, "events": recv_types_m}
        print(f"  done={done}  action={ui['action']}  turns={ui['turns']}  chunks={ui['chunks']}  subqueries={ui['subqueries'][:80]}")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 8. SESSION REFINEMENT TEST
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 8. Refinement: 'Only information after 2020' â•â•â•")
        ws_frames.clear()
        await send_query("Only information after 2020.")
        done = await wait_for_done(timeout=120)
        ui = await get_ui_state()
        await page.screenshot(path=ss("07_refinement"))
        recv_types_r = []
        for f in ws_frames:
            if f["dir"] == "recv":
                try: recv_types_r.append(json.loads(f["data"]).get("type","?"))
                except: pass
        report["refinement"] = {"done": done, "ui": ui, "events": recv_types_r}
        print(f"  done={done}  turns={ui['turns']}  constraints={ui['constraints']}  events={recv_types_r}")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 9. RESET TEST
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 9. Reset â•â•â•")
        await page.click("#btnReset")
        await asyncio.sleep(1)
        ui = await get_ui_state()
        await page.screenshot(path=ss("08_reset"))
        report["reset"] = {"ui": ui}
        print(f"  after reset: action={ui['action']}  turns={ui['turns']}  chunks={ui['chunks']}  answer_len={len(ui['answer'])}")
        
        # Quick follow-up query after reset
        await send_query("What is RAG?")
        done = await wait_for_done(timeout=120)
        ui2 = await get_ui_state()
        report["reset_followup"] = {"done": done, "ui": ui2}
        print(f"  follow-up after reset: done={done}  turns={ui2['turns']}  answer_len={len(ui2['answer'])}")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 10. RAPID-QUERY / RESPONSIVENESS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 10. Rapid sequential queries â•â•â•")
        await page.click("#btnReset")
        await asyncio.sleep(1)
        await send_query("What is CRISPR?")
        done1 = await wait_for_done(timeout=120)
        ui_a = await get_ui_state()
        await send_query("What is dark matter?")
        done2 = await wait_for_done(timeout=120)
        ui_b = await get_ui_state()
        await page.screenshot(path=ss("09_rapid"))
        report["rapid"] = {"q1_done": done1, "q2_done": done2, "q1_turns": ui_a['turns'], "q2_turns": ui_b['turns']}
        print(f"  q1 done={done1} turns={ui_a['turns']}  q2 done={done2} turns={ui_b['turns']}")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 11. ERROR HANDLING â€” skip (no safe mock)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        report["error_handling"] = "NOT TESTED â€” no safe UI-level error trigger exists"

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 12+13. Console & Network
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        report["console_logs"] = console_logs
        report["page_errors"] = page_errors

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # 15. VISUAL â€” narrow viewport
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        print("\nâ•â•â• 15. Narrow viewport test â•â•â•")
        await page.set_viewport_size({"width": 768, "height": 900})
        await asyncio.sleep(0.5)
        await page.screenshot(path=ss("10_narrow_viewport"))
        await page.set_viewport_size({"width": 1440, "height": 900})  # restore
        
        await browser.close()

    # â”€â”€ Write results â”€â”€
    out_path = os.path.join(ARTIFACT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    # â”€â”€ Summary â”€â”€
    print("\n" + "="*60)
    print("CONSOLE ERRORS:")
    errors = [c for c in console_logs if c["type"] == "error"]
    if errors:
        for e in errors: print(f"  {e['text'][:200]}")
    else:
        print("  None")
    print(f"\nPAGE ERRORS (uncaught): {len(page_errors)}")
    for e in page_errors: print(f"  {e[:200]}")

if __name__ == "__main__":
    asyncio.run(main())


