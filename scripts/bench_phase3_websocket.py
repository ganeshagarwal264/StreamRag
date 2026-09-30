"""Phase 3: Full RAG pipeline via WebSocket. Server must be running."""
import asyncio, time, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

async def main():
    import websockets
    
    print("==================================================")
    print("PHASE 3: FULL RAG WEBSOCKET TIMING (5 runs)")
    print("==================================================")
    
    query = "What is retrieval augmented generation?"
    results = []
    
    for i in range(5):
        session_id = f"bench_{i}_{int(time.time())}"
        uri = f"ws://localhost:8000/ws/{session_id}"
        
        try:
            async with websockets.connect(uri) as ws:
                # Send tokens
                for word in query.split():
                    await ws.send(json.dumps({"type": "token", "content": word + " "}))
                    await asyncio.sleep(0.02)
                
                # End utterance
                T0 = time.time()
                await ws.send(json.dumps({"type": "end_of_utterance"}))
                
                T1 = None  # status
                T2 = None  # retrieved
                T3 = None  # first token
                T5 = None  # done
                chunks = 0
                tokens = 0
                citations = 0
                answer = ""
                
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=120)
                        data = json.loads(msg)
                        t = time.time()
                        
                        if data['type'] == 'status':
                            T1 = t
                            print(f"  Run {i+1}: status={data['action']} ({(T1-T0)*1000:.0f}ms)")
                            if data['action'] != 'RETRIEVE':
                                break
                        elif data['type'] == 'decomposed':
                            print(f"  Run {i+1}: decomposed={len(data['sub_queries'])} queries ({(t-T0)*1000:.0f}ms)")
                        elif data['type'] == 'retrieved':
                            T2 = t
                            chunks = data['chunk_count']
                            print(f"  Run {i+1}: retrieved={chunks} chunks ({(T2-T0)*1000:.0f}ms)")
                        elif data['type'] == 'token':
                            if T3 is None:
                                T3 = t
                            tokens += 1
                            answer += data.get('content', '')
                        elif data['type'] == 'citation':
                            citations += 1
                        elif data['type'] == 'done':
                            T5 = t
                            break
                        elif data['type'] == 'insufficient_evidence':
                            pass
                        elif data['type'] == 'error':
                            print(f"  Run {i+1}: ERROR: {data.get('message','')}")
                            break
                    except asyncio.TimeoutError:
                        print(f"  Run {i+1}: TIMEOUT after 120s")
                        break
                
                if T1 and T2 and T3 and T5:
                    controller_ms = (T1 - T0) * 1000
                    retrieval_ms = (T2 - T1) * 1000
                    ttft_after_retrieval = (T3 - T2) * 1000
                    total_ttft = (T3 - T0) * 1000
                    generation_ms = (T5 - T3) * 1000
                    total_ms = (T5 - T0) * 1000
                    gen_tok_sec = tokens / ((T5 - T3)) if (T5 - T3) > 0 else 0
                    
                    r = {
                        "run": i+1,
                        "controller_ms": round(controller_ms, 1),
                        "retrieval_ms": round(retrieval_ms, 1),
                        "ttft_after_retrieval_ms": round(ttft_after_retrieval, 1),
                        "total_ttft_ms": round(total_ttft, 1),
                        "generation_ms": round(generation_ms, 1),
                        "total_ms": round(total_ms, 1),
                        "tokens": tokens,
                        "citations": citations,
                        "chunks": chunks,
                        "tok_sec": round(gen_tok_sec, 1),
                    }
                    results.append(r)
                    print(f"  Run {i+1} COMPLETE: Controller={controller_ms:.0f}ms  Retrieval={retrieval_ms:.0f}ms  TTFT(synth)={ttft_after_retrieval:.0f}ms  Gen={generation_ms:.0f}ms  Total={total_ms:.0f}ms  Tok/s={gen_tok_sec:.1f}  Citations={citations}")
                
                # Reset
                await ws.send(json.dumps({"type": "reset"}))
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"  Run {i+1} FAILED: {e}")
    
    if results:
        print("\n==================================================")
        print("SUMMARY")
        print("==================================================")
        for key in ["controller_ms", "retrieval_ms", "ttft_after_retrieval_ms", "total_ttft_ms", "generation_ms", "total_ms", "tok_sec"]:
            vals = [r[key] for r in results]
            mean = sum(vals) / len(vals)
            vals_sorted = sorted(vals)
            median = vals_sorted[len(vals_sorted)//2]
            print(f"  {key:30s}: min={min(vals):8.1f}  max={max(vals):8.1f}  mean={mean:8.1f}  median={median:8.1f}")

asyncio.run(main())
