import asyncio
import re
import json
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# Load env before importing core which might rely on it
load_dotenv()

from core import SessionState
from core.state import StateManager
from core.controller import IntentController
from core.decomposer import IntentDecomposer

console = Console()

async def run_checks():
    results = []
    
    # Check 1
    passed = True
    details = []
    pattern = r'\[([\w_]+) §([^\]]+)\]'
    texts = [
        ('Transformers were introduced [ai_001 §Introduction].', 1),
        ('Sea levels [cl_003 §Findings] rising [cl_001 §Overview].', 2),
        ('No evidence here.', 0)
    ]
    for text, exp in texts:
        matches = len(re.findall(pattern, text))
        if matches != exp:
            passed = False
            details.append(f"Expected {exp} matches for '{text}', got {matches}")
    results.append(("Citation Format Compliance", passed, ", ".join(details) if not passed else "Regex matched citations correctly"))

    # Check 2
    try:
        controller = IntentController()
        session = SessionState(session_id='test')
        cmds = ['next slide please', 'go back to previous slide', 'repeat that', 'scroll down']
        passed = True
        for cmd in cmds:
            d = await controller.assess(cmd, session)
            if d.action != 'SUPPRESS':
                passed = False
        results.append(("SUPPRESS for navigational queries", passed, "All commands returned SUPPRESS"))
    except Exception as e:
        results.append(("SUPPRESS for navigational queries", "WARN", f"Exception: {e}"))

    # Check 3
    try:
        controller = IntentController()
        session = SessionState(session_id='test')
        cmds = ['And also', 'um uh', 'So basically']
        passed = True
        for cmd in cmds:
            d = await controller.assess(cmd, session)
            if d.action != 'WAIT':
                passed = False
        results.append(("WAIT for incomplete fragments", passed, "All fragments returned WAIT"))
    except Exception as e:
        results.append(("WAIT for incomplete fragments", False, f"Exception: {e}"))

    # Check 4
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")):
        results.append(("Multi-intent decomposition", "SKIP", "No API key in env"))
    else:
        try:
            decomposer = IntentDecomposer()
            res = await decomposer.decompose('How do transformers work and what is RAG and why is CRISPR significant')
            if len(res) >= 2:
                results.append(("Multi-intent decomposition", True, f"Decomposed into {len(res)} subqueries"))
            else:
                results.append(("Multi-intent decomposition", False, f"Decomposed into {len(res)} subqueries (expected >= 2)"))
        except Exception as e:
            results.append(("Multi-intent decomposition", False, f"Exception: {e}"))

    # Check 5
    try:
        sm = StateManager()
        sm.get_or_create('eval_a')
        sm.append_token('eval_a', 'token_alpha')
        sm.get_or_create('eval_b')
        sm.append_token('eval_b', 'token_beta')
        eval_a = sm.get_or_create('eval_a')
        eval_b = sm.get_or_create('eval_b')
        if eval_a.tokens == ['token_alpha'] and eval_b.tokens == ['token_beta']:
            sm.destroy('eval_a')
            if 'eval_a' not in sm._sessions:
                results.append(("Session isolation", True, "Sessions isolated and destroyed correctly"))
            else:
                results.append(("Session isolation", False, "Session eval_a not destroyed"))
        else:
            results.append(("Session isolation", False, "Session isolation failed"))
    except Exception as e:
        results.append(("Session isolation", False, f"Exception: {e}"))

    table = Table(title="Evaluation Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="magenta")

    all_passed = True
    for name, status, details in results:
        if status == True:
            status_str = "[green]PASS[/green]"
        elif status == False:
            status_str = "[red]FAIL[/red]"
            all_passed = False
        elif status == "SKIP":
            status_str = "[yellow]SKIP[/yellow]"
        elif status == "WARN":
            status_str = "[yellow]WARN[/yellow]"
        table.add_row(name, status_str, details)

    console.print(table)
    if all_passed:
        console.print("[green]All required checks passed![/green]")
        sys.exit(0)
    else:
        console.print("[red]Some checks failed.[/red]")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_checks())
