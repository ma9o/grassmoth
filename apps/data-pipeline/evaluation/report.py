"""One sortable report keeps reserved budgets, measured time and invoiced cost distinct."""

from __future__ import annotations

import hashlib
import html
from typing import TYPE_CHECKING

from nof1_causal_lab.public.contracts import Value

if TYPE_CHECKING:
    from pathlib import Path

    from evaluation.budget import BudgetLedger
    from evaluation.declarations import Experiment
    from nof1_causal_lab.public.contracts import JsonObject


class BillingReport(Value):
    """Source bytes of a separately supplied billing report, never estimated from fit time."""

    filename: str
    sha256: str
    text: str

    @classmethod
    def read(cls, path: Path) -> BillingReport:
        """Retain the original output and its digest for cost reconciliation."""
        content = path.read_bytes()
        return cls(
            filename=path.name, sha256=hashlib.sha256(content).hexdigest(), text=content.decode()
        )


def write_report(
    path: Path,
    experiment: Experiment,
    rows: list[JsonObject],
    billing: BillingReport | None,
    budget: BudgetLedger | None = None,
) -> None:
    """Render saved results and separately supplied Modal billing-report allocations."""
    keys = tuple(dict.fromkeys(key for row in rows for key in row))
    headers = "".join(
        f'<th><button data-column="{i}">{html.escape(key)}</button></th>'
        for i, key in enumerate(keys)
    )
    cells = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in keys) + "</tr>"
        for row in rows
    )
    reservation = html.escape(budget.summary() if budget is not None else "No paid admissions")
    payload = html.escape(experiment.model_dump_json(indent=2))
    invoices = html.escape(
        billing.model_dump_json(indent=2) if billing is not None else "Not supplied"
    )
    path.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{html.escape(experiment.name)}</title><style>
body{{font:14px system-ui;margin:24px;color:#172033}} table{{border-collapse:collapse;display:block;overflow:auto}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left;white-space:pre-wrap}}th{{position:sticky;top:0;background:#eef2f5}}
button{{font:inherit;border:0;background:none;cursor:pointer}}pre{{white-space:pre-wrap;max-width:100%}}
</style><h1>{html.escape(experiment.name)}</h1>
<details open><summary>Budget ledger</summary><pre>{reservation}</pre></details>
<p>Measured compute seconds are retained fit evidence,
not billed GPU time. Reused calls do not represent a new timing measurement.</p>
<p>Compilation time is the union of recorded compile intervals; overlapping work is counted once.
Adapted tuning is an outcome. Summary equivalence covers marginal means and standard deviations,
not equality of the joint posterior. Truth recovery remains separate from same-data reference comparison.</p>
<table id="results"><thead><tr>{headers}</tr></thead><tbody>{cells}</tbody></table>
<details><summary>Declaration and predeclared tolerances</summary><pre>{payload}</pre></details>
<details><summary>Billed cost — allocations from modal billing report</summary><pre>{invoices}</pre></details>
<p>Raw action envelopes and case/attempt manifests are saved beside this report. Missing billing is unknown, not zero.</p>
<script>document.querySelectorAll('button[data-column]').forEach(b=>b.onclick=()=>{{
const body=document.querySelector('#results tbody'),i=Number(b.dataset.column),direction=b.dataset.order==='asc'?-1:1;
b.dataset.order=direction===1?'asc':'desc';const rows=Array.from(body.rows);
rows.sort((a,c)=>{{let x=a.cells[i].textContent,y=c.cells[i].textContent;
return direction*(x.trim()!==''&&y.trim()!==''&&Number.isFinite(Number(x))&&Number.isFinite(Number(y))?Number(x)-Number(y):x.localeCompare(y));}});
rows.forEach(row=>body.appendChild(row));}});</script></html>""")
