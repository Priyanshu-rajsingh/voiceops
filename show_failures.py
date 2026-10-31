import csv
from pathlib import Path

def rows(name):
    p = Path("eval_out") / name
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []

out = []
for r in rows("conversations.csv"):
    if r["pass"] != "True":
        out.append(f"CONV {r['conversation_id']} t{r['turn']} [{r['scenario']}] expected={r['expected']} got={r['decision']} tools_ok={r['tools_ok']}\n"
                   f"  missing: {r['missing']} | forbidden: {r['forbidden']}\n"
                   f"  tools expected: {r['tools_expected']} | called: {r['tools_called']}\n"
                   f"  customer: {r['customer'][:120]}\n  reply: {r['reply'][:260]}\n")
for r in rows("step3_questions.csv"):
    if r["pass"] != "True":
        out.append(f"Q {r['id']} [{r['intent']}] expected={r['expected']} got={r['decision']}\n"
                   f"  missing: {r['missing']} | forbidden: {r['forbidden']}\n"
                   f"  chunks: {r['chunks_used']}\n  reply: {r['reply'][:260]}\n")
Path("eval_out/failures.txt").write_text("\n".join(out), encoding="utf-8")
print(len(out), "failing rows written to eval_out/failures.txt")