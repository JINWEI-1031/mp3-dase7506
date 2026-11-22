"""Bounded candidate expansion after the frozen CPU strategy reaches fallback."""
from pathlib import Path

from mp3_sql.sqlite_tool import validate_sql
from step4_agent import identity, passed
from step4_edits import Schema, simple
from step5_agent import Step5Agent, load_policy
from step5x_edits import aggregate_edits, foreign_key_edits

SOURCES = {"fk": (foreign_key_edits,), "aggregate": (aggregate_edits,),
           "combined": (foreign_key_edits, aggregate_edits)}


class ExpansionAgent(Step5Agent):
    def __init__(self, base, policy, mode):
        super().__init__(base, policy)
        self.generators = SOURCES[mode]
        self.pending = []
        self.generated = False

    def act(self, observation):
        choice = super().act(observation)
        if choice["tool"] != "submit":
            return choice
        history = observation["history"]
        checks = {e["sql"]: e["observation"] for e in history if e["tool"] == "check"}
        if any(passed(value) for value in checks.values()):
            return choice
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        remaining = observation["remaining"]
        if tables is None:
            return {"tool": "schema"} if remaining >= 2 else choice
        if not self.generated:
            self.generated = True
            self.generate(observation, checks, Schema(tables))
        if remaining and self.pending:
            return {"tool": "check", "sql": self.pending.pop(0)}
        return choice

    def generate(self, observation, checks, schema):
        question = observation["task"]["question"]
        arity = len(observation["example"]["columns"])
        parents = list(checks.items())[:2]  # draft and first checked hint
        for generate in self.generators:
            count = 0
            for parent, feedback in parents:
                tree = simple(parent)
                if tree is None or tree.args.get("with"):
                    continue
                for location, sql, evidence in generate(tree, schema, feedback, question, arity):
                    key = identity(sql)
                    if key in self.audit["candidates"]:
                        continue
                    try:
                        validate_sql(sql)
                    except ValueError:
                        continue
                    self.pending.append(self.add(sql, generate.__name__, location, evidence, parent))
                    count += 1
                    if count == 2:
                        break
                if count == 2:
                    break


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    base, policy = load_policy(path.with_name("step5-cpu_s0_d0.json"))
    return ExpansionAgent(base, policy, path.read_text(encoding="utf-8").strip())

