"""Use the public check to select between the draft and fixed hint candidates."""


class RepairAgent:
    def act(self, observation):
        draft = observation["issue_sql"]
        history = observation["history"]
        remaining = observation["remaining"]

        # A passing draft wins immediately. Otherwise prefer the first passing
        # hint in the tool's order. The draft is also the last-resort fallback.
        checks = {event["sql"]: event["observation"] for event in history
                  if event["tool"] == "check"}
        for sql, feedback in checks.items():
            if feedback["ok"] and feedback["matches_example"]:
                return {"tool": "submit", "sql": sql}

        if remaining:
            if draft not in checks:
                return {"tool": "check", "sql": draft}

            hint = next((event for event in history if event["tool"] == "repair_hint"), None)
            if hint is None and remaining >= 2:
                return {"tool": "repair_hint"}
            if hint is not None:
                for sql in hint["observation"] or []:
                    if (isinstance(sql, str) and sql.strip()
                            and len(sql.encode("utf-8")) <= 16_000 and sql not in checks):
                        return {"tool": "check", "sql": sql}

        return {"tool": "submit", "sql": draft}


def create_agent(checkpoint=None):
    if checkpoint is not None:
        from step3_agent import RuleAgent, load_config
        return RuleAgent(load_config(checkpoint))
    return RepairAgent()

