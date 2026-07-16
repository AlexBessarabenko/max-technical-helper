from src.data_gen.ad_gen import generate_employees


def test_generate_employees_structure_and_determinism():
    emps = generate_employees(n=120, seed=42)
    assert len(emps) == 120
    e = emps[0]
    for key in ("id", "full_name", "department", "position", "phone", "email", "manager_id", "samaccountname"):
        assert key in e
    assert generate_employees(n=120, seed=42) == emps  # детерминизм по seed


def test_manager_graph_acyclic_and_rooted():
    emps = generate_employees(n=120, seed=42)
    by_id = {e["id"]: e for e in emps}
    assert sum(1 for e in emps if e["manager_id"] is None) == 1  # один CEO
    for e in emps:
        seen, cur = set(), e
        while cur["manager_id"] is not None:
            assert cur["manager_id"] not in seen, "цикл в графе менеджеров"
            seen.add(cur["id"])
            cur = by_id[cur["manager_id"]]
