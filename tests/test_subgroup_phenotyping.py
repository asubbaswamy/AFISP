import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from sirus import Condition, Rule

from afisp import SubgroupPhenotyper


def _model(rules):
    """A minimal stand-in for a fitted SIRUS model (only .rules_ is used)."""
    return SimpleNamespace(rules_=rules)


# --------------------------------------------------------------------------- #
# Unit tests: rule negation / extraction (no fitting)                          #
# --------------------------------------------------------------------------- #

def test_negate_simple_rule():
    sp = SubgroupPhenotyper()
    assert sp._negate_simple_rule("x1 >= 1") == "x1 < 1"
    assert sp._negate_simple_rule("x1 < 0.5") == "x1 >= 0.5"


def test_extract_keeps_rule_pointing_at_worst_subset():
    # output_in > output_out -> the rule already selects the worst subset
    sp = SubgroupPhenotyper()
    r = Rule([Condition(0, "x1", ">=", 0.6)], 0.9, 0.8, 0.2, 100, 400)
    assert sp._extract_sirus_rules(_model([r])) == ["x1 >= 0.6"]


def test_extract_negates_simple_rule():
    # output_in < output_out -> negate the (single) condition
    sp = SubgroupPhenotyper()
    r = Rule([Condition(0, "x1", ">=", 0.6)], 0.9, 0.2, 0.8, 100, 400)
    assert sp._extract_sirus_rules(_model([r])) == ["x1 < 0.6"]


def test_extract_negates_compound_rule_per_conjunct():
    # De Morgan: ~(x1 < 0.6 & x2 >= 30) -> two separate negated rules
    sp = SubgroupPhenotyper()
    r = Rule([Condition(0, "x1", "<", 0.6), Condition(1, "x2", ">=", 30)],
             0.9, 0.2, 0.8, 100, 400)
    assert sp._extract_sirus_rules(_model([r])) == ["x1 >= 0.6", "x2 < 30"]


def test_extract_keeps_compound_rule_as_is():
    sp = SubgroupPhenotyper()
    r = Rule([Condition(0, "x1", ">=", 0.6), Condition(1, "x2", "<", 30)],
             0.9, 0.8, 0.2, 100, 400)
    assert sp._extract_sirus_rules(_model([r])) == ["x1 >= 0.6 & x2 < 30"]


def test_extract_ordered_dedup():
    sp = SubgroupPhenotyper()
    r1 = Rule([Condition(0, "x1", ">=", 0.6)], 0.9, 0.8, 0.2, 100, 400)
    r2 = Rule([Condition(0, "x1", ">=", 0.6)], 0.7, 0.8, 0.2, 100, 400)
    assert sp._extract_sirus_rules(_model([r1, r2])) == ["x1 >= 0.6"]


def test_extract_empty_model_returns_empty_list():
    sp = SubgroupPhenotyper()
    assert sp._extract_sirus_rules(_model([])) == []


def test_extract_rejects_categorical_condition():
    sp = SubgroupPhenotyper()
    r = Rule([Condition(0, "color", "in", ("red", "blue"))], 0.9, 0.8, 0.2, 100, 400)
    with pytest.raises(ValueError):
        sp._extract_sirus_rules(_model([r]))


# --------------------------------------------------------------------------- #
# End-to-end tests (fit against the pure-Python sirus package, no R)           #
# --------------------------------------------------------------------------- #

def test_sirus_end_to_end_recovers_planted_feature(planted_data):
    sp = SubgroupPhenotyper()
    rules = sp.fit(planted_data["X"], planted_data["subset_labels"],
                   planted_data["test_loss"], method="SIRUS", random_state=0)
    assert isinstance(rules, list)
    assert any("x1" in r for r in rules)


def test_sirus_invokes_no_subprocess(planted_data, monkeypatch):
    """Regression guard: the SIRUS path must not shell out to R."""
    called = []

    def boom(*args, **kwargs):
        called.append(1)
        raise RuntimeError("subprocess should not be used by the SIRUS path")

    for name in ("call", "run", "Popen", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, boom)

    sp = SubgroupPhenotyper()
    sp.fit(planted_data["X"], planted_data["subset_labels"],
           planted_data["test_loss"], method="SIRUS", random_state=0)
    assert called == []


def test_sirus_empty_rule_set_is_graceful(planted_data):
    # a near-1 threshold selects (essentially) no rules; must not crash
    sp = SubgroupPhenotyper()
    rules = sp.fit(planted_data["X"], planted_data["subset_labels"],
                   planted_data["test_loss"], method="SIRUS", p0=0.999,
                   random_state=0)
    assert isinstance(rules, list)


def test_non_numeric_column_raises(planted_data):
    X = planted_data["X"].copy()
    X["cat"] = "a"
    sp = SubgroupPhenotyper()
    with pytest.raises(ValueError):
        sp.fit(X, planted_data["subset_labels"], planted_data["test_loss"],
               method="SIRUS", random_state=0)


def test_single_class_labels_raise(planted_data):
    sp = SubgroupPhenotyper()
    labels = np.zeros(len(planted_data["subset_labels"]), dtype=int)
    with pytest.raises(ValueError):
        sp.fit(planted_data["X"], labels, planted_data["test_loss"],
               method="SIRUS", random_state=0)


def test_p0_out_of_range_raises(planted_data):
    sp = SubgroupPhenotyper()
    with pytest.raises(ValueError):
        sp.fit(planted_data["X"], planted_data["subset_labels"],
               planted_data["test_loss"], method="SIRUS", p0=2.0)


def test_decision_list_method_runs(planted_data):
    sp = SubgroupPhenotyper()
    rules = sp.fit(planted_data["X"], planted_data["subset_labels"],
                   planted_data["test_loss"], method="DecisionList")
    assert isinstance(rules, list)


def test_invalid_method_raises(planted_data):
    sp = SubgroupPhenotyper()
    with pytest.raises(RuntimeError):
        sp.fit(planted_data["X"], planted_data["subset_labels"],
               planted_data["test_loss"], method="NotAMethod")


def test_generate_subgroup_table(planted_data):
    sp = SubgroupPhenotyper()
    sp.fit(planted_data["X"], planted_data["subset_labels"],
           planted_data["test_loss"], method="SIRUS", random_state=0)
    table = sp.generate_subgroup_table(planted_data["y_test"], planted_data["test_preds"])
    assert list(table.columns) == ["Phenotype", "Performance", "N", "Lower", "Upper"]
