"""Admission is atomic; failures and unknown billing retain their full liability."""

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from email.message import Message
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from evaluation.budget import BudgetLedger, BudgetSpec, ProviderBinding
from evaluation.client import ActionClient

from nof1_causal_lab.actions.contracts.bodies import ContinueFitTarget, FitInput
from nof1_causal_lab.actions.contracts.requests import FitRequest, call_identity
from nof1_causal_lab.artifacts.sampling import FitSettingsSpec, InferenceHardware
from tests.support.git import git_oid

pytestmark = pytest.mark.contract


def _ledger(tmp_path, limit="0.06"):
    return BudgetLedger(
        BudgetSpec(
            directory=tmp_path / "ledger",
            hold_file=tmp_path / "HOLD",
            rates={InferenceHardware.L40S: Decimal("0.001")},
            max_usd=Decimal(limit),
            max_concurrent=2,
            termination_seconds=10,
        )
    )


def _settings():
    return FitSettingsSpec(hardware="L40S", time_limit_seconds=10)


def test_atomic_reservations_hold_failures_and_reconcile_only_stopped_billed_apps(tmp_path):
    ledger = _ledger(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tuple(pool.map(lambda n: ledger.reserve("api", f"w{n}", f"call{n}", _settings()), range(2)))
    records = ledger.records()
    assert len(records) == 2
    assert {r.reserved_usd for r in records} == {Decimal("0.03")}
    for r in records:
        ledger.finish(r.endpoint, r.workspace, r.call_id, "failed")
    with pytest.raises(RuntimeError, match="budget"):
        ledger.reserve("api", "new", "new", _settings())
    ledger.policy.hold_file.touch()
    with pytest.raises(RuntimeError, match="reviewer"):
        ledger.reserve("api", "held", "held", _settings())
    ledger.policy.hold_file.unlink()
    bindings = {
        r.identity: ProviderBinding(app_ids=(f"ap-{i}",), evidence=f"provider trace {i}")
        for i, r in enumerate(records)
    }
    bills, apps = tmp_path / "billing.json", tmp_path / "apps.json"
    apps.write_text(json.dumps([{"app_id": f"ap-{i}", "state": "stopped"} for i in range(2)]))
    bills.write_text(
        json.dumps(
            [{"object_id": "ap-0", "cost": "0.005"}, {"object_id": "unrelated", "cost": "100"}]
        )
    )
    ledger.reconcile(bills, apps, bindings)
    assert json.loads(ledger.summary())["accounted_usd"] == "0.035"
    with pytest.raises(RuntimeError, match="budget"):
        ledger.reserve("api", "new", "new", _settings())
    bills.write_text(json.dumps([{"object_id": f"ap-{i}", "cost": "0.005"} for i in range(2)]))
    ledger.reconcile(bills, apps, bindings)
    ledger.reserve("api", "new", "new", _settings())
    assert json.loads(ledger.summary())["accounted_usd"] == "0.040"
    assert len(tuple(ledger.policy.directory.glob("reconciliation-*.json"))) == 2
    assert len(ledger.records()) == 3


def test_client_reserves_after_cache_lookup_before_post_and_keeps_ambiguous_response(
    tmp_path, monkeypatch
):
    ledger = _ledger(tmp_path)
    request = FitRequest(
        input=FitInput(target=ContinueFitTarget(fit_ref=git_oid(1)), settings=_settings())
    )
    client = ActionClient("http://isolated.invalid", "arm", tmp_path / "attempt", budget=ledger)
    methods = []

    def read(request):
        method = request.get_method()
        methods.append(method)
        if method == "GET":
            assert ledger.records() == ()
            raise HTTPError("lookup", 404, "not saved", Message(), None)
        assert len(ledger.records()) == 1
        raise HTTPError("submission", 500, "response lost", Message(), None)

    monkeypatch.setattr(client, "_read", read)
    with pytest.raises(HTTPError, match="response lost"):
        client.call(request)
    (record,) = ledger.records()
    assert record.call_id == call_identity(request)
    assert record.terminal is None
    assert methods == ["GET", "POST"]
    with pytest.raises(RuntimeError, match="already reserved"):
        ledger.reserve(client.endpoint, client.workspace, record.call_id, _settings())
    from nof1_causal_lab.actions.contracts.polling import FailedPoll

    recovered = FailedPoll(
        call_id=call_identity(request), action="fit", commit_id=git_oid(2), messages=()
    )
    monkeypatch.setattr(client, "_read", Mock(return_value=recovered))
    with pytest.raises(ValueError, match="fit failed"):
        client.call(request)
    (record,) = ledger.records()
    assert record.terminal.status == "failed"
    assert record.reserved_usd == Decimal("0.03")


def test_held_or_unbudgeted_fits_stop_before_paid_submission(tmp_path, monkeypatch):
    from evaluation.run import _fit

    ledger = _ledger(tmp_path)
    client = ActionClient("http://isolated.invalid", "arm", tmp_path / "attempt", budget=ledger)
    ledger.policy.hold_file.touch()
    request = FitRequest(
        input=FitInput(target=ContinueFitTarget(fit_ref=git_oid(1)), settings=_settings())
    )
    read = Mock(side_effect=HTTPError("lookup", 404, "not saved", Message(), None))
    monkeypatch.setattr(client, "_read", read)
    with pytest.raises(RuntimeError, match="reviewer"):
        client.call(request)
    assert read.call_count == 1
    assert ledger.records() == ()
    with pytest.raises(ValueError, match="evaluation budget"):
        _fit(
            ActionClient("http://isolated.invalid", "unbounded", tmp_path / "unbounded"),
            Mock(),
            _settings(),
            0,
        )


@pytest.mark.parametrize("reused", [False, True])
def test_terminal_failure_keeps_its_debit_and_cache_reuse_needs_no_reservation(
    tmp_path, monkeypatch, reused
):
    from nof1_causal_lab.actions.contracts.polling import FailedPoll

    ledger = _ledger(tmp_path)
    request = FitRequest(
        input=FitInput(target=ContinueFitTarget(fit_ref=git_oid(1)), settings=_settings())
    )
    result = FailedPoll(
        call_id=call_identity(request), action="fit", commit_id=git_oid(2), messages=()
    )
    client = ActionClient("http://isolated.invalid", "arm", tmp_path / "attempt", budget=ledger)
    reads = Mock(
        side_effect=[
            result if reused else HTTPError("lookup", 404, "not saved", Message(), None),
            result,
        ]
    )
    monkeypatch.setattr(client, "_read", reads)
    with pytest.raises(ValueError, match="fit failed"):
        client.call(request)
    assert client.reused[result.call_id] is reused
    if reused:
        assert ledger.records() == ()
    else:
        (record,) = ledger.records()
        assert record.terminal.status == "failed"
        assert json.loads(ledger.summary())["accounted_usd"] == "0.030"
