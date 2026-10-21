"""Benchmark-owned admission and billing reconciliation; no production compute hooks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from evaluation.report import BillingReport
from nof1_causal_lab.public.contracts import FitSettingsSpec, InferenceHardware, Value

type Usd = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
type Rate = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]


class BudgetSpec(Value):
    """Admission policy for one ledger, shared across stages and scheduler processes."""

    directory: Path
    hold_file: Path
    rates: Mapping[InferenceHardware, Rate]
    max_usd: Rate
    max_concurrent: int = Field(default=4, ge=1)
    termination_seconds: int = Field(default=60, ge=2)
    extra_usd_per_fit: Usd = Decimal(0)


class TerminalRecord(Value):
    """A received terminal response; a lost acknowledgement remains unresolved."""

    status: Literal["success", "failed"]
    observed_at: float


class AllocationRecord(Value):
    """The full liability is retained before the public fit POST, including failures."""

    identity: str
    endpoint: str
    workspace: str
    call_id: str
    created_at: float
    settings: FitSettingsSpec
    reserved_usd: Usd
    terminal: TerminalRecord | None = None


class ProviderBinding(Value):
    """Reviewed provider IDs and the evidence connecting them to one fit admission."""

    app_ids: tuple[str, ...] = Field(min_length=1)
    evidence: str = Field(min_length=1)


class ProviderCharge(BaseModel):
    """Only the fields consumed from the provider's original billing export."""

    object_id: str
    cost: Usd


class ProviderApp(BaseModel):
    """The provider's observed lifecycle state, retained with the original export."""

    app_id: str
    state: str


class Settlement(Value):
    """An invoice releases a reservation only when all bound apps are stopped and billed."""

    binding: ProviderBinding
    reported_usd: Usd
    stopped: bool
    fully_billed: bool

    def accounted_usd(self, allocation: AllocationRecord) -> Decimal:
        """Unknown or live billing never becomes a zero-dollar refund."""
        if self.stopped and self.fully_billed:
            return self.reported_usd
        return max(allocation.reserved_usd, self.reported_usd)


class Reconciliation(Value):
    """Original provider exports and their explicit, auditable assignment to this ledger."""

    observed_at: float
    billing: BillingReport
    apps: BillingReport
    settlements: Mapping[str, Settlement]


@dataclass(frozen=True)
class BudgetLedger:
    """File-locked admission and immutable snapshots, outside the scientific framework."""

    policy: BudgetSpec

    @classmethod
    def read(cls, path: Path) -> BudgetLedger:
        """Load an explicitly supplied policy; importing evaluation performs no I/O."""
        return cls(BudgetSpec.model_validate_json(path.read_text()))

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Serialize admission, completion and reconciliation across schedulers."""
        self.policy.directory.mkdir(parents=True, exist_ok=True)
        with (self.policy.directory / "admission.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield

    def records(self) -> tuple[AllocationRecord, ...]:
        """Read retained admissions, including failed and ambiguously acknowledged calls."""
        return tuple(
            AllocationRecord.model_validate_json(path.read_text())
            for path in sorted((self.policy.directory / "allocations").glob("*.json"))
        )

    def reconciliation(self) -> Reconciliation | None:
        """An absent reconciliation leaves every reservation outstanding."""
        path = self.policy.directory / "reconciliation.json"
        return Reconciliation.model_validate_json(path.read_text()) if path.exists() else None

    def _save(self, record: AllocationRecord) -> None:
        directory = self.policy.directory / "allocations"
        directory.mkdir(exist_ok=True)
        path = directory / (record.identity + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(record.model_dump_json(indent=2))
        temporary.replace(path)

    @staticmethod
    def identity(endpoint: str, workspace: str, call_id: str) -> str:
        """The public call in its workspace is the reservation identity."""
        return hashlib.sha256(json.dumps((endpoint, workspace, call_id)).encode()).hexdigest()

    def reserve(
        self, endpoint: str, workspace: str, call_id: str, settings: FitSettingsSpec
    ) -> None:
        """Reserve the bounded startup, execution and termination cost before submission."""
        hardware, limit = settings.hardware, settings.time_limit_seconds
        if hardware is None or hardware == InferenceHardware.LOCAL or limit is None:
            raise ValueError("Paid fits require explicit hardware and time_limit_seconds")
        rate = self.policy.rates[hardware]
        # Modal receives the same bound for startup and execution. Neither
        # phase may be priced as a typical duration instead of its ceiling.
        reservation = rate * (2 * limit + self.policy.termination_seconds)
        reservation += self.policy.extra_usd_per_fit
        identity = self.identity(endpoint, workspace, call_id)
        with self.locked():
            if self.policy.hold_file.exists():
                raise RuntimeError("Paid submission held by the campaign reviewer")
            records, reconciliation = self.records(), self.reconciliation()
            settlements = {} if reconciliation is None else reconciliation.settlements
            if any(record.identity == identity for record in records):
                raise RuntimeError(
                    "Fit already reserved; recover its saved call before resubmitting"
                )
            accounted = sum(
                (
                    settlements[r.identity].accounted_usd(r)
                    if r.identity in settlements
                    else r.reserved_usd
                    for r in records
                ),
                Decimal(0),
            )
            if accounted + reservation > self.policy.max_usd:
                raise RuntimeError("Fit would exceed the campaign budget")
            active = sum(
                r.terminal is None
                and not (r.identity in settlements and settlements[r.identity].stopped)
                for r in records
            )
            if active >= self.policy.max_concurrent:
                raise RuntimeError("Fit would exceed campaign concurrency")
            self._save(
                AllocationRecord(
                    identity=identity,
                    endpoint=endpoint,
                    workspace=workspace,
                    call_id=call_id,
                    created_at=time.time(),
                    settings=settings,
                    reserved_usd=reservation,
                )
            )

    def finish(
        self, endpoint: str, workspace: str, call_id: str, status: Literal["success", "failed"]
    ) -> None:
        """Finish owned admissions, including recovered calls; unowned cache hits cost nothing."""
        identity = self.identity(endpoint, workspace, call_id)
        with self.locked():
            record = next((r for r in self.records() if r.identity == identity), None)
            if record is not None:
                self._save(
                    record.revised(terminal=TerminalRecord(status=status, observed_at=time.time()))
                )

    def reconcile(
        self, billing_path: Path, apps_path: Path, bindings: Mapping[str, ProviderBinding]
    ) -> Reconciliation:
        """Import original CLI exports; never infer a provider ID from timing or app name."""
        billing, apps = BillingReport.read(billing_path), BillingReport.read(apps_path)
        charges = TypeAdapter(tuple[ProviderCharge, ...]).validate_json(billing.text)
        states = {
            app.app_id: app.state
            for app in TypeAdapter(tuple[ProviderApp, ...]).validate_json(apps.text)
        }
        app_ids = tuple(app for binding in bindings.values() for app in binding.app_ids)
        if len(set(app_ids)) != len(app_ids):
            raise ValueError("A provider app can settle only one reservation")
        with self.locked():
            identities = {r.identity for r in self.records()}
            if set(bindings) - identities:
                raise ValueError("Provider bindings name an unknown allocation")
            settlements = {
                identity: Settlement(
                    binding=binding,
                    reported_usd=sum(
                        (row.cost for row in charges if row.object_id in binding.app_ids),
                        Decimal(0),
                    ),
                    stopped=all(states.get(app) == "stopped" for app in binding.app_ids),
                    fully_billed=all(
                        any(row.object_id == app for row in charges) for app in binding.app_ids
                    ),
                )
                for identity, binding in bindings.items()
            }
            result = Reconciliation(
                observed_at=time.time(), billing=billing, apps=apps, settlements=settlements
            )
            snapshot = self.policy.directory / ("reconciliation-" + str(time.time_ns()) + ".json")
            snapshot.write_text(result.model_dump_json(indent=2))
            temporary = self.policy.directory / "reconciliation.tmp"
            temporary.write_text(result.model_dump_json(indent=2))
            temporary.replace(self.policy.directory / "reconciliation.json")
            return result

    def summary(self) -> str:
        """Render reported and outstanding liabilities without estimating an invoice."""
        with self.locked():
            records, reconciliation = self.records(), self.reconciliation()
            settlements = {} if reconciliation is None else reconciliation.settlements
            return json.dumps(
                {
                    "directory": str(self.policy.directory),
                    "admissions": len(records),
                    "last_response_observed_at": max(
                        (r.terminal.observed_at for r in records if r.terminal is not None),
                        default=None,
                    ),
                    "billing_observed_at": None
                    if reconciliation is None
                    else reconciliation.observed_at,
                    "historically_reserved_usd": str(
                        sum((r.reserved_usd for r in records), Decimal(0))
                    ),
                    "reported_usd": str(
                        sum((s.reported_usd for s in settlements.values()), Decimal(0))
                    ),
                    "accounted_usd": str(
                        sum(
                            (
                                settlements[r.identity].accounted_usd(r)
                                if r.identity in settlements
                                else r.reserved_usd
                                for r in records
                            ),
                            Decimal(0),
                        )
                    ),
                    "limit_usd": str(self.policy.max_usd),
                },
                indent=2,
            )


def main() -> None:
    """Reconcile reviewed app bindings against saved modal billing report/app list exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--billing-report", type=Path, required=True)
    parser.add_argument("--apps", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    args = parser.parse_args()
    ledger = BudgetLedger.read(args.policy)
    bindings = TypeAdapter(dict[str, ProviderBinding]).validate_json(args.bindings.read_text())
    ledger.reconcile(args.billing_report, args.apps, bindings)
    print(ledger.summary())


if __name__ == "__main__":
    main()
