"""Runtime profiling helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hw.timing import ProfileCounters


@dataclass(frozen=True)
class LaunchRecord:
    """Profile snapshot for one kernel launch."""

    cycles: int
    profile: ProfileCounters | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"cycles": self.cycles}
        if self.profile is not None:
            result["profile"] = self.profile.as_dict()
        return result


@dataclass
class RuntimeProfiler:
    """Collect launch records from a :class:`Driver`."""

    records: list[LaunchRecord] = field(default_factory=list)

    def add(self, record: LaunchRecord) -> None:
        self.records.append(record)

    @property
    def total_cycles(self) -> int:
        return sum(record.cycles for record in self.records)

    def summary(self) -> dict[str, Any]:
        return {
            "launches": len(self.records),
            "total_cycles": self.total_cycles,
            "records": [record.as_dict() for record in self.records],
        }
