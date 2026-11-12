"""
Party and Candidate: a candidate's party classification, including the
coalition-independent override logic that can't live on Party alone.

Idaho's Todd Achilles and Natalie Fleming share the identical raw party
label "ind" but have different coalition status (Achilles is the real
challenger, Fleming is a minor repeat candidate) -- classification has to
happen at the candidate level, not the label level, which is why this
isn't just a bigger dict on Party.

ALASKA / RANKED-CHOICE CAVEAT (not yet handled by this module):
Alaska (and Maine, for federal races) determine their actual winner via
ranked-choice instant-runoff reallocation of eliminated candidates' votes,
not first-round plurality. This project's two_party_votes convention
everywhere (sum D candidates, sum R candidates, drop everyone else) is a
reasonable approximation of an eventual RCV outcome, but it is NOT the
same thing -- it implicitly assumes eliminated candidates' votes would
split the same way the already-counted D/R votes do, which has no reason
to be true. Alaska's 2026 Senate race (Sullivan/R 47.3%, Peltola/D 48.9%,
Heikes/R 2.2%, no majority as of the last check) is exactly the case where
this matters: Heikes is also a Republican, so his voters' second choices
plausibly lean Sullivan, which could close or flip the Sullivan-Peltola
gap once RCV reallocates -- first-round totals alone can't tell us that.
This is a third, independent reason Alaska is excluded from this project
right now, on top of the geography mismatch (County vs. State House
District) and the race being uncalled: fixing the geography problem alone
would NOT make Alaska's numbers trustworthy. Candidate/Party deliberately
do not paper over this with an approximation -- there is no
`is_ranked_choice` handling here yet. Whoever adds Alaska back needs to
either source the state's own RCV-tabulated final round, or add an
explicit `is_ranked_choice_estimate` flag to whatever uses first-round
Candidate data, so the approximation is never silently treated as final.
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

_DEM_LABELS = {
    "democrat",
    "democratic",
    "democrat, working families",
    "democratic-farmer-labor",
    "d",
    "dem",
}
_REP_LABELS = {
    "republican",
    "r",
    "gop",
}


class Party(Enum):
    DEM = "D"
    REP = "R"
    OTHER = "OTHER"

    @classmethod
    def from_label(cls, raw_label: str) -> "Party":
        key = str(raw_label).strip().lower()
        if key in _DEM_LABELS:
            return cls.DEM
        if key in _REP_LABELS:
            return cls.REP
        return cls.OTHER


@dataclass(frozen=True)
class Candidate:
    name: str
    state: str
    raw_party: Party

    # (state, lowercased name) -> this candidate's votes count as DEM for
    # two-party purposes, regardless of raw_party. See
    # pipeline_walkthrough.md for how each of these was confirmed (web
    # search + the raw party-label dump, not assumed): NE's and SD's
    # Democrats withdrew after their primaries; ID's withdrew in July.
    _COALITION_OVERRIDES: ClassVar[set[tuple[str, str]]] = {
        ("NE", "dan osborn"),
        ("SD", "brian bengs"),
        ("ID", "todd achilles"),
    }

    @property
    def is_coalition_independent(self) -> bool:
        return (self.state, self.name.strip().lower()) in self._COALITION_OVERRIDES

    @property
    def effective_party(self) -> Party:
        return Party.DEM if self.is_coalition_independent else self.raw_party

    @classmethod
    def coalition_states(cls) -> set[str]:
        """States with at least one coalition-independent override --
        e.g. used to flag every county in that state's race, not just the
        specific candidate's rows."""
        return {state for state, _ in cls._COALITION_OVERRIDES}
