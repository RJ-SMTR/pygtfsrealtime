from abc import ABC, abstractmethod
from typing import Generic, TypeVar

RawT = TypeVar("RawT")
ValidatedT = TypeVar("ValidatedT")
ResultT = TypeVar("ResultT")


class Ingester(ABC, Generic[RawT, ValidatedT, ResultT]):
    """Common shape every domain's ingester class (GTFSScheduleIngester,
    GPSIngester) must respect: fetch the raw external data, validate it into
    a trustworthy form, and ingest() orchestrates both - plus whatever
    domain-specific compute step follows - into the final artifact a Loop
    publishes. Subclassing this (rather than just following the convention by
    hand) means a class missing one of the three methods fails loudly at
    instantiation (TypeError: Can't instantiate abstract class ...), not
    silently at whatever cycle first calls the missing method.

    ingest()'s `raw` parameter is optional so every ingester can be called
    standalone (`ingester.ingest()` fetches for itself) - callers that need
    the raw fetch result for their own purposes first (e.g. GTFSScheduleLoop
    hashing it to decide whether to skip a cycle) call fetch() themselves and
    pass the result through, so it isn't fetched twice.
    """

    @abstractmethod
    def fetch(self) -> RawT:
        """Fetch this cycle's raw external data."""
        ...

    @abstractmethod
    def validate(self, raw: RawT) -> ValidatedT:
        """Turn raw fetched data into a trustworthy, validated form."""
        ...

    @abstractmethod
    def ingest(self, raw: RawT | None = None) -> ResultT:
        """Orchestrate fetch/validate (fetching for itself if `raw` isn't
        already given) plus any domain-specific compute step, producing the
        final artifact a Loop publishes.
        """
        ...
