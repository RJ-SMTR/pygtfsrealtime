class GTFSIngestError(Exception):
    """Raised when the GTFS schedule callback's raw data can't be turned into DataFrames."""
