class FatalConfigurationError(Exception):
    """Required configuration missing/invalid, discovered only after a background
    loop cycle has already run (e.g. GTFS timezone, only resolvable once
    agency.txt has been read). Unlike ordinary per-cycle exceptions - which
    run_periodic/run_conditional log and swallow so one bad cycle doesn't kill
    the loop - this one is deliberately let through so it can stop the whole
    engine instead of retrying forever.
    """
