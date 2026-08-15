-- Export one JSON line per HUMAN game step for behavior-cloning data.
--
-- Run with psql, for example:
--   psql "$DATABASE_URL" -t -A -f scripts/dump_human_steps.sql > human_steps.jsonl
--
-- Each line is a JSON object whose raw fields are consumed by
-- scripts/export_human_traces.py. The previous step's lastDiscard is included
-- so the exporter can reconstruct the state *before* the human acted.

SELECT json_build_object(
    'gameId', gs."gameId",
    'stepIndex', gs."stepIndex",
    'playerIndex', gs."playerIndex",
    'action', gs."actionJson",
    'legalActions', gs."legalActionsJson",
    'view', gs."publicViewJson",
    'prevLastDiscard', lag(gs."publicViewJson" -> 'lastDiscard') OVER (
        PARTITION BY gs."gameId" ORDER BY gs."stepIndex"
    )
)::text
FROM "GameStep" gs
WHERE gs."actionSource" = 'HUMAN'
ORDER BY gs."gameId", gs."stepIndex";
