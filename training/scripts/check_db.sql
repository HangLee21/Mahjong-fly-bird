SELECT "status", jsonb_pretty("resultJson") AS result, jsonb_pretty("finalScoreJson") AS final_scores
FROM "Game"
WHERE status = 'FINISHED'
ORDER BY "finishedAt" DESC
LIMIT 1;
