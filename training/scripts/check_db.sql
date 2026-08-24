SELECT
  "actionSource",
  count(*) AS steps,
  count(DISTINCT "gameId") AS games,
  count(DISTINCT "playerIndex") AS seats
FROM "GameStep"
GROUP BY "actionSource"
ORDER BY steps DESC;
