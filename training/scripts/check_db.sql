SELECT
  (SELECT count(*) FROM "GameStep") AS total_steps,
  (SELECT count(*) FROM "GameStep" WHERE "actionSource"='HUMAN') AS human_steps,
  (SELECT count(*) FROM "GameStep" WHERE "actionSource"='AI') AS ai_steps,
  (SELECT count(*) FROM "Game") AS games,
  (SELECT max("createdAt") FROM "GameStep") AS last_step,
  (SELECT max("startedAt") FROM "Game") AS last_game;
