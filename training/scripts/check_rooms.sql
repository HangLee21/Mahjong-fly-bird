SELECT 'room_total' AS check_name, count(*) AS n FROM "Room";
SELECT 'game_count' AS check_name, count(*) AS n FROM "Game";
SELECT 'step_count' AS check_name, count(*) AS n FROM "GameStep";
SELECT 'steps_by_source' AS check_name, "actionSource", count(*) AS n FROM "GameStep" GROUP BY "actionSource" ORDER BY n DESC;
SELECT 'games_without_room' AS check_name, count(*) AS n FROM "Game" g LEFT JOIN "Room" r ON r.id = g."roomId" WHERE r.id IS NULL;
SELECT 'rooms_without_game' AS check_name, count(*) AS n FROM "Room" r LEFT JOIN "Game" g ON g."roomId" = r.id WHERE g.id IS NULL;
