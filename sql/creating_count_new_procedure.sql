DELIMITER $$

CREATE PROCEDURE ready_for_retrain()
BEGIN
DECLARE number_of_new INT DEFAULT 0;

SELECT COUNT(*) FROM docs WHERE is_new=TRUE INTO number_of_new;

IF number_of_new >= 2 THEN
	SET SQL_SAFE_UPDATES = 0;
	UPDATE docs SET is_new=0 WHERE is_new=1;
    SET SQL_SAFE_UPDATES = 1;
	SELECT intent, processed FROM docs ORDER BY id DESC LIMIT number_of_new;    
ELSE
SELECT number_of_new;
END IF;

END$$

DELIMITER ;

