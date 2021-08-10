CREATE TABLE intents(
	id INT PRIMARY KEY NOT NULL,
    intent_name VARCHAR(50) NOT NULL    
);

INSERT INTO intents (id, intent_name) VALUES
(0, 'AddToPlaylist'),
(1, 'BookRestaurant'),
(2, 'GetWeather'),
(3, 'PlayMusic'),
(4, 'RateBook'),
(5, 'SearchCreativeWork'),
(6, 'SearchScreeningEvent');

CREATE TABLE docs(
	id INT PRIMARY KEY AUTO_INCREMENT,
    intent int NOT NULL,
    raw VARCHAR(500) NOT NULL,
    processed VARCHAR(500) NOT NULL,
    is_new BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (intent) REFERENCES intents(id)
);

CREATE TABLE gensim_models(
	id INT PRIMARY KEY AUTO_INCREMENT,
    model MEDIUMBLOB NOT NULL,
    created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP    
);

CREATE TABLE sklearn_models(
	id INT PRIMARY KEY AUTO_INCREMENT,
    model MEDIUMBLOB NOT NULL,
    accuracy FLOAT NOT NULL,
    prec FLOAT NOT NULL,
    recall FLOAT NOT NULL,
    f1 FLOAT NOT NULL,
    created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP    
);
