# NLP intent detector

This software imitates the behavior of a smart assistant like Siri. The way it works the user has
to provide and example sentence with one of the following intents
(e.g. *play a random song*):

* Add To Playlist
* Book Restaurant
* Get Weather
* Play Music
* Rate Book
* Search Creative Work
* Search Screening Event

And then the application will try identify the intention.

After a couple of inputs the application retrains its prediction model, which might take a little time.

## How to run the application

* First you need to have a MySQL server and create a database for the application.\
This name can be arbitrary but needs to be added to an environment variable.\
Other parameters are needed to be set as environment variables such as\
host, user and password. Like this:
    ```
    export DB_HOST={your host}
    export DB_USER={your user}
    export DB_PASSWORD={your password}
    export DB_NAME={your database name}
    ```
    After you are done  with this you will have to create the database structure by running the sql files from the ```sql``` folder.
    first the ```creating_tables.sql``` and then ```creating_count_new_procedure.sql```
* Install the dependencies: ```pip install -r requirements.txt```
* Create an empty ```models``` folder on level of ```main.py```
* Run ```main.py``` from the terminal. Note it might take some time before\
the application starts to run, when it does you should see:
    ```
    * Running on http://192.168.0.67:8080/ (Press CTRL+C to quit)
    ```
* Head to ```http://192.168.0.67:8080/``` and use the application according to\
the instruction in the UI.

❗ Disclaimer: This application is implemented as a web application just in order to have a simple UI.
Certain aspects like how the retraining or global model access would work
in case of multiple concurrent requests were not considered.

## ML and NLP theoretical background

The most important keywords in this project are *word embeddings*, *gensim* and *word2vec*.

Word embeddings is the idea that we represent words as vectors, and vectors which are close to each other in the vector space, they are also close to each other semantically. Gensim is the library itself I used and word2vec is the specific implementation.\

For the prediction model I used the decision tree classifier from the sklearn library, which a relatively simple model with it's pros and cons.\
(read more at: https://scikit-learn.org/stable/modules/tree.html)

The hyperparameters were tuned experimentally.

The applications also uses different metrics from the sklearn library such as accuracy, precision, recall and f1. These numbers help to know how "good" the classifier is.\
(read more at: https://scikit-learn.org/stable/modules/model_evaluation.html)

For the evaluation of the model I used a technic called cross-validation.\
(read more at: https://scikit-learn.org/stable/modules/cross_validation.html)