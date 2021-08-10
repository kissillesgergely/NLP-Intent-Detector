import sys
sys.path.append('../Lib')
from flask import Flask, render_template, request
from classifier import Classifier
from db_operator import DBOperator

#just make it global, but it will be changed in main to the model for usage

num2cat_map = {
	'AddToPlaylist': 0,
	'BookRestaurant': 1,
	'GetWeather': 2,
	'PlayMusic': 3,
	'RateBook': 4,
	'SearchCreativeWork': 5,
	'SearchScreeningEvent': 6
}

cat2num_map = dict([(b,a) for a,b in num2cat_map.items()])

classifier = Classifier()
db_operator = DBOperator()

app = Flask(__name__)

@app.route('/')
def form():
	return render_template('index.html', classifier=classifier)

@app.route('/submitted', methods=['POST'])
def submitted_form():
	text=request.form['input_text']
	pred = classifier.predict(text)	

	return render_template(
		'predicted_form.html',
		sentence=text,
		prediction=cat2num_map[int(pred)],
		accuracy = classifier.accuracy * 100
	)

@app.route('/feedback_received', methods=['POST'])
def feedback_received():
	feedback = request.form['feedback']
	sentence = request.form['sentence']
	result = db_operator.add_new_data_to_db(sentence, feedback)
	if result is not None:
		ground_truth, docs = db_operator.get_data_for_training()		
		classifier.train_classifier(docs, ground_truth)
		db_operator.upload_classifier(classifier)
	return render_template('feedback_received.html', sentence=sentence, feedback=cat2num_map[int(feedback)])


if __name__ == '__main__':
	#it has a condition in itself, does not run the populating all the time
	print('database population...')
	db_operator.populate_db(num2cat_map)

	print('downloading word to vector model')
	classifier.word2vec_model = db_operator.download_word2vec_model()

	if classifier.word2vec_model == None:
		print('training word to vector modell')
		intents, docs = db_operator.get_data_for_training()
		classifier.create_word2vec_model(docs)
		db_operator.upload_word2vec_model(classifier.word2vec_model)

	print('downloading classifier')
	(
		classifier.prediction_model,
		classifier.accuracy,
		classifier.precision,
		classifier.recall,
		classifier.f1
	) = db_operator.download_classifier()

	if classifier.prediction_model == None:
		print('training classifier')
		ground_truth, docs = db_operator.get_data_for_training()		
		classifier.train_classifier(docs, ground_truth)

		db_operator.upload_classifier(classifier)

	app.run(host='0.0.0.0', port=8080, debug=True)
