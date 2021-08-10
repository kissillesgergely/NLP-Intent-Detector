import os
import pickle
import re
import pymysql
import nltk

class DBOperator:
	def __init__(self):
		self.connection = pymysql.connect(
			host=os.getenv('DB_HOST'),
			user=os.getenv('DB_USER'),
			password=os.getenv('DB_PASSWORD'),
			db=os.getenv('DB_NAME'),
			charset='utf8mb4'
		)

		self.stop_words = set(nltk.corpus.stopwords.words('english'))

	def populate_db(self, map):
		number_of_docs = 0

		with self.connection.cursor() as cur:
			q = "SELECT COUNT(id) FROM docs"
			cur.execute(q)
			results = cur.fetchall()
			number_of_docs = results[0][0]

		if number_of_docs < 1000:
			dict = {}
			dict['raw'] = []
			dict['processed'] = []
			dict['intentions'] = []

			# In the original dataset there were separated training, validation and test batches
			# I use fold testing later, therefore all the available data can be used the same way
			dict = self.raw_file_to_dict('train', dict, map)
			dict = self.raw_file_to_dict('valid', dict, map)
			dict = self.raw_file_to_dict('test', dict, map)

			with self.connection.cursor() as cur:
				rows = []
				index = 0
				for raw in dict['raw']:
					rows.append((dict['intentions'][index], raw, dict['processed'][index], False))
					index = index + 1
				
				# INGORE is added for invalid inputs which have e.g. funny characters
				q = """
						INSERT IGNORE INTO docs (intent, raw, processed, is_new) VALUES (%s, %s, %s, %s)
				"""
				cur.executemany(q, rows)
				self.connection.commit()

		else:
			print('database is already populated')

	def add_new_data_to_db(self, raw, intent):
		word_tokens = nltk.tokenize.word_tokenize(raw)
		processed = [w for w in word_tokens if not w in self.stop_words]
		processed = ' '.join(processed)

		with self.connection.cursor() as cur:
			row = (intent, raw, processed, True)
			# INGORE is added for invalid inputs which have e.g. funny characters
			q = "INSERT IGNORE INTO docs (intent, raw, processed, is_new) VALUES (%s, %s, %s, %s)"
			cur.execute(q, row)
			self.connection.commit()

			q = "CALL ready_for_retrain();"
			cur.execute(q)
			results = cur.fetchall()

			if len(results) > 1:
				return results
			else:
				return None

	def upload_classifier(self, classifier):
		# Fist save it into a file and then upload
		filename = 'classifier_to_upload.sav'
		pickle.dump(classifier.prediction_model, open(os.path.join('./models',filename), 'wb'))
		file_to_upload = self.convertToBinaryData(os.path.join('./models',filename))

		with self.connection.cursor() as cur:
			row = (
				file_to_upload,
				float(classifier.accuracy),
				float(classifier.precision),
				float(classifier.recall),
				float(classifier.f1)
			)           
			q = "INSERT INTO sklearn_models (model, accuracy, prec, recall, f1) VALUES (%s, %s, %s, %s, %s)"
			cur.execute(q, row)
			self.connection.commit()

	def convertToBinaryData(self, filename):
		with open(filename, 'rb') as file:
			binaryData = file.read()
		return binaryData
            
	def raw_file_to_dict(self, filename, dict, map):
		with open(os.path.join('./data', filename), 'r', encoding='utf-8') as f:
			for line in f:
				dict['raw'].append(line)
				# Removal of the slot marks
				line = re.sub(r':[\w-]+ ', ' ', line)
				# The text and the intention are going to be separated
				# We assume we know how the raw document is structured
				doc, intention = line.split(' <=> ')
				word_tokens = nltk.tokenize.word_tokenize(doc)
				filtered_line_word_list = [w for w in word_tokens if not w in self.stop_words]
				filtered_line = ' '.join(filtered_line_word_list)
				dict['processed'].append(filtered_line)
				dict['intentions'].append(map[intention.replace('\n', '')])
		return dict

	def download_classifier(self):
		loaded_model = None
		accuracy = None
		precision = None
		recall = None
		f1 = None
		filename = 'classifier_to_download.sav'
		with self.connection.cursor() as cur:
			q = """
					SELECT model, accuracy, prec, recall, f1 id from sklearn_models
					ORDER BY created DESC
			"""
			cur.execute(q)
			result = cur.fetchone()
			if result:
				file = result[0]
				accuracy = result[1]
				precision = result[2]
				recall = result[3]
				f1 = result[4]

				with open(os.path.join('./models', filename), 'wb') as f:
					f.write(file)
				loaded_model = pickle.load(open(os.path.join('./models', filename), 'rb'))

		return loaded_model, accuracy, precision, recall, f1

	def get_data_for_training(self):
		with self.connection.cursor() as cur:
			q = "SELECT intent, processed FROM docs"
			cur.execute(q)
			results = cur.fetchall()
			intents = [results[x][0] for x in range(len(results))]
			processed = [results[x][1] for x in range(len(results))]
			return intents, processed

	def download_word2vec_model(self):
		loaded_model = None
		filename = 'word2vec_model_to_download.sav'
		try:
			with self.connection.cursor() as cur:
				q = "SELECT model FROM gensim_models ORDER BY id DESC"
				cur.execute(q)
				results = cur.fetchone()
				if len(results) != 0:
					file = results[0]
					with open(os.path.join('./models', filename), 'wb') as f:
						f.write(file)
					loaded_model = pickle.loads(file)
		except:
			# Error with downloading/loading the model
			# We'll get the model from training
			pass

		return loaded_model

	def upload_word2vec_model(self, word2vec_model):
		# Fist save it into a file and then upload
		filename = 'word2vec_model_to_upload.sav'
		pickle.dump(word2vec_model, open(os.path.join('./models',filename), 'wb'))
		file_to_upload = self.convertToBinaryData(os.path.join('./models',filename))

		with self.connection.cursor() as cur:
			filesize = len(file_to_upload)
			# Chunk size in bytes
			chunk_size = 1024
			file_chunks = [file_to_upload[i:i+chunk_size] for i in range(0, filesize, chunk_size)]
			row = (file_chunks[0])
			q = "INSERT INTO gensim_models (model) VALUES (%s)"
			cur.execute(q, row)
			self.connection.commit()

			q = "SELECT id FROM gensim_models ORDER BY id DESC"
			cur.execute(q)
			results = cur.fetchone()
			id = results[0]
			q = "UPDATE gensim_models SET model=concat(model, %s) WHERE id="+str(id)
			rows = tuple(file_chunks[1:])
			cur.executemany(q, rows)
			self.connection.commit()
