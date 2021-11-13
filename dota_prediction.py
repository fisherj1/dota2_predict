#Встроенные
import argparse
import random
import sys

#Для анализа данных
import pandas as pd
import numpy as np

#Для првоедения экспериментов
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

#Для отрисовки графиков
import matplotlib.pyplot as plt

#Для измерения времени работы кода
import time
import datetime

#================Аргпарсеры для передачи параметров через кмоандную строку================#
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data-filename', type=str, help='file name of data as .csv')
	parser.add_argument('--test-data-filename', type=str, default='data/features_test.cvs', help='file name of test data as .csv')
	parser.add_argument('--base-data', action='store_true', help='use base data')
	parser.add_argument('--fillna', type=str, default='zeros', help='fill nan in data (must be: zeros, min, max, mean)')
	parser.add_argument('--model-type', type=int, default=0, help='must be 0 (gradboost) or 1 (linreg)')
	parser.add_argument('--n-splits', type=int, default=5, help='number of folds of cross-validator')
	parser.add_argument('--base-n-estimators', action='store_true', help='n-estimators for model [10, 20, 30]')
	parser.add_argument('--categorical', action='store_true', help='use linreg model without categorical values')
	parser.add_argument('--percent', type=float, default=1.0, help='precent of data to train')
	parser.add_argument('--graph-name', type=str, default='1', help='name of graph')
	parser.add_argument('--normalize', action='store_true', help='normalize data')
	parser.add_argument('--add-bag', action='store_true', help='add a bag of words')
	parser.add_argument('--n-estimators', type=int, default=10, help='n-estimators for gradboost model')
	parser.add_argument('--c', type=float, default=1.0, help='c for linreg model')

	args = parser.parse_args()
	cfg = vars(args)
	args = argparse.Namespace(**cfg)
	print(args, '\n')

	assert args.data_filename is not None
	return args

#================Чтобы заполнить пропуски в данных нулями, либо минмиальным (максимальным или средним) значением этого признака================#
def get_fillna_values(features, fillna): #получаем values для fillna
	fillna_columns = features.columns[features.isna().any()].tolist()
	fillna_values = features[fillna_columns].agg(['min', 'max', 'mean'])
	if fillna == 'zeros':
		return 0
	elif fillna == 'min':
		return fillna_values.loc['min'].to_dict()
	elif fillna == 'max':
		return fillna_values.loc['max'].to_dict()
	elif fillna == 'mean':
		return fillna_values.loc['mean'].to_dict()

#================Загрузка и предобработка данных================#	
def get_data(cfg):
	features = pd.read_csv(cfg.data_filename, index_col='match_id')
	features = features.sample(frac=cfg.percent)
	target_class = features.loc[:, 'radiant_win']

	if cfg.base_data:
		features = features.drop(labels=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire', 
			'barracks_status_radiant', 'barracks_status_dire'], axis=1)
	
	
	unique_values = np.unique(features[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
	 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']].values)
	N = unique_values.size
	unique_values_inds = np.arange(1, N+1)
	unique_values_and_inds_dict = dict(zip(unique_values, unique_values_inds))
	print("Unique heros:", N)
	X_pick = None

	if cfg.categorical:
		if cfg.add_bag:
			X_pick = np.zeros((features.shape[0], N))
			start_time = datetime.datetime.now()
			for i, match_id in enumerate(features.index):
				for p in range(5):
					X_pick[i, unique_values_and_inds_dict[int(features.loc[match_id, 'r%d_hero' % (p+1)])]-1] = 1
					X_pick[i, unique_values_and_inds_dict[int(features.loc[match_id, 'd%d_hero' % (p+1)])]-1] = -1
			print('Time elapsed to make bag:', datetime.datetime.now() - start_time)		
		features = features.drop(labels=['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 
			'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
		
	features = features.fillna(value=get_fillna_values(features, cfg.fillna), axis=0)
	if cfg.normalize:
		features=(features-features.min())/(features.max()-features.min())

	X = features.values
	if X_pick is not None:
		X = np.concatenate((X, X_pick), axis=1)
	y = target_class.values
	return X, y

#================Загрузка и предобработка тестовых данных================#	
def get_test_data(cfg):
	features = pd.read_csv(cfg.test_data_filename, index_col='match_id')
	
	unique_values = np.unique(features[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
	 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']].values)
	N = unique_values.size
	unique_values_inds = np.arange(1, N+1)
	unique_values_and_inds_dict = dict(zip(unique_values, unique_values_inds))
	print("Unique test heros:", N)
	X_pick = None

	if cfg.categorical:
		if cfg.add_bag:
			X_pick = np.zeros((features.shape[0], N))
			start_time = datetime.datetime.now()
			for i, match_id in enumerate(features.index):
				for p in range(5):
					X_pick[i, unique_values_and_inds_dict[int(features.loc[match_id, 'r%d_hero' % (p+1)])]-1] = 1
					X_pick[i, unique_values_and_inds_dict[int(features.loc[match_id, 'd%d_hero' % (p+1)])]-1] = -1
			print('Time elapsed to make bag:', datetime.datetime.now() - start_time)		
		features = features.drop(labels=['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 
			'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
		

	features = features.fillna(value=get_fillna_values(features, cfg.fillna), axis=0)

	if cfg.normalize:
		features=(features-features.min())/(features.max()-features.min())

	X = features.values
	ids = features.index.values
	if X_pick is not None:
		X = np.concatenate((X, X_pick), axis=1)
	return X, ids

#================Оценка качества той или иной модели================#
def get_estimator(n_splits, X, y, model_type, base_n_estimators=False, graph_name='1'):

	kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)

	if model_type == 0:
		if base_n_estimators:
			n_estimators = [10, 20, 30]
		else:
			n_estimators = range(1, 100, 10)

		scores = list()
		for i in n_estimators:
			model = GradientBoostingClassifier(n_estimators=i)
			start_time = datetime.datetime.now()
			score = cross_val_score(model, X, y, scoring='roc_auc', cv=kf).mean()
			print('Time elapsed:', datetime.datetime.now() - start_time, "n-estimators =", i, "score =", score)
			scores.append(score)

		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(n_estimators, scores)
		fig.savefig(graph_name+'.png')
		plt.close(fig) 

	elif model_type == 1:

		cs = np.logspace(-1, -0.01, 10)
		scores = list()
		for c in cs:
			model = LogisticRegression(C=c, random_state=1)
			start_time = datetime.datetime.now()
			score = cross_val_score(model, X, y, scoring='roc_auc' ,cv=kf).mean()
			print('Time elapsed:', datetime.datetime.now() - start_time, "c =", round(c, 4), "score =", score)
			scores.append(score)

		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(cs, scores)
		fig.savefig(graph_name+'.png')
		plt.close(fig)


def get_trained_model(cfg, X, y):
	model = None
	if cfg.model_type == 0:
		model = GradientBoostingClassifier(n_estimators=cfg.n_estimators, random_state=1)
	elif cfg.model_type == 1:
		model = LogisticRegression(C=cfg.c, random_state=1)
	model.fit(X, y)
	return model

def test(model, X_test, ids):
	preds = model.predict(X_test)
	df = pd.DataFrame({"match_id" : ids, "radiant_win" : preds})
	df.to_csv("submission.csv", index=False)
	preds_proba = pd.Series(model.predict_proba(X_test)[:, 1])
	print(preds_proba.describe())

if __name__ == '__main__':
	args = parse_args()
	X, y = get_data(args)
	print(X.shape, y.shape)
	#get_estimator(args.n_splits, X, y, args.model_type, base_n_estimators=args.base_n_estimators, graph_name=args.graph_name)
	X_test, ids = get_test_data(args)
	print(X_test.shape, ids)
	test_model = get_trained_model(args, X, y)
	test(test_model, X_test, ids)



