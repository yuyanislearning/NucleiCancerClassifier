from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor

def load_data(data_path = None, test = False):
	result = pd.read_csv(data_path)
	iid = result['slide_id']
	if test:
		y = None
		X = result
	else:
		y = result['label']
		X = result.drop(columns = ['label'],axis = 1)
	# TODO please drop the extra features that can't be used in prediction. e.g. slice_id, patient_id or index of the record
	X = X.drop(columns = 'slide_id')#'header',"X",'texture_X0',
	return X, y, iid

def select_feature(  X_train, y_train, n_features = 150, fs_n_iter = 5):
	# Initialize an empty array to hold feature importances
	feature_importances = np.zeros(X_train.shape[1])
	rf = RandomForestClassifier()
	
	print('Selecting features...')
	print('It will take some time...')
	for i in range(fs_n_iter):
		rf.fit(X_train, y_train)
		# Record the feature importances
		feature_importances += rf.feature_importances_

	feature_importances = feature_importances / fs_n_iter
	feature_importances = pd.DataFrame({'feature': list(X_train.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)
	zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
	
	print('There are %d features with zero importance:' % len(zero_features))
	print(zero_features)

	bottom_features = list(feature_importances['feature'][(n_features+1):X_train.shape[1]])
	print(bottom_features)
	return bottom_features
'''
def train(X_train,X_test, y_train, y_test, n_iter = 50, cv = 10):
	print("start training!")
	n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 19)]
	max_features = ['auto', 'sqrt']
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	bootstrap = [True, False]
	random_grid = {'n_estimators': n_estimators,
				'max_features': max_features,
				'max_depth': max_depth,
				'bootstrap': bootstrap}
	
	print("it will take longer time...")
	#rf = RandomForestRegressor()
	rf = RandomForestClassifier()

	rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = n_iter, cv = cv, verbose=2, random_state=42, n_jobs = -1)
	rf_random.fit(X_train, y_train)
	print("The best hyperparameters for random forest:")
	print(rf_random.best_params_)
	best_random = rf_random.best_estimator_		

	predictions = best_random.predict(X_test)
	print('precision, recall and f1:')
	print(precision_recall_fscore_support(y_test, predictions, average='binary'))
	fpr, tpr, _ = metrics.roc_curve(y_test, predictions, pos_label=1)
	print('AUC:')
	print(metrics.auc(fpr, tpr))
	print("training on all data now")
	best_rf = RandomForestClassifier(n_estimators = rf_random.best_params_['n_estimators'],
									max_features = rf_random.best_params_['max_features'],
									max_depth = rf_random.best_params_['max_depth'],
									bootstrap = rf_random.best_params_['bootstrap'])
	best_rf.fit(pd.concat([X_train, X_test]), pd.concat([y_train,y_test]))

	return best_rf
'''
def train(X_train,X_test, y_train, y_test):
    # train with XGBoost
    model = XGBClassifier(n_estimators = 25, max_depth = 5)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)
    predictions = prob
    #bin_pre = prob<0
    #print(precision_recall_fscore_support(y_test, bin_pre, average='binary'))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions[:,1], pos_label=1)
    print('testing auc:')
    print(metrics.auc(fpr, tpr))
    print("training on all data now")
    model = XGBClassifier(n_estimators = 25, max_depth = 5)
    model.fit(pd.concat([X_train, X_test]), pd.concat([y_train,y_test]))
    
    return model


def predict(train_data_path, test_data_path):
	# load data
	X, y, _ = load_data(train_data_path)
	X_test, _, iid = load_data(test_data_path, test = True)
	X_train, X_dev ,y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=233)
	
	# feature selection
	'''
	drop_features = select_feature(  X_train, y_train,n_features = 150, fs_n_iter = 5)
	X_train = X_train.drop(columns = drop_features)
	X_test = X_test.drop(columns = drop_features)
	X_dev = X_dev.drop(columns = drop_features)
	'''
	# Training!
	clf = train(X_train,X_dev, y_train, y_dev)
	# Prediction
	y_test = clf.predict_proba(X_test)
    
	pd.DataFrame({'id' : iid, 'predictions' : y_test[:,1]}).to_csv('prediction.csv', index = False)

