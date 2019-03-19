import matplotlib
matplotlib.use('Agg')
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pylab
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, RidgeCV, BayesianRidge
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, OrthogonalMatchingPursuit
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from pystacknet.pystacknet import StackNetRegressor
from xgboost import XGBRegressor

def plot_fluid_intelligence_histogram(labels):
	'''plot the histogram of fluid intelligence'''
	n, bins, patches = plt.hist(labels, 50, density=True, facecolor='g', alpha=0.75)
	plt.xlabel('Residual Fluid Intelligence Score')
	plt.ylabel('Number of Subjects')
	plt.title('Histogram of Fluid Intelligence')
	plt.axis([-40, 30, 0, 0.05])
	plt.grid(True)
	plt.savefig('example')

def get_group_labels(labels, number_of_group):
	''' divid the subjects into different groups which have approximately same number of subjects'''
	sorted_labels = np.sort(labels)
	number_of_subject_in_a_group = round(labels.shape[0]/number_of_group)
	boundries = [sorted_labels[i] for i in number_of_subject_in_a_group*np.arange(1, number_of_group)]
	group_labels = np.zeros(labels.shape, labels.dtype)
	for idx, boundry in enumerate(boundries):
		#print(idx+1, boundry)
		group_labels[labels >= boundry] = idx+1
	return group_labels

################################# please change the paths accordingly ########################################

# training file path
train_btsv_file = '/home/pkao/abcd/trainData/btsv01.txt'
train_fi_file = '/home/pkao/abcd/trainData/results/training_fluid_intelligenceV1.csv'

# validation file path
valid_btsv_file = '/home/pkao/abcd/validData/btsv01.txt'
valid_fi_file = '/home/pkao/abcd/validData/results/validation_fluid_intelligenceV1.csv'

# test file path
test_btsv_file = '/home/pkao/abcd/testData/btsv01.txt'

############################### check if the provided files exists or not #####################

assert os.path.exists(train_btsv_file),"btsv01.txt of training set does not exist!!!"
assert os.path.exists(train_fi_file),"training_fluid_intelligenceV1.csv does not exist!!!"
assert os.path.exists(valid_btsv_file),"btsv01.txt of validation set does not exist!!!"
assert os.path.exists(valid_fi_file),"validation_fluid_intelligenceV1.csv does not exist!!!"
assert os.path.exists(test_btsv_file),"btsv01.txt of testing set does not exist!!!"

###############################################################################################################

seed = 1989

dropped_names = ['collection_id', 'btsv01_id', 'dataset_id', 'subjectkey', 'src_subject_id', 
'interview_date', 'collection_title', 'study_cohort_name', 'subject', 'interview_age', 'sex']

############################################# Training Dataset #################################################

df_train_btsv = pd.read_csv(train_btsv_file, delim_whitespace=True, low_memory=False)
df_train_fi = pd.read_csv(train_fi_file)
merged_train_df = pd.merge(df_train_btsv, df_train_fi, how='inner', left_on='subjectkey', right_on='subject')



# histogram of fluid intelligence of training subjects
#fig, ax = plt.subplots()
#df_train_fi.hist(column='residual_fluid_intelligence_score', bins=100, ax=ax)
#fig.savefig('training.png')

df_train_age_gender = merged_train_df[['interview_age', 'sex']]
df_train_age_gender = pd.get_dummies(df_train_age_gender)

df_train = merged_train_df.drop(dropped_names, axis=1)

# Ground-truth of the fluid intelligence score
train_labels = df_train['residual_fluid_intelligence_score'].values
#print(np.mean(train_labels), np.std(train_labels))

# training dataset in dataframe
df_train = df_train.drop('residual_fluid_intelligence_score', axis = 1)

# feature list and features in numpy matrix
feature_list = list(df_train.columns)
train_features = df_train.values
train_age_gender = df_train_age_gender.values
#print(train_features.dtype, train_features.shape)

############################################# Validation Dataset #################################################

df_valid_btsv = pd.read_csv(valid_btsv_file, delim_whitespace=True, low_memory=False)
df_valid_fi = pd.read_csv(valid_fi_file)
merged_valid_df = pd.merge(df_valid_btsv, df_valid_fi, how='inner', left_on='subjectkey', right_on='subject')

# histogram of fluid intelligence of validation subjects
#fig, ax = plt.subplots()
#df_valid_fi.hist(column='residual_fluid_intelligence_score', bins=100, ax=ax)
#fig.savefig('validation.png')

df_valid_age_gender = merged_valid_df[['interview_age', 'sex']]
df_valid_age_gender = pd.get_dummies(df_valid_age_gender)

df_valid = merged_valid_df.drop(dropped_names, axis=1)

valid_labels = df_valid['residual_fluid_intelligence_score'].values
#print(np.mean(valid_labels), np.std(valid_labels))

df_valid = df_valid.drop('residual_fluid_intelligence_score', axis = 1)

valid_features = df_valid.values

valid_age_gender = df_valid_age_gender.values


df_valid_subject_name = merged_valid_df[['subject']]


######################################## Data Pre-processing #####################################################

'''
print('Normalizing features...')
scaler_1 = StandardScaler()
normalized_train_features = scaler_1.fit_transform(train_features)
normalized_valid_features = scaler_1.fit_transform(valid_features)
#print(normalized_train_features.shape[1])

#normalized_age_gender = scaler.fit_transform(age_gender)

print('Performing feature dimension reduction...')
pca_1 = PCA(n_components='mle', svd_solver='full', random_state=seed)
pca_normalized_train_features = pca_1.fit_transform(normalized_train_features)
pca_normalized_valid_features = pca_1.transform(normalized_valid_features)
print(pca_normalized_train_features.shape[1])

print('Removing features with low variance...')
sel_1 = VarianceThreshold(0.5*(1-0.5))
high_variance_pca_normalized_train_features = sel_1.fit_transform(pca_normalized_train_features)
high_variance_pca_normalized_valid_features = sel_1.transform(pca_normalized_valid_features)
#high_variance_feature_list = [name for idx, name in enumerate(feature_list) if sel.get_support()[idx]]
#print(high_variance_pca_normalized_train_features.shape[1])



print('Performing feature selection...')
print('Univariate Selection...')
skb_1 = SelectKBest(f_regression, k=24)
selected_high_variance_pca_normalized_train_features = skb_1.fit_transform(high_variance_pca_normalized_train_features, train_labels)
selected_high_variance_pca_normalized_valid_features = skb_1.transform(high_variance_pca_normalized_valid_features)

'''
############################################ Training + Validation #################################################


df_train_valid_fi = pd.concat([df_train_fi, df_valid_fi])
df_train_valid = pd.concat([df_train, df_valid])
train_valid_labels = np.concatenate((train_labels, valid_labels), axis=0)
train_valid_features = np.array(df_train_valid)
#print(np.mean(train_valid_labels), np.std(train_valid_labels))

# histogram of fluid intelligence of training + validation subjects
#fig, ax = plt.subplots()
#df_train_valid_fi.hist(column='residual_fluid_intelligence_score', bins=100, ax=ax)
#fig.savefig('train_valid.png')

######################################## Data Pre-processing #####################################################


#print('Normalizing features...')
scaler = StandardScaler()
normalized_train_valid_features = scaler.fit_transform(train_valid_features.astype('float64'))
#print(normalized_train_valid_features.shape[1])

#print('Performing feature dimension reduction...')
pca = PCA(n_components='mle', svd_solver='full', random_state=seed)
pca_normalized_train_valid_features = pca.fit_transform(normalized_train_valid_features)
#print(pca_normalized_train_valid_features.shape[1])

#print('Removing features with low variance...')
sel = VarianceThreshold(0.5*(1-0.5))
high_variance_pca_normalized_train_valid_features = sel.fit_transform(pca_normalized_train_valid_features)
#high_variance_feature_list = [name for idx, name in enumerate(feature_list) if sel.get_support()[idx]]
#print(high_variance_pca_normalized_train_valid_features.shape[1])

#print('Performing feature selection...')
#print('Univariate Selection...')
skb = SelectKBest(f_regression, k=24)
selected_high_variance_pca_normalized_train_valid_features = skb.fit_transform(high_variance_pca_normalized_train_valid_features, train_valid_labels)



############################################### StackNet #################################################

'''
### training, validation
models=[
       ###### First Level ########## 
       [
       RandomForestRegressor(n_estimators=800, max_depth=7, random_state=seed, n_jobs=-1),
       RandomForestRegressor(n_estimators=1000, max_depth=3, n_jobs=-1, random_state=seed),
       XGBRegressor(n_estimators=60, max_depth=3, n_jobs=-1, random_state=seed),
       ExtraTreesRegressor(n_estimators=1800, max_depth=7, random_state=seed, n_jobs=-1),
       GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=seed),
       KNeighborsRegressor(n_neighbors=300, weights='uniform', n_jobs=-1),
       ElasticNet(random_state=seed),
       ],
       ####### Second Level #######
       [
       RandomForestRegressor(n_estimators=800, max_depth=3, n_jobs=-1, random_state=seed),
       ],
       ]
'''

### training + validation
models=[
       ###### First Level ########## 
       [
       KernelRidge(alpha=512),
       #MLPRegressor(hidden_layer_sizes=(2), solver='adam', max_iter=10000, random_state=seed),
       #KNeighborsRegressor(n_neighbors=330,  weights='uniform', n_jobs=-1),
       RandomForestRegressor(n_estimators=800, max_depth=11, random_state=seed, n_jobs=-1),
       RandomForestRegressor(n_estimators=1000, max_depth=7, random_state=seed, n_jobs=-1),
       ExtraTreesRegressor(n_estimators=1800, max_depth=9, random_state=seed, n_jobs=-1),
       GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=seed),
       GradientBoostingRegressor(n_estimators=20, max_depth=5, random_state=seed),
       ],
       ####### Second Level #######
       [
       RandomForestRegressor(n_estimators=1000, max_depth=9, n_jobs=-1, random_state=seed),
       ExtraTreesRegressor(n_estimators=1800, max_depth=9, random_state=seed, n_jobs=-1),
       BayesianRidge()
       ],
       [
       Ridge(alpha=512, random_state=seed)
       ]
       ]

############################################# Five-fold Cross-Validation ################################

# divided subjects into differet groups with approximately same numbe of subjects
number_of_group = 60
#group_labels = get_group_labels(train_labels, number_of_group)
group_labels = get_group_labels(train_valid_labels, number_of_group)

# Select which feature to use
#X = np.concatenate((selected_normalized_high_variance_features, normalized_age_gender), axis=1)
#X = selected_high_variance_pca_normalized_train_features
X = selected_high_variance_pca_normalized_train_valid_features
#X = normalized_age_gender
#y = train_labels
y = train_valid_labels

np.save('X.npy', X)
np.save('y.npy', y)

cv_fold = 10
y_mse = np.zeros(cv_fold)
skf = StratifiedKFold(n_splits=cv_fold,shuffle=True,random_state=seed)

#print('Hyper-parameter: ', sys.argv[1])

i = 0
for train_idx, test_idx in skf.split(X, group_labels):
	X_train, X_test = X[train_idx], X[test_idx]
	y_train, y_test = y[train_idx], y[test_idx]

	#print(np.mean(y_train), np.std(y_train))
	#print(np.mean(y_test), np.std(y_test))
	
	# regressor
	#regr = LinearSVR(random_state=seed)
	#regr = SVR(gamma='scale', C=float(sys.argv[1]))
	#regr = Ridge(alpha=int(sys.argv[1]), random_state=seed)
	#regr = LinearRegression(n_jobs=-1)
	#regr = RandomForestRegressor(n_estimators=int(sys.argv[1]), max_depth=15, random_state=seed, n_jobs=-1)
	#regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7), n_estimators=int(sys.argv[1]), random_state=seed)
	#regr = XGBRegressor(n_estimators=int(sys.argv[1]), max_depth=5, n_jobs=-1, random_state=seed)
	#regr = ExtraTreesRegressor(n_estimators=int(sys.argv[1]), max_depth=15, random_state=seed, n_jobs=-1)	
	#regr = ExtraTreesRegressor(n_estimators=1800, max_depth=7, random_state=seed, n_jobs=-1)
	#regr = GradientBoostingRegressor(n_estimators=int(sys.argv[1]), max_depth=5, random_state=seed)
	#regr = GradientBoostingRegressor(n_estimators=10, max_depth=7, random_state=seed)
	#regr = Lasso(alpha=0.001)
	#regr = ElasticNet(alpha=0.001)
	#regr = OrthogonalMatchingPursuit()
	#regr = BayesianRidge()
	#regr = HuberRegressor()
	#regr = KernelRidge(alpha=512)
	#regr = KNeighborsRegressor(n_neighbors=int(sys.argv[1]), weights='uniform', n_jobs=-1)
	#regr = MLPRegressor(hidden_layer_sizes=(2), solver='adam', max_iter=10000, random_state=seed)
	

	regr = StackNetRegressor(models, metric="rmse", folds=5, restacking=True, use_retraining=True, random_state=seed, n_jobs=-1, verbose=1)
	regr.fit(X_train, y_train)
	
	y_pred = regr.predict(X_test)

	# The mean squared error
	print("Fold %d Mean squared error: %.4f" % (i, mean_squared_error(y_test, y_pred)))
	y_mse[i] = mean_squared_error(y_test, y_pred)
	i += 1

print("Training with %d fold cross-validation, Mean squared error: %.4f" %(cv_fold, np.mean(y_mse)))

print("#############################################################################################")

exit()
#################################### Training and Validation ##########################################
X_train = selected_high_variance_pca_normalized_train_features
y_train = train_labels

X_valid = selected_high_variance_pca_normalized_valid_features
y_valid = valid_labels

regr = StackNetRegressor(models, metric="rmse", folds=4, restacking=False, use_retraining=True, random_state=seed, n_jobs=-1, verbose=1)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_valid)

print("Validation Mean squared error: %.4f" % mean_squared_error(y_valid, y_pred))

# save predictions of validation dataset
df_valid_predict_fi = pd.DataFrame(y_pred.flatten(), columns=['predicted_score'])
df_predicted_fi_valid = pd.merge(left=df_valid_subject_name, left_index=True, right=df_valid_predict_fi, right_index=True, how='inner')
df_predicted_fi_valid.to_csv('pred_validation.csv', index= False)
