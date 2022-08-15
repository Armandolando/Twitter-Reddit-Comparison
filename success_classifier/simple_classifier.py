import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# Sklearn modules & classes
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import scale
import itertools

data = pd.read_csv('data_twitter_final.csv')

data_success = data[data['label']==1]
data_not_success = data[data['label']==0]
df_not_success_downsampled = resample(data_not_success, 
                                        replace=False,     # sample with replacement
                                        n_samples=len(data_success['label']),    # to match majority class
                                        random_state=123) # reproducible results
data = pd.concat([data_success, df_not_success_downsampled])
data = data.sample(frac=1)

#data = data.drop(columns=['ovv'])

print(data['label'].value_counts())

X = data.drop(columns=['id', 'label'])

X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=1, stratify=data['label'])

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
'''
param_grid = {'C': [0.01, 1, 5, 10], 'kernel':['rbf','linear'], 'gamma': [0.1, 0.5, 0.01, 0.001, 0.00001]}
clf_cv = GridSearchCV(SVC(), param_grid, cv = 10, verbose=2)
clf_cv.fit(X_train_std, y_train)
print("Best Parameters:\n", clf_cv.best_params_)
'''
'''
param_grid = {'min_samples_leaf': [1, 2, 3, 5, 10], 'min_samples_split':[2,4,6,8,10,15], 'max_features': ['sqrt', 'log2']}
clf_cv = GridSearchCV(RandomForestClassifier(random_state=137), param_grid, cv = 10, verbose=2)
clf_cv.fit(X_train_std, y_train)
print("Best Parameters:\n", clf_cv.best_params_)
'''
#model = SVC(C=clf_cv.best_params_['C'], gamma=clf_cv.best_params_['gamma'], random_state=1, kernel=clf_cv.best_params_['kernel'])
model = RandomForestClassifier(min_samples_leaf=5,min_samples_split=8, max_features='auto', random_state=1)
 
# Fit the model
model.fit(X_train_std, y_train)

y_predict = model.predict(X_test_std)
 
# Measure the performance
print("Accuracy score %.3f" %metrics.accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict, digits=5))

importance = pd.DataFrame({'Variable':X.columns,
              'Importance':model.feature_importances_}).sort_values('Importance', ascending=False)

print(importance.head(20))

start_time = time.time()
importances = model.feature_importances_
std = np.std([
    tree.feature_importances_ for tree in model.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=X.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
#ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,  fontsize=15)
    plt.yticks(tick_marks, classes, rotation=90, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize = 14)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

def class_feature_importance(X, Y, feature_importances):
    N, M = X.shape
    X = scale(X)

    out = {}
    for c in set(Y):
        out[c] = dict(
            zip(range(N), np.mean(X[Y==c, :], axis=0)*feature_importances)
        )

    return out

cnf_matrix = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(7,7))
plot_confusion_matrix(cnf_matrix, classes=['not_success','success'], title="Confusion matrix")
#plot_confusion_matrix(cnf_matrix, classes=[ "low", 'medium', 'high' ], title="Confusion matrix")
plt.show()

ff = np.array(X.columns)
result = class_feature_importance(X, data['label'], importances)

print(result)
# Plot the feature importances of the forest

plt.rc('font', size=15)

titles = ["Not Successful", "Successful"]
for t, i in zip(titles, range(len(result))):
    plt.figure()
    #plt.rcParams['figure.figsize'] = [16, 6]
    #plt.title(t)
    plt.bar(range(len(result[i])), result[i].values(),
           color="b", align="center")
    plt.xticks(range(len(result[i])), ff[list(result[i].keys())], rotation=90)
    plt.ylabel("Mean decrease in impurity")
    plt.xlim([-1, len(result[i])])
    plt.tight_layout()
    plt.show()



data_topics_no_success = pd.read_csv('data_topics_no_succ.csv')
data_topics = pd.read_csv('data_topics_final.csv')

X = data_topics.drop(columns=['id', 'topic'])
X = sc.transform(X)

X_no = data_topics_no_success.drop(columns=['id', 'topic'])
X_no = sc.transform(X_no)

y_predict = model.predict(X)
y_predict_no = model.predict(X_no)

print(y_predict)
print(len(y_predict))
print(len(data_topics['topic']))
data_topics['predict'] =  y_predict
data_topics_no_success['predict'] =  y_predict_no

print(data_topics.head())

topic_list = ['Smartphones', 'Human-Relationships', 'Body-care', 'Education', 'Books', 'Money', 'Racism', 'Sexual-orientation', 'Baseball', 'Fishing', 'Sleeping', 'Music']
perc_list = []
for topic in topic_list:
    topics_df = data_topics[data_topics['topic'] == topic]
    print(f"{topic}: {int(len(topics_df[topics_df['predict'] > 0])/len(topics_df['predict']) * 100)}")
    perc_list.append(int(len(topics_df[topics_df['predict'] > 0])/len(topics_df['predict']) * 100))

perc_list_no = []
for topic in topic_list:
    topics_df = data_topics_no_success[data_topics_no_success['topic'] == topic]
    print(f"{topic}: {int(len(topics_df[topics_df['predict'] > 0])/len(topics_df['predict']) * 100)}")
    perc_list_no.append(int(len(topics_df[topics_df['predict'] > 0])/len(topics_df['predict']) * 100))


fig, ax = plt.subplots(figsize=(12, 6))
barWidth = 0.25
r1 = np.arange(len(perc_list))
ax.bar(r1, perc_list, barWidth, color='blue')
# Add xticks on the middle of the group bars
plt.xlabel('Topic', fontweight='bold')
plt.ylabel('High-Engagement Post Percentage', fontweight='bold')
plt.xticks([r for r in range(len(perc_list))], topic_list, rotation=40)
ax.set_ylim(0,100)
# Create legend & Show graphic
#plt.title("Percentage of succesful retweets")
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
barWidth = 0.25
r1 = np.arange(len(perc_list_no))
ax.bar(r1, perc_list_no, barWidth, color='blue')
# Add xticks on the middle of the group bars
plt.xlabel('Topic', fontweight='bold')
plt.ylabel('High-Engagement Post Percentage', fontweight='bold')
plt.xticks([r for r in range(len(perc_list_no))], topic_list, rotation=40)
ax.set_ylim(0,100)
# Create legend & Show graphic
#plt.title("Percentage of succesful retweets")
plt.grid()
plt.show()

'''
thresh = 100
#threshs = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
threshs = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

df_succ = topics_df[topics_df['predict'] > 0]
df_succ_no = data_topics_no_success[data_topics_no_success['predict'] > 0]
for thresh in reversed(threshs):
    
    #print(thresh)
    #print(f"user_followers_count:{df_succ[df_succ['user_followers_count']< thresh]['user_followers_count'].mean()}")
    #print(f"reading_time:{df_succ[df_succ['user_followers_count']< thresh]['reading_time'].mean()}")
    #print(f"char_count:{df_succ[df_succ['user_followers_count']< thresh]['char_count'].mean()}")
    #print(f"CLI:{df_succ[df_succ['user_followers_count']< thresh]['CLI'].mean()}")
    #print(f"ARI:{df_succ[df_succ['user_followers_count']< thresh]['ARI'].mean()}")
    
    print(f"{thresh} & {round(df_succ[df_succ['user_followers_count']< thresh]['reading_time'].mean(), 2)} & {round(df_succ_no[df_succ_no['user_followers_count']< thresh]['reading_time'].mean(), 2)} & {round(df_succ[df_succ['user_followers_count']< thresh]['char_count'].mean(),0)} & {round(df_succ_no[df_succ_no['user_followers_count']< thresh]['char_count'].mean(),0)} & {round(df_succ[df_succ['user_followers_count']< thresh]['CLI'].mean(),1)} & {round(df_succ_no[df_succ_no['user_followers_count']< thresh]['CLI'].mean(),1)} & {round(df_succ[df_succ['user_followers_count']< thresh]['ARI'].mean(),1)} & {round(df_succ_no[df_succ_no['user_followers_count']< thresh]['ARI'].mean(),1)} \\".replace('\\', '\\\\'))
'''