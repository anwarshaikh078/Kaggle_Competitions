import pandas as pd
from sklearn import tree

train = pd.read_csv("dataset/train.csv")
train = train.drop("Cabin",axis=1)
train = train.dropna()

y = train['Survived']
X = train.drop(["Survived", "PassengerId","Name","Ticket"],axis=1)
X = pd.get_dummies(X)
#print(X)

dtc = tree.DecisionTreeClassifier()
dtc.fit(X,y)

test = pd.read_csv("dataset/test.csv")
ids = test[['PassengerId']]
test.drop(["Cabin", "PassengerId","Name","Ticket"],axis=1,inplace=True)
test.fillna(2, inplace=True)
test = pd.get_dummies(test)
#print(test)

predictions = dtc.predict(test)
print("Predictions ")
print(type(predictions))
print("IDS")
print(ids)
results = ids.assign(Survived = predictions) # assign predictions to ids
results.to_csv("titanic-results.csv", index=False)
#results = ids.assign(predictions) # assign predictions to ids
#results.to_csv("titanic-results.csv", index=False)