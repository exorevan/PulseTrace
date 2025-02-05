import lime
import lime.lime_tabular
import joblib
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

model = joblib.load("checkpoints/svc/sklearn_xor.pkl")
print(model.predict([[1.0, 0.0]]))
