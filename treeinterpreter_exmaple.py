import random
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from treeinterpreter import treeinterpreter as ti

calif_housing = fetch_california_housing()
for line in calif_housing.DESCR.split("\n")[5:22]:
    print(line)

calif_housing_df = pd.DataFrame(data=calif_housing.data, columns=calif_housing.feature_names)
calif_housing_df["Price($)"] = calif_housing.target
calif_housing_df.head()
X_calif, Y_calif = calif_housing.data, calif_housing.target
print("Dataset Size : ", X_calif.shape, Y_calif.shape)
X_train_calif, X_test_calif, Y_train_calif, Y_test_calif = train_test_split(X_calif, Y_calif, train_size=0.8, test_size=0.2, random_state=123)
print("Train/Test Size : ", X_train_calif.shape, X_test_calif.shape, Y_train_calif.shape, Y_test_calif.shape)

dtree_reg = DecisionTreeRegressor(max_depth=10)
dtree_reg.fit(X_train_calif, Y_train_calif)
print("Test  R^2 Score : %.2f" % dtree_reg.score(X_test_calif, Y_test_calif))
print("Train R^2 Score : %.2f" % dtree_reg.score(X_train_calif, Y_train_calif))

preds, bias, contributions = ti.predict(dtree_reg, X_test_calif)
preds.shape, bias.shape, contributions.shape

random_sample = random.randint(1, len(X_test_calif))
print("Selected Sample     : %d" % random_sample)
print("Actual Target Value : %.2f" % Y_test_calif[random_sample])
print("Predicted Value     : %.2f" % preds[random_sample][0])


def create_contrbutions_df(contributions, random_sample, feature_names):
    contribs = contributions[random_sample].tolist()
    contribs.insert(0, bias[random_sample])
    contribs = np.array(contribs)
    contrib_df = pd.DataFrame(data=contribs, index=["Base"] + feature_names, columns=["Contributions"])
    prediction = contrib_df.Contributions.sum()
    contrib_df.loc["Prediction"] = prediction
    return contrib_df


contrib_df = create_contrbutions_df(contributions, random_sample, calif_housing.feature_names)
contrib_df
