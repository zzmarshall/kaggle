
import warnings
import seaborn as sns
import pandas as pd
from base import plot_heatmap
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore',category=DeprecationWarning)


def load_data():
    df = pd.read_csv(r"C:\Users\marshall\Downloads\train.csv")
    df_target = pd.read_csv(r"C:\Users\marshall\Downloads\test.csv")

    y = df.Survived
    X = df.drop(["Survived"], axis=1)

    X_test, X_valid, y_test, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=1
    )

    return df, X_test, X_valid, y_test, y_valid, df_target


def preprocessor(df_X, drop_id=True):

    df = df_X.copy()

    df.Pclass = df.Pclass.astype("object")

    # todo: embarked
    df.loc[df.Embarked.isnull(), "Embarked"] = "S"

    # todo: fare
    df.loc[df.Fare.isnull(), "Fare"] = df[(df.Pclass==3) & (df.Embarked=="S") & (df.Sex=="male")].Fare.mean()
    df.Fare = MinMaxScaler(). \
        fit_transform(
        df.Fare.values.reshape(len(df.Fare), 1)
    )

    # todo: SibSp Parch
    df["Has_family"] = 0
    df.loc[(df.SibSp > 0 )|( df.Parch > 0), "Has_family"] = 1
    df["Family_size"] = df["SibSp"] + df["Parch"]

    # todo: cabin
    df["Has_Cabin"] = 0
    df.loc[(df.Cabin.notnull()), "Has_Cabin"] = 1
    df.loc[(df.Cabin.isna()), "Cabin"] = 0
    df["Cabin"] = df.Cabin.apply(
        lambda x: str(x)[0]
    )

    # todo: Name
    newtitles = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"}

    df["Title"] = df.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
    df["Title"] = df.Title.map(newtitles)
#    df["Name_male"] = 0
#    df["Name_female"] = 0
#    df.loc[df.Name.str.contains("Mr"), "Name_male"] = 1
#    df.loc[df.Name.str.contains("Mrs") | df.Name.str.contains("Miss"), "Name_female"] = 1

    # todo: Ticket
    df["Ticket_shared"] = 0
    df["Ticket_length"] = 0
    df.loc[df.Ticket.duplicated(), "Ticket_shared"] = 1
    df["Ticket_length"] = df.Ticket.apply(lambda x: len(str(x)))

    # todo: Age
    df.loc[(df.Age.isnull()), "Age"] = 0
    df.Age = df.Age.astype("int")

    grouped = df.groupby(["Sex", "Pclass", "Title"])
    df.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

#    df_age = df.filter(
#        regex="Age|Embarked|Cabin|Has_Cabin|Sex|Pclass|Fare"
#    )
#    df_age = pd.get_dummies(df_age)
#
#    df_train = df_age[df_age.Age > 0].values
#    df_predict = df_age[df_age.Age == 0].values
#    y = df_train[:, 0]
#    X = df_train[:, 1:]
#
#    clf_age = RandomForestClassifier(n_estimators=100, random_state=0)
#    clf_age.fit(X, y)
#    predict_age = clf_age.predict(df_predict[:,1:])
#    df.loc[(df.Age == 0), "Age"] = predict_age

    df.Age = MinMaxScaler(). \
        fit_transform(df.Age.values.reshape(len(df.Age), 1))


    if drop_id:
        df.drop(["PassengerId", "Name","Ticket", "SibSp","Parch"], axis=1, inplace=True)
    else:
        df.drop(["Name","Ticket", "SibSp","Parch"], axis=1, inplace=True)
#    plot_heatmap(df)
    dummies_pclass = pd.get_dummies(df.Pclass, prefix="Pclass")
    df[["Sex","Cabin","Embarked","Title"]] = \
        df[["Sex","Cabin","Embarked","Title"]].apply(LabelEncoder().fit_transform)
    df = pd.concat([df, dummies_pclass], axis=1).drop(["Pclass"], axis=1)

    if "Cabin_0" in df:
        df.drop(["Cabin_0"], axis=1, inplace=True)

    return df


def model(X_test, X_valid, y_test, y_valid, tunning=False):
    output = list()
    # kBest
#    model = SelectKBest(chi2, k=10)
#    model.fit_transform(pd.concat([y_test, X_test], axis=1), y_test)
#    opt_df = pd.DataFrame({
#        "name": pd.concat([y_test, X_test], axis=1).columns,
#        "score": model.scores_,
#        "pvalue": model.pvalues_
#    })
#    print(opt_df.sort_values(by="score", ascending=False))
#    # heatmap
#    plot_heatmap(pd.concat([y_test, X_test], axis=1))
    sys.exit(0)
    # LightGBM
    lgbm = LGBMClassifier(n_estimators=500, n_jobs=-1, random_state=0)
    lgbm.fit(X_test, y_test, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])
    acc_lgbm = cross_val_score(lgbm, X_valid, y_valid, cv=10, scoring="accuracy").mean()

    XGB = XGBClassifier(n_estimators=70, n_jobs=-1, random_state=0)
#    tunning=True
    if tunning:
        params = {
            "learning_rate": [0.2],
            "n_estimators": [70],
            "max_depth": [7],
            "min_child_weight": [1],
            "subsample": [1],
            "colsample_bytree": [0.8],
            "gamma": [0.1],
            "reg_alpha": [0],
            "reg_lambda": [1]
        }
        params_tunning = {
        "learning_rate": [0.1, 0.2, 0.3],
        "n_estimators": range(100,500,100),
        "max_depth": range(1,11,2),
        "min_child_weight": range(1,11,2),
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "gamma": [0.1, 0.2, 0.3],
        "reg_alpha": range(1,3,1),
        "reg_lambda": range(1,3,1)
        }
        grid_clf = GridSearchCV(XGB, params, scoring="accuracy", n_jobs=-1)
        grid_clf.fit(X_test, y_test, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])
        XGB = grid_clf.best_estimator_
        print(grid_clf.best_score_)
        print(grid_clf.best_estimator_)
    else:
        XGB.fit(X_test, y_test, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])
    acc_xgb = cross_val_score(XGB, X_valid, y_valid, cv=80, scoring="accuracy").mean()

    # Logistic Regression
    logreg = LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)
    logreg.fit(X_test, y_test)
    acc_lr = cross_val_score(logreg, X_valid, y_valid, cv=10, scoring="accuracy").mean()

    # Random Forest
    rf = RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0)
    if tunning:
        params = {'n_estimators': [80],
                      'max_features': ['auto'],
                      'criterion': ['gini'],
                      'max_depth': [5],
                      'min_samples_split': [6],
                      'min_samples_leaf': [1]
                      }
        params_tunning = {'n_estimators': range(10,100,10),
                      'max_features': ['auto'],
                      'criterion': ['entropy', 'gini'],
                      'max_depth': [5, 10, 15],
                      'min_samples_split': [5, 6, 7],
                      'min_samples_leaf': [1]
                      }
        grid_clf = GridSearchCV(rf, params, scoring="accuracy")
        grid_clf.fit(X_test, y_test)
        rf = grid_clf.best_estimator_
        print(grid_clf.best_score_)
        print(grid_clf.best_estimator_)
    else:
        rf.fit(X_test, y_test)
        print("Before: %s" % rf)
    acc_rf = cross_val_score(rf, X_valid, y_valid, cv=10, scoring="accuracy").mean()

    output = {
        "XGB": acc_xgb,
        "Logistic Regression": acc_lr,
        "Random Forest": acc_rf,
        "Light GBM": acc_lgbm
    }


    importance = pd.DataFrame({
        "columns": X_test.columns,
        "importance": rf.feature_importances_
    })
#    print(importance.sort_values(by='importance', ascending=False))

    print("After: %s" % rf)
    pprint(output, indent=4)
    sys.exit(0)

    return XGB


def predication(clf, X_target):
    preds = clf.predict(X_target)
    return preds


if __name__ == "__main__":
    pd.set_option('display.max_columns', 30)
    _, X_test, X_valid, y_test, y_valid, X_target = load_data()
    X_test = preprocessor(X_test)
    X_valid = preprocessor(X_valid)
    y_target = X_target.Survived
    X_target = X_target.drop(["Survived"], axis=1)
    X_target_ = preprocessor(X_target)
#    print(X_test.describe())
#    print("="*20)
#    print(X_valid.describe())
    if "Cabin_T" in X_test:
        X_test = X_test.drop(["Cabin_T"], axis=1)
    clf = model(X_test, X_valid, y_test, y_valid)
    preds = predication(clf, X_target_)
    pd.DataFrame({
        "PassengerID":X_target.PassengerId,
        "Survived": preds.astype(int)
    }).to_csv(r"C:\Users\marshall\Downloads\result1.csv", index=False)
