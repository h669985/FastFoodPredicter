# model.py
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def train_and_predict(X_train, y_train, X_test, seed=None, model_selection=RandomForestClassifier):
    model = model_selection(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred
