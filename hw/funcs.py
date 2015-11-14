

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn import metrics


def nb_models(X_train_dtm, X_test_dtm, y_train, y_test, clf=None):
    nb = clf()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    
    y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
    accuracy_score = metrics.accuracy_score(y_test, y_pred_class)
    roc_auc = metrics.roc_auc_score(y_test, y_pred_prob)
    creport = metrics.classification_report(y_test, y_pred_class)
    
    return """Model: %s \n 
    Accuracy Score: %s \n 
    ROC AUC Score: %s \n 
    Classification Report: \n %s
    
    """ % (clf, accuracy_score, roc_auc, creport)

def many_nb_models(X_train_dtm, X_test_dtm, y_train, y_test):
    report = []

    list_models = [BernoulliNB, LogisticRegression, MultinomialNB]
    for clfr in list_models:
        report.append(nb_models(X_train_dtm, X_test_dtm, y_train, y_test, clf=clfr))
        # print clfr
    return report






