from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.utils.class_weight import compute_sample_weight

# Split the 'features' and 'income' data into training and testing sets
def xy_split(data, target, random_state):

    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    target, 
                                                    test_size = 0.2, 
                                                    random_state = random_state)
    # Show the results of the split
    #print("Training set has {} samples.".format(X_train.shape[0]))
    #print("Testing set has {} samples.".format(X_test.shape[0]))

    return X_train, X_test, y_train, y_test
    
    
def train_predict(data, target, clf, beta):
    
    X_train, X_test, y_train, y_test = xy_split(data, target, 0)

    sample_weights = compute_sample_weight('balanced', y_train, indices=None)
    
    clf = clf
    clf_name = clf.__class__.__name__
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
     
    return clf, y_test, y_pred
    
if __name__ == '__main__':
    print('No direct calling of this module.')