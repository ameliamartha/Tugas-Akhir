from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import model_selection

class FitnessFunction:
    
    def __init__(self,n_splits = 5,*args,**kwargs):
        """
            Parameters
            -----------
            n_splits :int, 
                Number of splits for cv
            
            verbose: 0 or 1
        """
        self.n_splits = n_splits
    

    def calculate_fitness(self,model,x,y):
        cv_set = np.repeat(-1.,x.shape[0])
        skf = StratifiedKFold(n_splits = self.n_splits)
        for train_index,test_index in skf.split(x,y):
            x_train,x_test = x[train_index],x[test_index]
            y_train,y_test = y[train_index],y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()
            model.fit(x_train,y_train)
            predicted_y = model.predict(x_test)
            cv_set[test_index] = predicted_y
        print('f1-score :',f1_score(y,cv_set))
        P = accuracy_score(y,cv_set)
        cm = confusion_matrix(y, cv_set)
        print(cm)
        print('accuracy_score :',accuracy_score(y,cv_set))
        print('Classification Report :',classification_report(y,cv_set))
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(cv_set)):
            if y[i]==cv_set[i]==1:
               TP += 1
            if cv_set[i]==1 and y[i]!=cv_set[i]:
               FP += 1
            if y[i]==cv_set[i]==0:
               TN += 1
            if cv_set[i]==0 and y[i]!=cv_set[i]:
               FN += 1
        seed = 7
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        scoring = 'accuracy'
        results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        print(results)
        print('result mean : {}, result std : {}'.format(results.mean(), results.std()))
        probas = model.predict_proba(x_test)
        print('CONFUSION MATRIX :')
        print('ACCURACY :',((TP+TN)/(TP+TN+FP+FN))*100)
        print('SENSITIVITY :',(TP/(TP+FN))*100)
        print('SPECIFICITY :',(TN/(TN+FP))*100)
        skplt.metrics.plot_precision_recall_curve(y_test,probas)
        plt.show()
#         fitness = (0.01*(1.0 - P) + (1.0 - 0.01)*(1.0 - (x.shape[1])/x.shape[1]))
        return accuracy_score(y,cv_set)
