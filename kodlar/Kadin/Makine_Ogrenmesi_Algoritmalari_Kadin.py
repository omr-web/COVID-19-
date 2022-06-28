# -*- coding: utf-8 -*-
"""
Copyright (C) 2022 "Omer CEBECİ,Abdullah MEMİŞOĞLU,Mehmet Berk KARTAL"
@yazarlar: Omer CEBECİ,Abdullah MEMİŞOĞLU,Mehmet Berk KARTAL
"""


import numpy as np 
from sklearn import preprocessing 
import pandas as pd 



from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


########### makine öğrenmesi #########3
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn import preprocessing ,neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import sklearn.metrics as metrics

df=pd.read_csv('Olusturulan_Excel_Kadin.csv')



df_numeric=pd.get_dummies(df, columns = ['status'])


  
y=df_numeric.iloc[:,157:158] ##bağımlı değişkenlerin alınması
x=df_numeric.iloc[:,2:156]  ## bağımsız değişkenlerin alınması






Y=np.array(y)
Y=Y.ravel()
X=np.array(x)



#verilerin egitim ve test icin bolunmesi
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)





## Adaptive SMOTE

ada=ADASYN(random_state=130)
X_train_res, y_train_res = ada.fit_resample(x_train, y_train)


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

######################################### GEREKLİ FONKSİYONLAR #################################
def AreaCalculate (fpr, tpr):
    S = []
    if len(fpr) == len(tpr):
        for i in range(1, len(fpr)):
            S.append((fpr[i] - fpr[i - 1])* ((tpr[i] + tpr[i - 1])/2 - 0))
        
    print ("sum " + str(sum(S)))

def Only_Area(fpr, tpr):
    S = []
    if len(fpr) == len(tpr):
        for i in range(1, len(fpr)):
            S.append((fpr[i] - fpr[i - 1])* ((tpr[i] + tpr[i - 1])/2 - 0))
    return (sum(S))


def Calculate_TPR_FPR_and_Draw_ROCCurve(y_test,y_proba,name_of_method): ## Kendimiz tpr ve fpr hesabı yaparak elde ettiğimiz roc curve
    thresholds = np.linspace(1,0,101)                                   ## name of method kullanılan makine öğrenmesi algoritmasının adı

    ROC = np.zeros((101,2))
    for i in range(101):
        t = thresholds[i]
    
       
        TP_t = np.logical_and( y_proba > t, y_test==1 ).sum()
        TN_t = np.logical_and( y_proba <=t, y_test==0 ).sum()
        FP_t = np.logical_and( y_proba > t, y_test==0 ).sum()
        FN_t = np.logical_and( y_proba <=t, y_test==1 ).sum()

      
        FPR_t = FP_t / float(FP_t + TN_t)
        ROC[i,0] = FPR_t
    
       
        TPR_t = TP_t / float(TP_t + FN_t)
        ROC[i,1] = TPR_t
    roc_auc=Only_Area(ROC[:,0], ROC[:,1])
   
    fig = plt.figure(figsize=(6,6))
    plt.plot(ROC[:,0], ROC[:,1], lw=2,label = 'AUC = %f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.title(name_of_method + ' icin ROC CURVE (manual olarak elde edilen fpr tpr ile)')
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xlabel('$FPR(t)$')
    plt.ylabel('$TPR(t)$')
    plt.grid()
   

def DRAW_ROCCURVE_WITH_SKLEARN_TPR_FPR_VALUES(fpr,tpr,name_of_method):  ## name of method kullanılan makine öğrenmesi algoritmasının adı
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.figure()
    plt.title('Receiver Operating Characteristic for ' + name_of_method + " with SKLEARN tpr fpr VALUES" )
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


################################################################################################################
##svm modeli



dvm = SVC( C=33 ,kernel = 'poly',probability=True) 
dvm.fit(X_train_res,y_train_res)
y_pred_dvm=dvm.predict_proba(x_test)
                        
            

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_dvm[:,1])
roc_auc_dvm = metrics.auc(fpr, tpr)
## ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları
print("ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları")
print("SVM")      
print("Hazır sonuc:" + str(roc_auc_dvm))
print("El ile elde edilen sonuc:")
AreaCalculate(fpr,tpr)

#### Grafikler
DRAW_ROCCURVE_WITH_SKLEARN_TPR_FPR_VALUES(fpr,tpr,'SVM')
Calculate_TPR_FPR_and_Draw_ROCCurve(y_test,y_pred_dvm[:,1],'SVM') 


##knn


clf=neighbors.KNeighborsClassifier(n_neighbors=77,weights='distance')  
clf.fit(X_train_res,y_train_res)
y_pred_clf=clf.predict_proba(x_test)
                
                
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_clf[:,1])
roc_auc_clf = metrics.auc(fpr, tpr)
    
## ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları
print("ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları")
print("KNN")      
print("Hazır sonuc:" + str(roc_auc_clf))
print("El ile elde edilen sonuc:")
AreaCalculate(fpr,tpr)

#### Grafikler
DRAW_ROCCURVE_WITH_SKLEARN_TPR_FPR_VALUES(fpr,tpr,'KNN')
Calculate_TPR_FPR_and_Draw_ROCCurve(y_test, y_pred_clf[:,1],'KNN') 
    

## RANDOM FOREST modeli  

rf_reg=RandomForestClassifier(n_estimators =44,random_state=0,criterion='entropy',max_features='log2',max_depth=23) 
                
rf_reg.fit(X_train_res,y_train_res)                          
y_pred_rf_reg=rf_reg.predict_proba(x_test)                         
                
                
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_rf_reg[:,1])
roc_auc_rf_reg = metrics.auc(fpr, tpr)
    
    ## ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları
print("ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları")
print("Random Forest")      
print("Hazır sonuc:" + str(roc_auc_rf_reg))
print("El ile elde edilen sonuc:")
AreaCalculate(fpr,tpr)

#### Grafikler
DRAW_ROCCURVE_WITH_SKLEARN_TPR_FPR_VALUES(fpr,tpr,'RF')
Calculate_TPR_FPR_and_Draw_ROCCurve(y_test,y_pred_rf_reg[:,1],'RF') 
## Karar Ağacı


r_dt = DecisionTreeClassifier(random_state=0,criterion='entropy',max_features=107) 
r_dt.fit(X_train_res,y_train_res)
                        
                        
  # Test verisinin tahminleri
y_pred_r_dt = r_dt.predict_proba(x_test)
                    
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_r_dt[:,1])
roc_auc_dt = metrics.auc(fpr, tpr)
                    
## ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları
print("ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları")
print("Decison Tree")      
print("Hazır sonuc:" + str(roc_auc_dt))
print("El ile elde edilen sonuc:")
AreaCalculate(fpr,tpr)

#### Grafikler
DRAW_ROCCURVE_WITH_SKLEARN_TPR_FPR_VALUES(fpr,tpr,'DT')
Calculate_TPR_FPR_and_Draw_ROCCurve(y_test,y_pred_r_dt[:,1],'DT') 


#################### Gaussian Naive Bias  


gnb = GaussianNB(priors=[0.3,0.7],var_smoothing=7)
gnb.fit(X_train_res, y_train_res)
            
y_pred_gnb = gnb.predict_proba(x_test)
            
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_gnb[:,1])
roc_auc_gnb_reg = metrics.auc(fpr, tpr)
            
    ## ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları
print("ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları")
print("Gaussian Naive")      
print("Hazır sonuc:" + str(roc_auc_gnb_reg))
print("El ile elde edilen sonuc:")
AreaCalculate(fpr,tpr)

#### Grafikler
DRAW_ROCCURVE_WITH_SKLEARN_TPR_FPR_VALUES(fpr,tpr,'GNB')
Calculate_TPR_FPR_and_Draw_ROCCurve(y_test,y_pred_gnb[:,1],'GNB') 

################################################
### Ensemble Learning
                          
clf_voting=VotingClassifier(
                                    estimators=[('label3',r_dt),('lable3',dvm)],voting='soft',n_jobs=1)
                            
clf_voting.fit(X_train_res,y_train_res)
y_pred_clf_voting=clf_voting.predict_proba(x_test)
                            
                            
fpr3, tpr3, threshold3 = metrics.roc_curve(y_test, y_pred_clf_voting[:,1])
roc_auc_voting = metrics.auc(fpr3, tpr3)

## ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları
print("ROC AUC değerlerinin hazır sklearn fonksiyonu ve el ile yazılan fonksiyon ile elde edilen sonucları")
print("Ensemble Learning")      
print("Hazır sonuc:" + str(roc_auc_voting))
print("El ile elde edilen sonuc:")
AreaCalculate(fpr3,tpr3)

#### Grafikler
DRAW_ROCCURVE_WITH_SKLEARN_TPR_FPR_VALUES(fpr3,tpr3,'Voting')
Calculate_TPR_FPR_and_Draw_ROCCurve(y_test,y_pred_clf_voting[:,1],'Voting') 

##########################################################################################################################



## Aşağıdaki kod satırları elde edilen sonuçlar kaydedilmek istenirse yorum'dan çıkartılabilir.

############################## modellerin ve verilerin kaydedilmesi 
## verilerin kaydedilmesi 

'''
import pickle
with open("train__x", "wb") as file:
    pickle.dump(x_train, file)
    
with open("train_y", "wb") as file:
    pickle.dump(y_train, file)

with open("test_x", "wb") as file:
    pickle.dump(x_test, file)
    
with open("test_y", "wb") as file:
    pickle.dump(y_test, file)


with open("X_train_res", "wb") as file:
    pickle.dump(X_train_res, file)
    
with open("y_train_res", "wb") as file:
    pickle.dump(y_train_res, file)


### modellerin kaydedilmesi ######
model_filename = "My_SVM_model.sav"
model_filename_knn = "My_knn_model.sav"
model_filename_rf = "My_rf_model.sav"
model_filename_dt = "My_dt_model.sav"
model_filename_voting = "My_voting_model.sav"
model_filename_gn = "My_gn_model.sav"

saved_model1 = pickle.dump(dvm, open(model_filename,'wb'))
saved_model2 = pickle.dump(clf, open(model_filename_knn,'wb'))
saved_model3 = pickle.dump(rf_reg, open(model_filename_rf,'wb'))
saved_model4 = pickle.dump(r_dt, open(model_filename_dt,'wb'))
saved_model5 = pickle.dump(gnb, open(model_filename_gn,'wb'))
saved_model6 = pickle.dump(clf_voting, open(model_filename_voting,'wb'))
'''
