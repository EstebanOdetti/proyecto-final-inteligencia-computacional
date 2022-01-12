from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report
from sklearn import svm
import numpy as np

# Cargamos datos
datos_totales = pd.read_csv(r'heart_normalizado.csv')  # ESTE ANDA

# Dividimos en salida y datos
salida_esperada = datos_totales.iloc[1:-1, -1]
datos = datos_totales.iloc[1:-1, 0:-1]

# Usamos train test split para hacer esquema 80-20
datos_tr, datos_ts, salida_esperada_tr, salida_esperada_ts = train_test_split(datos, salida_esperada, test_size=0.2,
                                                                              random_state=0)
#################### Feature importance ###########################
"""
clfRFC = RandomForestClassifier()
clfRFC.fit(datos_tr.values, salida_esperada_tr.values)
# Normalizamos features
feature_importance_normalizado = np.empty(clfRFC.feature_importances_.shape[0])
for i in range(clfRFC.feature_importances_.shape[0]):
    maximo = max(clfRFC.feature_importances_)
    minimo = min(clfRFC.feature_importances_)
    feature_importance_normalizado[i] = (clfRFC.feature_importances_[i] - minimo) / (maximo - minimo)
plt.figure()
plt.barh(datos_totales.iloc[1:-1, 0:-1].columns.values, feature_importance_normalizado)
plt.ylabel("Feature")
plt.xlabel("Feature importance for RFC")
plt.show()
"""
####Ploteamos algunas features#####

##datos_plot = pd.read_csv(r'heart.csv')
##plt.figure()
##datos_plot['age'].sort_values().value_counts(sort=False).plot(kind="bar")
##plt.xlabel("Edad")
##plt.ylabel("Cantidad")
##plt.show()

##plt.figure()
##datos_plot['cp'].sort_values().value_counts(sort=False).plot(kind="bar")
##plt.xlabel("Chest pain")
##plt.ylabel("Cantidad")
##plt.show()

##plt.figure()
##datos_plot['sex'].value_counts().plot(kind="pie",autopct='%.0f%%')
##plt.show()

##plt.figure()
##datos_plot['target'].value_counts().plot(kind="pie",autopct='%.0f%%')
##plt.show()

plt.figure()
salida_esperada_tr.value_counts().plot(kind="pie",autopct='%.0f%%')
plt.show()
print(salida_esperada_ts.value_counts())
plt.figure()
salida_esperada_ts.value_counts().plot(kind="pie",autopct='%.0f%%')
plt.show()

#################### SVC ###########################
inicio = time.time()
clfSVC = svm.SVC(random_state=0, probability=True)
CV_SVC = cross_val_score(clfSVC, datos_tr.values, salida_esperada_tr.values, cv=5)
clfSVC.fit(datos_tr.values, salida_esperada_tr.values)
print('SVC cross-validation score sin optimizacion: ')
print('Accuracy of ' + str(round(CV_SVC.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_SVC.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

"""
param_grid_SVC = {
    'C': [1.1,1.5,2.5,10.5],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree' : [1, 2, 3],
    'probability': [True,False]
}

gridSVC = GridSearchCV(clfSVC, param_grid_SVC, return_train_score=True)
gridSVC.fit(datos_tr.values, salida_esperada_tr.values)
modelo= gridSVC.best_estimator_
print("el mejor modelo es {} ".format(modelo))
"""

inicio = time.time()
clfSVC2 = svm.SVC(C=2.5, random_state=0, probability=True,degree=1)
CV_SVC2 = cross_val_score(clfSVC2, datos_tr.values, salida_esperada_tr.values, cv=5)
clfSVC2.fit(datos_tr.values, salida_esperada_tr.values)
print('SVC cross-validation score con optimizacion: ')
print('Accuracy of ' + str(round(CV_SVC2.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_SVC2.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')
#################### RFC ###########################

inicio = time.time()
clfRFC = RandomForestClassifier(n_jobs=-1, random_state=0)
CV_RFC = cross_val_score(clfRFC, datos_tr.values, salida_esperada_tr.values, cv=5)
clfRFC.fit(datos_tr.values, salida_esperada_tr.values)
print('RFC cross-validation score sin optimizacion: ')
print('Accuracy of ' + str(round(CV_RFC.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_RFC.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

"""
param_grid_RFC = {
    'n_estimators': [2, 3, 13, 50,100,500],
    'max_features': ['auto','sqrt','log2'],
    'max_depth': [None,1,2,3,10],
    'criterion': ['gini','entropy']
}

gridRFC = GridSearchCV(clfRFC, param_grid_RFC, return_train_score=True)
gridRFC.fit(datos_tr.values, salida_esperada_tr.values)
modelo= gridRFC.best_estimator_
print("el mejor modelo es {} ".format(modelo))
"""
inicio = time.time()
clfRFC2 = RandomForestClassifier(max_depth=2, n_estimators=500, n_jobs=-1, random_state=0)
CV_RFC2 = cross_val_score(clfRFC2, datos_tr.values, salida_esperada_tr.values, cv=5)
clfRFC2.fit(datos_tr.values, salida_esperada_tr.values)
print('RFC cross-validation score con optimizacion: ')
print('Accuracy of ' + str(round(CV_RFC2.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_RFC2.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

#################### MLP ###########################
inicio = time.time()
clfMPL = MLPClassifier(hidden_layer_sizes=(2,), random_state=0, max_iter=10000)
CV_MPL = cross_val_score(clfMPL, datos_tr.values, salida_esperada_tr.values, cv=5)
clfMPL.fit(datos_tr.values, salida_esperada_tr.values)
print('MPL cross-validation score sin optimizacion: ')
print('Accuracy of ' + str(round(CV_MPL.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_MPL.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

"""
param_grid_MLP = {
    'max_iter': [100,1000],
    'solver': ['sgd', 'adam','lbfgs'],
    'learning_rate': ['constant', 'adaptive','invscaling'],
    'activation': ['identity', 'logistic','tanh','relu'],
    'learning_rate_init': [0.1, 0.2, 0.001]
}
gridMLP = GridSearchCV(clfMPL, param_grid_MLP, return_train_score=True)
gridMLP.fit(datos_tr.values, salida_esperada_tr.values)
print(gridMLP.best_params_)
"""

inicio = time.time()
clfMPL2 = MLPClassifier(hidden_layer_sizes=(2,), solver='adam', activation='logistic',
                        learning_rate='constant', learning_rate_init=0.1,
                        momentum=0.9, random_state=0, max_iter=1000)
CV_MPL2 = cross_val_score(clfMPL2, datos_tr.values, salida_esperada_tr.values, cv=5)
clfMPL2.fit(datos_tr.values, salida_esperada_tr.values)
print('MPL cross-validation score con optimizacion: ')
print('Accuracy of ' + str(round(CV_MPL2.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_MPL2.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

############# KNC ######################
inicio = time.time()
clfKNC = KNeighborsClassifier(n_jobs=-1)
CV_KNC = cross_val_score(clfKNC, datos_tr.values, salida_esperada_tr.values, cv=5)
clfKNC.fit(datos_tr.values, salida_esperada_tr.values)
print('KNC cross-validation score sin optimizacion: ')
print('Accuracy of ' + str(round(CV_KNC.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_KNC.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

"""
param_grid_KNC = {
    'n_neighbors': [1, 2, 3, 7, 13],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'weights': ['uniform', 'distance'],
    'leaf_size':[5,10,20,30,100],
    'p': [1,2]
}
gridKNC = GridSearchCV(clfKNC, param_grid_KNC, return_train_score=True)
gridKNC.fit(datos_tr.values, salida_esperada_tr.values)
print(gridKNC.best_params_)
"""

inicio = time.time()
clfKNC2 = KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=5, p=1, weights='uniform')
CV_KNC2 = cross_val_score(clfKNC2, datos_tr.values, salida_esperada_tr.values, cv=5)
clfKNC2.fit(datos_tr.values, salida_esperada_tr.values)
print('KNC cross-validation score con optimizacion: ')
print('Accuracy of ' + str(round(CV_KNC2.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_KNC2.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

############# Voting Ensembled Method Hard ######################
inicio = time.time()
clfVM = VotingClassifier(estimators=[
    ('RFC', clfRFC), ('MLP', clfMPL), ('KNC', clfKNC), ('SVC', clfSVC)], voting='hard')
CV_VM = cross_val_score(clfVM, datos_tr.values, salida_esperada_tr.values, cv=5)
clfVM.fit(datos_tr.values, salida_esperada_tr.values)
print('VM hard cross-validation score sin optimizacion: ')
print('Accuracy of ' + str(round(CV_VM.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_VM.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

inicio = time.time()
clfVM2 = VotingClassifier(estimators=[
    ('RFC', clfRFC2), ('MLP', clfMPL2), ('KNC', clfKNC2), ('SVC', clfSVC2)], voting='hard')
CV_VM2 = cross_val_score(clfVM2, datos_tr.values, salida_esperada_tr.values, cv=5)
clfVM2.fit(datos_tr.values, salida_esperada_tr.values)
print('VM hard cross-validation score con optimizacion: ')
print('Accuracy of ' + str(round(CV_VM2.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_VM2.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

############# Voting Ensembled Method Soft ######################
inicio = time.time()
clfVMS = VotingClassifier(estimators=[
    ('RFC', clfRFC), ('MLP', clfMPL), ('KNC', clfKNC),  ('SVC', clfSVC)], voting='soft')
CV_VMS = cross_val_score(clfVMS, datos_tr.values, salida_esperada_tr.values, cv=5)
clfVMS.fit(datos_tr.values, salida_esperada_tr.values)
print('VM soft cross-validation score sin optimizacion: ')
print('Accuracy of ' + str(round(CV_VMS.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_VMS.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

inicio = time.time()
clfVMS2 = VotingClassifier(estimators=[
    ('RFC', clfRFC2), ('MLP', clfMPL2), ('KNC', clfKNC2),  ('SVC', clfSVC2)], voting='soft')
CV_VMS2 = cross_val_score(clfVMS2, datos_tr.values, salida_esperada_tr.values, cv=5)
clfVMS2.fit(datos_tr.values, salida_esperada_tr.values)
print('VM soft cross-validation score con optimizacion: ')
print('Accuracy of ' + str(round(CV_VMS2.mean(), 2)) + ' with a standard deviation of ' + str(round(CV_VMS2.std(), 2)))
fin = time.time()
print('Tiempo de ejecucion ' + str(fin - inicio) + '\n')

y_pred_MPL2 = clfMPL2.predict_proba(datos_ts.values)
y_pred_RFC2 = clfRFC2.predict_proba(datos_ts.values)
y_pred_KNC2 = clfKNC2.predict_proba(datos_ts.values)
y_pred_SVC2 = clfSVC2.predict_proba(datos_ts.values)
y_pred_VMS2 = clfVMS2.predict_proba(datos_ts.values)


y_pred_MPL = clfMPL.predict_proba(datos_ts.values)
y_pred_RFC = clfRFC.predict_proba(datos_ts.values)
y_pred_KNC = clfKNC.predict_proba(datos_ts.values)
y_pred_SVC = clfSVC.predict_proba(datos_ts.values)
y_pred_VMS = clfVMS.predict_proba(datos_ts.values)

#Los predicta para la accuracy etc
y_predict_MPL2 = clfMPL2.predict(datos_ts.values)
y_predict_RFC2 = clfRFC2.predict(datos_ts.values)
y_predict_KNC2 = clfKNC2.predict(datos_ts.values)
y_predict_SVC2 = clfSVC2.predict(datos_ts.values)
y_predict_VMS2 = clfVMS2.predict(datos_ts.values)
y_predict_VMH2 = clfVM2.predict(datos_ts.values)

cm_rf=confusion_matrix(salida_esperada_ts,y_predict_RFC2)

print("Reporte de clasificacion de random forest optimizado: ")

print(classification_report(salida_esperada_ts,y_predict_RFC2))

print(classification_report(salida_esperada_ts,y_predict_SVC2))

print(classification_report(salida_esperada_ts,y_predict_MPL2))

print(classification_report(salida_esperada_ts,y_predict_KNC2))

print(classification_report(salida_esperada_ts,y_predict_VMH2))

print(classification_report(salida_esperada_ts,y_predict_VMS2))

# Calculamos el FPR , TPR y el area bajo la curva para hacer la ROC
fprMLP, tprMLP, thresholdsMLP = metrics.roc_curve(salida_esperada_ts, y_pred_MPL[:, 1])
roc_aucMLP = metrics.roc_auc_score(salida_esperada_ts, y_pred_MPL[:, 1])

fprMLP2, tprMLP2, thresholdsMLP2 = metrics.roc_curve(salida_esperada_ts, y_pred_MPL2[:, 1])
roc_aucMLP2 = metrics.roc_auc_score(salida_esperada_ts, y_pred_MPL2[:, 1])

fprVMS, tprVMS, thresholdsVMS = metrics.roc_curve(salida_esperada_ts, y_pred_VMS[:, 1])
roc_aucVMS = metrics.roc_auc_score(salida_esperada_ts, y_pred_VMS[:, 1])

fprVMS2, tprVMS2, thresholdsVMS2 = metrics.roc_curve(salida_esperada_ts, y_pred_VMS2[:, 1])
roc_aucVMS2 = metrics.roc_auc_score(salida_esperada_ts, y_pred_VMS2[:, 1])

fprSVC, tprSVC, thresholdsSVC = metrics.roc_curve(salida_esperada_ts, y_pred_SVC[:, 1])
roc_aucSVC = metrics.roc_auc_score(salida_esperada_ts, y_pred_SVC[:, 1])

fprSVC2, tprSVC2, thresholdsSVC2 = metrics.roc_curve(salida_esperada_ts, y_pred_SVC2[:, 1])
roc_aucSVC2 = metrics.roc_auc_score(salida_esperada_ts, y_pred_SVC2[:, 1])

fprRFC, tprRFC, thresholdsRFC = metrics.roc_curve(salida_esperada_ts, y_pred_RFC[:, 1])
roc_aucRFC = metrics.roc_auc_score(salida_esperada_ts, y_pred_RFC[:, 1])

fprRFC2, tprRFC2, thresholdsRFC2 = metrics.roc_curve(salida_esperada_ts, y_pred_RFC2[:, 1])
roc_aucRFC2 = metrics.roc_auc_score(salida_esperada_ts, y_pred_RFC2[:, 1])

fprKNC, tprKNC, thresholdsKNC = metrics.roc_curve(salida_esperada_ts, y_pred_KNC[:, 1])
roc_aucKNC = metrics.roc_auc_score(salida_esperada_ts, y_pred_KNC[:, 1])

fprKNC2, tprKNC2, thresholdsKNC2 = metrics.roc_curve(salida_esperada_ts, y_pred_KNC2[:, 1])
roc_aucKNC2 = metrics.roc_auc_score(salida_esperada_ts, y_pred_KNC2[:, 1])

labels = ['SVC', 'RFC', 'MLP', 'KNC', 'VMH', 'VMS']
mean_NO=[CV_SVC.mean(),CV_RFC.mean(),CV_MPL.mean(),CV_KNC.mean(), CV_VM.mean(), CV_VMS.mean()]
mean_O=[CV_SVC2.mean(),CV_RFC2.mean(),CV_MPL2.mean(),CV_KNC2.mean(),CV_VM2.mean(), CV_VMS2.mean()]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_NO, width, label='No optimizado')
rects2 = ax.bar(x + width/2, mean_O, width, label='Optimizado')
ax.set_ylabel('Acurracy media de CV 5-fold')
ax.set_title('Acurracy media por clasificador')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=3)
plt.show()


# Graficamos ROC
plt.plot([0, 1], [0, 1], color="k", linestyle=":")
plt.plot(fprMLP, tprMLP, label='MLP (AUC= ' + ' ' + str(round(roc_aucMLP, 3)) + ')', linestyle="--", color="g")
plt.plot(fprRFC, tprRFC, label='RFC (AUC= ' + ' ' + str(round(roc_aucRFC, 3)) + ')', linestyle="--", color="r")
plt.plot(fprKNC, tprKNC, label='KNC (AUC= ' + ' ' + str(round(roc_aucKNC, 3)) + ')', linestyle="--", color="y")
plt.plot(fprSVC, tprSVC, label='SVC (AUC= ' + ' ' + str(round(roc_aucSVC, 3)) + ')', linestyle="--", color="c")
plt.plot(fprVMS, tprVMS, label='VMS (AUC= ' + ' ' + str(round(roc_aucVMS, 3)) + ')', linestyle="--", color="lime")
plt.ylabel("True Positive Rate o Sensibilidad")
plt.xlabel("False Positive Rate 1 - Especificidad")
plt.legend()

plt.figure()
plt.plot([0, 1], [0, 1], color="k", linestyle=":")
plt.plot(fprMLP2, tprMLP2, label='MLP con op (AUC= ' + ' ' + str(round(roc_aucMLP2, 3)) + ')', color="g")
plt.plot(fprRFC2, tprRFC2, label='RFC con op  (AUC= ' + ' ' + str(round(roc_aucRFC2, 3)) + ')', color="r")
plt.plot(fprKNC2, tprKNC2, label='KNC con op  (AUC= ' + ' ' + str(round(roc_aucKNC2, 3)) + ')', color="y")
plt.plot(fprSVC2, tprSVC2, label='SVC con op  (AUC= ' + ' ' + str(round(roc_aucSVC2, 3)) + ')', color="c")
plt.plot(fprVMS2, tprVMS2, label='VMS con op  (AUC= ' + ' ' + str(round(roc_aucVMS2, 3)) + ')', color="lime")
plt.ylabel("True Positive Rate o Sensibilidad")
plt.xlabel("False Positive Rate o 1 - Especificidad")
plt.legend()
plt.show()