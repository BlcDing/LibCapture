import random
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

##########
##########
    
TRAINTESTBOUNDARY = 0.7
PICKLE_NAME = "dataset3.p"

print "Loading " + PICKLE_NAME + "..."
flowlist = pickle.load(open(PICKLE_NAME, "rb"))
print "Done..."
print ""

print "Flows loaded: " + str(len(flowlist))

a = []
p = []
r = []
f = []

for i in range(1):
    ########## PREPARE STUFF
    examples = []
    trainingexamples = []
    testingexamples = []

    #classifier = svm.SVC(gamma=0.001, C=100, probability=True)
    classifier = ensemble.RandomForestClassifier()


    ########## GET FLOWS
    for package, flow in flowlist:
        examples.append((flow, package))
    print ""


    ########## SHUFFLE DATA to ensure classes are "evenly" distributed
    random.shuffle(examples)


    ########## TRAINING
    trainingexamples = examples[:int(TRAINTESTBOUNDARY * len(examples))]

    X_train = []
    y_train = []

    for flow, package in trainingexamples:       
        X_train.append(flow)
        y_train.append(package)

    print "Fitting classifier..."
    classifier.fit(X_train, y_train)
    print "Classifier fitted!"
    print ""

            
    ########## TESTING
    counter = 0
    correct = 0
        
    testingexamples = examples[int(TRAINTESTBOUNDARY * len(examples)):]

    X_test = []
    y_test = []
    y_pred = []

    for flow, package in testingexamples:   
        X_test.append(flow)
        y_test.append(package)

    #####

    y_pred = classifier.predict(X_test)
    y_score = classifier.predict_proba(X_test)

    print(accuracy_score(y_test, y_pred))
    print(precision_score(y_test, y_pred, average="macro"))
    print(recall_score(y_test, y_pred, average="macro"))
    print(f1_score(y_test, y_pred, average="macro"))
    print ""

    a.append(accuracy_score(y_test, y_pred))
    p.append(precision_score(y_test, y_pred, average="macro"))
    r.append(recall_score(y_test, y_pred, average="macro"))
    f.append(f1_score(y_test, y_pred, average="macro"))

 
print np.mean(a)
print np.mean(p)
print np.mean(r)
print np.mean(f)



#### Confusion Matrix
classes = range(0, 5)
confusion = confusion_matrix(y_pred, y_test)
plt.figure(figsize=(20, 20))
plt.imshow(confusion, cmap=plt.cm.Blues)
plt.title("Confusion Matrix", fontsize=35, pad=40)

indices = range(len(confusion))
plt.xticks(indices, classes, size=30)
plt.yticks(indices, classes, size=30)
plt.tick_params(width=4, length=12)
cbar = plt.colorbar(shrink=0.8)
cbar.ax.tick_params(labelsize="30")

ax = plt.gca()
ax.spines["bottom"].set_linewidth(4)
ax.spines["left"].set_linewidth(4)
ax.spines["right"].set_linewidth(4)
ax.spines["top"].set_linewidth(4)

plt.xlabel("Predicted Value", fontsize=30, labelpad=40)
plt.ylabel("True Value", fontsize=30, labelpad=40)

for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index], fontsize=30, verticalalignment = "center",horizontalalignment = "center")
 
plt.savefig("test.png", bbox_inches="tight", pad_inches=0.0)



#### ROC Curve
y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
n_classes = y_test.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# Plot all ROC curves
plt.figure(figsize=(20, 15))
ax = plt.gca()
ax.spines["bottom"].set_linewidth(4)
ax.spines["left"].set_linewidth(4)
ax.spines["right"].set_linewidth(4)
ax.spines["top"].set_linewidth(4)
plt.tick_params(width=4, length=12, labelsize=30)

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=8)
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=8)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, linewidth=8,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=30, labelpad=40)
plt.ylabel('True Positive Rate', fontsize=30, labelpad=40)
plt.title('Some extension of Receiver operating characteristic to multi-class', fontsize=35, pad=40)
plt.legend(loc="lower right", fontsize=30)
plt.savefig("test2.png", bbox_inches="tight", pad_inches=0.0)