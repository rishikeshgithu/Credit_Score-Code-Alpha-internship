import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import os

# 2. Read Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 3. Remove Bank Client ID Feature
train = train.drop(columns=['X'])
test = test.drop(columns=['X'])

# 4. Missing Value(s) Inspection
train.dropna(inplace=True)
test.dropna(inplace=True)

# 5. Data Types Conversion
numeric_columns = ['jumlah_kartu', 'outstanding', 'skor_delikuensi']
train[numeric_columns] = train[numeric_columns].apply(pd.to_numeric, errors='coerce')
test[numeric_columns] = test[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values after conversion
train.dropna(inplace=True)
test.dropna(inplace=True)

# Encode categorical target variable
label_encoder = LabelEncoder()
train['flag_kredit_macet'] = label_encoder.fit_transform(train['flag_kredit_macet'])

# 6. Standardize All Numerical Feature(s)
scaler = StandardScaler()
numeric_columns = train.select_dtypes(include=['float64']).columns
train[numeric_columns] = scaler.fit_transform(train[numeric_columns])
numeric_columns = test.select_dtypes(include=['float64']).columns
test[numeric_columns] = scaler.transform(test[numeric_columns])

# 7. Apply SMOTE for Nominal and Numerical Feature(s)
X_train = train.drop(columns=['flag_kredit_macet'])
y_train = train['flag_kredit_macet']
smote = SMOTE(sampling_strategy='auto', random_state=2020, n_jobs=-1)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Reconstruct the balanced_train DataFrame
balanced_train = pd.concat([pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name='flag_kredit_macet')], axis=1)

# 8. Split Train Data into Learning Data and Validation Data
learning, validation = train_test_split(balanced_train, test_size=0.3, random_state=2020)

# 9. Construct Logistic Regression Model
logmodel = LogisticRegression(random_state=2020)
logmodel.fit(learning.drop(columns=['flag_kredit_macet']), learning['flag_kredit_macet'])
prob_logpred = logmodel.predict_proba(validation.drop(columns=['flag_kredit_macet']))[:, 1]
logpred = pd.Series(prob_logpred > 0.5, name='flag_kredit_macet').map({True: "1", False: "0"})
print(pd.crosstab(logpred, validation['flag_kredit_macet']))
print("F1 Score:", f1_score(y_pred=logpred, y_true=validation['flag_kredit_macet'], pos_label=1, average='binary'))

# 10. Construct Support Vector Machine Model
svm_model = SVC(random_state=2020)
svm_model.fit(learning.drop(columns=['flag_kredit_macet']), learning['flag_kredit_macet'])
svm_pred = svm_model.predict(validation.drop(columns=['flag_kredit_macet']))
print(pd.crosstab(svm_pred, validation['flag_kredit_macet']))
print("F1 Score:", f1_score(validation['flag_kredit_macet'], svm_pred, pos_label='1', average='binary'))

# 11. Construct Decision Tree Model
tree_model = DecisionTreeClassifier(random_state=2020)
tree_model.fit(learning.drop(columns=['flag_kredit_macet']), learning['flag_kredit_macet'])
tree_pred = tree_model.predict(validation.drop(columns=['flag_kredit_macet']))
print(pd.crosstab(tree_pred, validation['flag_kredit_macet']))
print("F1 Score:", f1_score(validation['flag_kredit_macet'], tree_pred, pos_label='1', average='binary'))

# 12. Construct Nearest Neighbour Model
nnmodel = KNeighborsClassifier(n_neighbors=5)
nnmodel.fit(learning.drop(columns=['flag_kredit_macet']), learning['flag_kredit_macet'])
nnpred = nnmodel.predict(validation.drop(columns=['flag_kredit_macet']))
print(pd.crosstab(nnpred, validation['flag_kredit_macet']))
print("F1 Score:", f1_score(validation['flag_kredit_macet'], nnpred, pos_label='1', average='binary'))

# 13. Predicting "Default" Feature on Test Data
test_logpred = pd.Series(logmodel.predict(test.drop(columns=['flag_kredit_macet'])), name='flag_kredit_macet').map({True: "1", False: "0"})
test_svm_pred = pd.Series(svm_model.predict(test.drop(columns=['flag_kredit_macet'])), name='flag_kredit_macet')
test_tree_pred = pd.Series(tree_model.predict(test.drop(columns=['flag_kredit_macet'])), name='flag_kredit_macet').map({True: "1", False: "0"})
test_nnpred = pd.Series(nnmodel.predict(test.drop(columns=['flag_kredit_macet'])), name='flag_kredit_macet')

# 14. Best Model Selection (Best model, logically, means the model with the highest F1 Score)
print("Logistic Regression:")
print(pd.crosstab(test_logpred, test['flag_kredit_macet']))
print("Support Vector Machine:")
print(pd.crosstab(test_svm_pred, test['flag_kredit_macet']))
print("Decision Tree:")
print(pd.crosstab(test_tree_pred, test['flag_kredit_macet']))
print("Nearest Neighbour:")
print(pd.crosstab(test_nnpred, test['flag_kredit_macet']))

# 15. Clear Console and All Variables in The Global Environment
os.system('clear' if os.name == 'posix' else 'cls')