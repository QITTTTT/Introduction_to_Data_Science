import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 读取文件
file = pd.read_csv('train.csv')
file_2 = pd.read_excel('noisy ID1.xlsx')
file_5 = pd.read_excel('test1.xlsx')
file_6 = pd.read_excel('test2.xlsx')

# 任务一
file_1 = file.copy()
print(file_1)
file_1['Contacts_Count_12_mon'] = file_1['Contacts_Count_12_mon'].fillna('3')  # 众数填充
file_1['Gender'] = file_1['Gender'].fillna(method='bfill')  # Gender填充
file_1['Card_Category'] = file_1['Card_Category'].fillna('Red')  # 众数填充
print(file_1.isnull().any(axis=1))
file_1_2 = file_1.loc[~(file.isna().any(axis=1))]  # 没有缺失值的行
file_1_1 = file_1.copy()
file_1_1.fillna('Unknown')

# 特征转换-独热编码
file_1_1.Gender = file_1_1.Gender.replace({'F': 1, 'M': 0})
file_1_1 = pd.concat([file_1_1, pd.get_dummies(file_1['Education_Level']).drop(columns=['Unknown'])], axis=1)
file_1_1 = pd.concat([file_1_1, pd.get_dummies(file_1['Income_Category']).drop(columns=['Unknown'])], axis=1)
file_1_1 = pd.concat([file_1_1, pd.get_dummies(file_1['Marital_Status'])], axis=1)
file_1_1 = pd.concat([file_1_1, pd.get_dummies(file_1['Card_Category'])], axis=1)
file_1_1 = pd.concat([file_1_1, pd.get_dummies(file_1['Dependent_count'])], axis=1)
file_1_1 = pd.concat([file_1_1, pd.get_dummies(file_1['Total_Relationship_Count'])], axis=1)
file_1_1.drop(columns=['Education_Level', 'Income_Category', 'Marital_Status', 'Card_Category', 'Dependent_count',
                       'Total_Relationship_Count'], inplace=True)

# print(file_1_1.isnull().sum())  # 缺失值统计
# print(file_1_1)
# file_1_4 = file_1_1[file_1_1.Contacts_Count_12_mon.isna()]  # 有缺失值的行
# file_1_3 = file_1_1[file_1_1.Contacts_Count_12_mon.notna()]  # 没有缺失值的行
# print(file_1_3)
# print(file_1_4)

# 对于Contacts_Count_12_mon 使用模型模型准确度只有36%，与众数占比基本吻合，所以选择众数填充
# file_3 = file_1_3.drop(['Contacts_Count_12_mon', 'ID', 'Attrition_Flag'], axis=1)  # 其余特征
# print(file_3)
# X, y = file_3, file_1_3[['Contacts_Count_12_mon']]
# mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10],
#       max_iter=1000000000000000, activation='tanh')
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
# mlp.fit(X_train, y_train.values.ravel())
# print(mlp.score(X_test, y_test))

# 热力图
plt.figure(figsize=(40, 10))
cor = file_1_1.drop(["ID"], axis=1).corr(numeric_only=True)
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, annot_kws={'size': 4})
plt.show()

# 计算出与因变量之间的相关性
cor_target = abs(cor["Attrition_Flag"])
# 抛掉小于0.01的相关性系数
relevant_features = cor_target[cor_target < 0.01]
# 计算属性之间的相关性
print(relevant_features)
print(file_1_1[["Gender", "Contacts_Count_12_mon"]].corr(numeric_only=False))
print("=" * 50)
print(file_1_1[["Contacts_Count_12_mon", "Contacts_Count_13_mon"]].corr(numeric_only=False))
print("=" * 50)
print(file_1_1[["Total_Trans_Ct", "Total_Trans_Amt"]].corr())
print("=" * 50)
print(file_1_1[["Months_Inactive_12_mon", "Contacts_Count_12_mon"]].corr(numeric_only=False))

# 任务二
file_1_5 = file_1_1.copy()
for i in range(0, 7599):
    if file_1_5['ID'][i] in file_2.values:
        if file_1_5.loc[i, 'Attrition_Flag'] == 0:
            file_1_5.loc[i, 'Attrition_Flag'] = 1
        elif file_1_5['Attrition_Flag'][i] == 1:
            file_1_5.loc[i, 'Attrition_Flag'] = 0
# file_1_5是将标签更改完的file_1_1
# 验证修改效果
# print(file_1_5['Attrition_Flag'][8])
# print(file_1_5['ID'][8])

file_4 = file_1_5.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                        'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                        'Post-Graduate', 'Gray', 'Purple', 'Rainbow', 'Red', 'Total_Trans_Amt'], axis=1)  # 其余特征

X, y = file_4, file_1_5[['Attrition_Flag']]

# 决策树模型用来筛选错误特征
tree_1 = DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=100, min_samples_split=2,
                                max_leaf_nodes=625)
tree_1.fit(X, y.values.ravel())
print(tree_1.score(X, y.values.ravel()))
print('训练好的模型在原始数据上的准确率:{}'.format(tree_1.score(file_1_1.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                                                               'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy',
                                                               'Total_Amt_Chng_Q4_Q1',
                                                               'Post-Graduate', 'Gray', 'Purple', 'Rainbow', 'Red',
                                                               'Total_Trans_Amt'], axis=1),
                                                file_1_1[['Attrition_Flag']])))
pre = tree_1.predict(file_1_1.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                                    'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                                    'Post-Graduate', 'Gray', 'Purple', 'Rainbow', 'Red', 'Total_Trans_Amt'], axis=1))
print(X)
print('决策树的深度:{}'.format(tree_1.get_depth()))
# 定义数组下标
index = np.arange(0, 7600)
# 找到两个数组不相等元素的下标位置
ID = file_1_1['ID'][index[file_1_1['Attrition_Flag'] != pre]]
# pd.DataFrame(ID.values).to_csv(r"C:\Users\HP\Desktop\错误标签.csv")
x = 0
for i in ID.values:
    for j in file_2.values:
        if i == j:
            x += 1
print('已知的 380 个错误标签样本 ID 的检测准确率:{:.2f}%'.format(x / 380 * 100))

# 修改噪声标签
file_1_6 = file_1_1.copy()
words = ID.values
for i in range(0, 7599):
    if file_1_6['ID'][i] in words:
        if file_1_6.loc[i, 'Attrition_Flag'] == 0:
            file_1_6.loc[i, 'Attrition_Flag'] = 1
        elif file_1_6['Attrition_Flag'][i] == 1:
            file_1_6.loc[i, 'Attrition_Flag'] = 0
# file_1_6是将错误标签改正的file_1_1

file_1_7 = file_1_6.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                          'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                          'Post-Graduate', 'Gray', 'Purple', 'Rainbow', 'Red', 'Total_Trans_Amt'], axis=1)
X_0, y_0 = file_1_7, file_1_6[['Attrition_Flag']]
# 任务三
# 编码test1.xlsx
file_5.Gender = file_5.Gender.replace({'F': 1, 'M': 0})
file_5 = pd.concat([file_5, pd.get_dummies(file_5['Education_Level']).drop(columns=['Unknown'])], axis=1)
file_5 = pd.concat([file_5, pd.get_dummies(file_5['Income_Category']).drop(columns=['Unknown'])], axis=1)
file_5 = pd.concat([file_5, pd.get_dummies(file_5['Marital_Status'])], axis=1)
file_5 = pd.concat([file_5, pd.get_dummies(file_5['Card_Category'])], axis=1)
file_5 = pd.concat([file_5, pd.get_dummies(file_5['Dependent_count'])], axis=1)
file_5 = pd.concat([file_5, pd.get_dummies(file_5['Total_Relationship_Count'])], axis=1)
file_5.drop(columns=['Education_Level', 'Income_Category', 'Marital_Status', 'Card_Category', 'Dependent_count',
                     'Total_Relationship_Count'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_0, y_0, stratify=y, random_state=0)
# 神经网络
mlp = MLPClassifier(solver='sgd', random_state=1, hidden_layer_sizes=[20, 20, 20],
                    max_iter=1000, activation='logistic', alpha=0.001, learning_rate_init=0.5, batch_size=4)

mlp.fit(X_train, y_train.values.ravel())
print('神经网络在测试集上的准确率:{}'.format(mlp.score(X_test, y_test.values.ravel())))
print('神经网络在test1.xlsx上的准确率:{}'.format(mlp.score(file_5.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                                                              'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy',
                                                              'Total_Amt_Chng_Q4_Q1',
                                                              'Post-Graduate', 'Purple', 'Rainbow', 'Red',
                                                              'Total_Trans_Amt'], axis=1)
                                                 , file_5[['Attrition_Flag']].values.ravel())))
# 决策树
tree_2 = DecisionTreeClassifier(random_state=40, criterion="gini", max_depth=10000, min_samples_split=2,
                                max_leaf_nodes=625)
tree_2.fit(X_train, y_train.values.ravel())
print('决策树在测试集上的准确率:{}'.format(tree_2.score(X_test, y_test.values.ravel())))
print('决策树在test1.xlsx上的准确率:{}'.format(tree_2.score(file_5.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                                                              'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy',
                                                              'Total_Amt_Chng_Q4_Q1',
                                                              'Post-Graduate', 'Purple', 'Rainbow', 'Red',
                                                              'Total_Trans_Amt'], axis=1)
                                                 , file_5[['Attrition_Flag']].values.ravel())))
# Adaboost
Ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=60, learning_rate=0.1)
Ada.fit(X_train, y_train.values.ravel())
print('Adaboost在测试集上的准确率:{}'.format(Ada.score(X_test, y_test.values.ravel())))
print('Adaboost在test1.xlsx上的准确率:{}'.format(
    Ada.score(file_5.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                           'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                           'Post-Graduate', 'Purple', 'Rainbow', 'Red', 'Total_Trans_Amt'], axis=1)
              , file_5[['Attrition_Flag']].values.ravel())))

# 随机森林
rfMod = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=5,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='log2',
                               max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None,
                               verbose=0)

rfMod.fit(X_train, y_train.values.ravel())
print('随机森林在测试集上的准确率:{}'.format(rfMod.score(X_test, y_test.values.ravel())))
print(
    '随机森林在test1.xlsx上的准确率:{}'.format(rfMod.score(file_5.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                                                              'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy',
                                                              'Total_Amt_Chng_Q4_Q1',
                                                              'Post-Graduate', 'Purple', 'Rainbow', 'Red',
                                                              'Total_Trans_Amt'], axis=1)
                                                 , file_5[['Attrition_Flag']].values.ravel())))

# SVM
scaler = MinMaxScaler(feature_range=(-1, 1))
SVM = SVC(C=1.0, kernel='linear', max_iter=10000, random_state=0)
SVM.fit(pd.DataFrame(scaler.fit_transform(X_train)).values, y_train.values.ravel())
print('SVM在测试集上的准确率:{}'.format(SVM.score(pd.DataFrame(scaler.fit_transform(X_test)).values, y_test.values.ravel())))
print('SVM在text.xlsx上的准确率:{}'.format(
    SVM.score(pd.DataFrame(scaler.fit_transform(file_5.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                                                             'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy',
                                                             'Total_Amt_Chng_Q4_Q1',
                                                             'Post-Graduate', 'Purple', 'Rainbow', 'Red',
                                                             'Total_Trans_Amt'], axis=1))).values
              , file_5[['Attrition_Flag']].values.ravel())))

# 将test2进行编码
file_6.Gender = file_6.Gender.replace({'F': 1, 'M': 0})
file_6 = pd.concat([file_6, pd.get_dummies(file_6['Education_Level']).drop(columns=['Unknown'])], axis=1)
file_6 = pd.concat([file_6, pd.get_dummies(file_6['Income_Category']).drop(columns=['Unknown'])], axis=1)
file_6 = pd.concat([file_6, pd.get_dummies(file_6['Marital_Status'])], axis=1)
file_6 = pd.concat([file_6, pd.get_dummies(file_6['Card_Category'])], axis=1)
file_6 = pd.concat([file_6, pd.get_dummies(file_6['Dependent_count'])], axis=1)
file_6 = pd.concat([file_6, pd.get_dummies(file_6['Total_Relationship_Count'])], axis=1)
file_6.drop(columns=['Education_Level', 'Income_Category', 'Marital_Status', 'Card_Category', 'Dependent_count',
                     'Total_Relationship_Count'], inplace=True)
pre_2 = mlp.predict(file_6.drop(['ID', 'Attrition_Flag', 'Customer_Age', 'Months_on_book',
                                 'Credit_Limit', 'Avg_Open_To_Buy', 'Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                                 'Post-Graduate', 'Purple', 'Rainbow', 'Red', 'Total_Trans_Amt'], axis=1))

file_6_1 = file_6.copy()
for i in range(0, 1000):
    file_6_1.loc[i, "Attrition_Flag"] = pre_2[i]
# pd.DataFrame(file_6_1).to_csv(r"C:\Users\HP\Desktop\测试集2上的预测结果.csv")  # 导出预测结果
