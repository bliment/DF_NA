import numpy as np
import pandas as pd
from sklearn import preprocessing
from osgeo import gdal
import itertools
from utils_predict_model import extract_ras_name, get_ras_data, get_train_test_data, grid_search_model, plot_roc
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
from sklearn.metrics import confusion_matrix

input_lyr_label = r"C:\Users\Dong\Desktop\DU\label\label.tif"
input_lyr_g = r'C:\Users\Dong\Desktop\DU\distance\geo.tif'
input_lyr_f = r'C:\Users\Dong\Desktop\DU\distance\fault.tif'
input_lyr_mask = r"C:\Users\Dong\Desktop\DU\mask.tif"
input_lyr_euclidean_dis_one = r'C:\Users\Dong\Desktop\DU\distance\fault_dis.tif'
input_lyr_euclidean_dis_two = r'C:\Users\Dong\Desktop\DU\distance\g_dis.tif'
input_lyr_element_1 = r'C:\Users\Dong\Desktop\DU\sa\al\S_A4_back.tif'
# input_lyr_element_2 = r'C:\Users\Dong\Desktop\T77\11_3组会使用\usedata\LSA\BA.tif'
# input_lyr_element_3 = r'C:\Users\Dong\Desktop\T77\data\sa\ca\S_A4_back.tif'
input_lyr_element_4 = r'C:\Users\Dong\Desktop\DU\sa\co\S_A4_back.tif'
input_lyr_element_5 = r'C:\Users\Dong\Desktop\DU\sa\cr\S_A4_back.tif'
input_lyr_element_6 = r'C:\Users\Dong\Desktop\DU\sa\cu\S_A4_back.tif'
input_lyr_element_7 = r'C:\Users\Dong\Desktop\DU\sa\fe\S_A4_back.tif'
# input_lyr_element_8 = r'C:\Users\Dong\Desktop\T77\11_3组会使用\usedata\LSA\LA.tif'
input_lyr_element_9 = r'C:\Users\Dong\Desktop\DU\sa\mg\S_A4_back.tif'
# input_lyr_element_10 = r'C:\Users\Dong\Desktop\T77\11_3组会使用\usedata\LSA\MN.tif'
# input_lyr_element_11 = r'C:\Users\Dong\Desktop\T77\data\sa\na\S_A4_back.tif'
input_lyr_element_12 = r'C:\Users\Dong\Desktop\DU\sa\ni\S_A4_back.tif'
# input_lyr_element_13 = r'C:\Users\Dong\Desktop\T77\data\sa\pb\S_A4_back.tif'
input_lyr_element_14 = r'C:\Users\Dong\Desktop\DU\sa\sc\S_A4_back.tif' #钪
# input_lyr_element_15 = r'C:\Users\Dong\Desktop\T77\data\sa\sr\S_A4_back.tif' #锶
# input_lyr_element_16 = r'C:\Users\Dong\Desktop\T77\11_3组会使用\usedata\LSA\TH.tif' #钍
# input_lyr_element_17 = r'C:\Users\Dong\Desktop\T77\11_3组会使用\usedata\LSA\TI.tif' #钛
input_lyr_element_18 = r'C:\Users\Dong\Desktop\DU\sa\v\S_A4_back.tif' #钒
# input_lyr_element_19 = r'C:\Users\Dong\Desktop\T77\11_3组会使用\usedata\LSA\Y.tif' #钇
input_lyr_element_20 = r'C:\Users\Dong\Desktop\DU\sa\zn\S_A4_back.tif'
grv=r'C:\Users\Dong\Desktop\DU\grv.tif'
input_lyr_fc = r'C:\Users\Dong\Desktop\DU\distance\Line.tif'

# params_list=(input_lyr_euclidean_dis_one,input_lyr_euclidean_dis_two,
#             input_lyr_element_one, input_lyr_element_two,
#             input_lyr_element_three, input_lyr_element_four,
#             input_lyr_element_five, input_lyr_sa_ano, input_lyr_sa_back,input_lyr_element1,input_lyr_element2,
#              input_lyr_element3,input_lyr_element4,input_lyr_element5,input_lyr_element6,input_lyr_element7,
#              input_lyr_element8,input_lyr_element9,input_lyr_element10,input_lyr_element11,input_lyr_element12,
#              input_lyr_element13,input_lyr_label, input_lyr_mask)
#
# params_list=(input_lyr_mask,input_lyr_label,input_lyr_euclidean_dis_two,
#              input_lyr_element_1,input_lyr_element_2,input_lyr_element_3,input_lyr_element_4,input_lyr_element_5,
#              input_lyr_element_6,input_lyr_element_7,input_lyr_element_8,input_lyr_element_9,input_lyr_element_10,
#              input_lyr_element_11,input_lyr_element_12,input_lyr_element_13,input_lyr_element_14,input_lyr_element_15,
#              input_lyr_element_16,input_lyr_element_17,input_lyr_element_18,input_lyr_element_19,input_lyr_element_20,
#              grv
#              )
params_list=(input_lyr_mask,input_lyr_label, input_lyr_euclidean_dis_one,
             input_lyr_euclidean_dis_two,
             input_lyr_g,
             input_lyr_f,
             input_lyr_element_1,input_lyr_element_4,input_lyr_element_5,
             input_lyr_element_6,input_lyr_element_7,
             input_lyr_element_9,input_lyr_element_12,input_lyr_element_14,input_lyr_element_18,input_lyr_element_20,
             grv,input_lyr_fc
             )

input_lyr_list = [i for i in params_list]
input_lyr, input_lyr_name = extract_ras_name(input_lyr_list)
input_layers_array = list(gdal.Open(i) for i in input_lyr)
ds = input_layers_array[-1]

input_layers_array = np.array(list(map(lambda i: i.GetRasterBand(1).ReadAsArray(), input_layers_array)))

input_layers_array = np.swapaxes(input_layers_array, 0, 2)
input_layers_array = np.swapaxes(input_layers_array, 0, 1)
dimension = input_layers_array.shape
input_layers_array = input_layers_array.reshape((dimension[0] * dimension[1], dimension[2]))

# print(dimension)
input_layers_pd = pd.DataFrame(input_layers_array, columns=input_lyr_name)

ma_t_f = np.where(input_layers_pd['Mask'] == 1, True, False)
x_label_mask_pd = input_layers_pd[input_layers_pd['Mask'] == 1].copy()
# label_t_f = np.where(x_label_mask_pd['Label'] == 1, True, False)
label_t = np.where(x_label_mask_pd['Label'] == 1, True, False)
label_f = np.where((x_label_mask_pd['Label'] != 1) & ((x_label_mask_pd['Geo'] != 1) &
                   (x_label_mask_pd['Fault'] != 1)), True, False)
# label_f = np.where((x_label_mask_pd['Label'] != 1) , True, False)
x_pd = x_label_mask_pd.drop(labels=['Mask', "Label",'Geo','Fault'], axis=1)
x_sc_np = preprocessing.StandardScaler().fit_transform(x_pd)

# print(x_sc_np.shape[0])
# x_t_np = x_sc_np[label_t_f, :]
# x_f_np = x_sc_np[~label_t_f, :]
x_t_np = x_sc_np[label_t, :]
x_f_np = x_sc_np[label_f, :]
print(x_t_np.shape)
print(x_f_np.shape)
x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x_t_np, np.ones((x_t_np.shape[0], 1)),
                                                                test_size=0.2, random_state=45)
x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(x_f_np, np.zeros((x_f_np.shape[0], 1)),
                                                                test_size=0.2, random_state=45)
# print(x_train_f.shape[0],x_train_t.shape[0])
# print(x_test_f.shape[0],x_test_t.shape[0])

np.random.shuffle(x_train_f)
np.random.shuffle(x_test_f)
ratio = 2
x_train_f = x_train_f[:x_train_t.shape[0] * ratio, :]
y_train_f = y_train_f[:y_train_t.shape[0] * ratio, :]
# print(x_train_f.shape[0],x_train_t.shape[0])
x_test_f = x_test_f[:x_test_t.shape[0] * ratio, :]
y_test_f = y_test_f[:y_test_t.shape[0] * ratio, :]
# print(x_test_f.shape[0],x_test_t.shape[0])
x_tr = np.vstack((x_train_t, x_train_f))
y_tr = np.vstack((y_train_t, y_train_f))
print(x_train_t.shape,x_train_f.shape)
x_te = np.vstack((x_test_t, x_test_f))
y_te = np.vstack((y_test_t, y_test_f))
print(x_test_t.shape,x_test_f.shape)
print("Size of training set:{}, size of testing set:{}".format(x_tr.shape[0], x_te.shape[0]))
x_tr=x_tr.reshape(x_tr.shape[0],-1)
y_tr = y_tr.ravel()

from imblearn.over_sampling import SMOTE
# x_tr, y_tr=SMOTE().fit_resample(x_te, y_te)
# x_te, y_te=SMOTE().fit_resample(x_tr, y_tr)
print(x_tr.shape)
print(y_tr.shape)
print(x_te.shape)
print(y_te.shape)
model = CascadeForestClassifier (n_estimators=4,n_trees=100,max_depth=8,min_samples_split =20,min_samples_leaf=10,
                                 max_layers=3,backend='sklearn',verbose=1, random_state=90)
model.fit(x_tr, y_tr)   # Fitting estimator
y = model.predict(x_te)  # Evaluating estimator
y_predict = model.predict_proba(x_sc_np)[:, 1]

acc = accuracy_score(y_te, y) * 100
print("\nTesting Accuracy: {:.3f} %".format(acc))

importance= model.get_layer_feature_importances(0)
# # print(input_lyr_name)
print(importance)
print('importance_0')
importance= model.get_layer_feature_importances(1)
# print(input_lyr_name)
print(importance)
print('importance_1')
importance= model.get_layer_feature_importances(2)
# print(input_lyr_name)
print(importance)
print('importance_2')

# 重要性排序
# input_lyr_name = ["GEO","AL","CA","CU","FE","NA","NI","PB","SC","SR"]
# plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
# plt.rcParams['font.size'] = 12  # 字体大小
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# def plot_feature_importances(feature_importances, title, feature_names, change):
#     print('feature_importances', feature_importances)
#     print('names', feature_names)
#     #     feature_importances = 100.0*(feature_importances/max(feature_importances))
#
#     #     print('feature_importances',feature_importances)
#     #     将得分从小到大排序
#     index_sorted = np.argsort(feature_importances)
#     print('index_sorted', index_sorted)
#     print()
#     # 特征名称排序
#     chara = []
#     i=len(feature_names)
#     for col in range(0, i):
#         # print(feature_names)
#         value=(feature_names[index_sorted[col]])
#         chara.append(value)
#         print(chara[col])
#     print(chara)
#
#     #     让y坐标轴上的标签居中显示
#     pos = np.arange(index_sorted.shape[0]) + 0.5
#     # print(pos)
#     plt.figure(figsize=(16, 16))
#     # 0.9的分割数据
#     index1 = [i] * i
#     index2 = [i] * i
#     feature_importances = np.append(feature_importances, 0)
#
#     print('feature_importances', feature_importances)
#     sum = 0
#
#     for col in range(0, i):
#         k = feature_importances[index_sorted[i - col - 1]]
#         sum = sum + k
#         index1[col] = index_sorted[i - col - 1]
#         if (sum >= 0.6):
#             break
#
#     s = 0
#     for col in range(0, i):
#         k = feature_importances[index_sorted[col]]
#         print(k)
#         s = s + k
#         index2[col] = index_sorted[col]
#         if (s >= 0.4):
#             break
#
#     print('小于0.1', index2)
#     index1 = np.flipud(index1)
#     print('大于0.9', index1)
#
#     plt.barh(pos, feature_importances[index2], align='center')
#     plt.barh(pos, feature_importances[index1], align='center', color="red")
#     plt.yticks(pos, chara)
#     plt.xlabel('Relative Importance')
#     plt.title(title)
#
#     xlabel = feature_importances[index_sorted]
#     ylabel = pos
#     for x1, y1 in zip(xlabel, ylabel):
#         # 添加文本时添加偏移量使其显示更加美观
#         x1 = np.around(x1, decimals=3)
#         #         print("坐标",y1,x1)
#         plt.text(x1 + 0.00005, y1, '%.3f' % x1)
#
#     plt.show()
# plot_feature_importances(importance,'特征重要度排序',input_lyr_name,input_lyr_name)
# 网格搜索
# param_grid =[
#     {'n_estimators':[2,4,5,8],'n_trees':[15,30,45,60],"n_tolerant_rounds":[4,6,8,10]}]
#
#
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(CascadeForestClassifier( backend='sklearn'),
#                     param_grid=param_grid,cv=5)  # 默认是cv=3，即3折交叉验证
#
# grid.fit(x_tr, y_tr)
# print('Best：%f using %s' % (grid.best_score_, grid.best_params_))
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):#混淆矩阵画图

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_tr, model.predict(x_tr))  # 计算混淆矩阵
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')  # 绘制混淆矩阵
np.set_printoptions(precision=2)
print('Accuracy:', (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
                    cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[0, 0] + cnf_matrix[1, 0]))
print('Recall:', cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0]))
print('Precision:', cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[0, 1]))
print('Specificity:', cnf_matrix[0, 0] / (cnf_matrix[0, 1] + cnf_matrix[0, 0]))
cnf_matrix = confusion_matrix(y_te, model.predict(x_te))  # 计算混淆矩阵
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')  # 绘制混淆矩阵
np.set_printoptions(precision=2)
print('Accuracy:', (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
                cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[0, 0] + cnf_matrix[1, 0]))
print('Recall:', cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0]))
print('Precision:', cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[0, 1]))
print('Specificity:', cnf_matrix[0, 0] / (cnf_matrix[0, 1] + cnf_matrix[0, 0]))
plt.show()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true=y_te, y_pred=y)
print('gcForest accuracy : {}'.format(accuracy))

def write_geotiff(input_layer, input_array, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = input_array.shape
    dataset = driver.Create(input_layer, cols, rows, 1, eType=gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(input_array)
    band.SetNoDataValue(-9999)
    band.ComputeStatistics(False)
    dataset.BuildOverviews('average', [2, 4, 8, 16, 32])

    del dataset

output_layer= r'C:\Users\Dong\Desktop\DU\GC\gc0.tif'
y_predict_final = np.zeros((dimension[0] * dimension[1], 1))
y_predict_final[ma_t_f, 0] = y_predict
y_predict_final[~ma_t_f, 0] = np.nan
y_predict_final = y_predict_final.reshape(dimension[0], dimension[1])
write_geotiff(output_layer, y_predict_final, ds.GetGeoTransform(), ds.GetProjectionRef())
#
from sklearn.metrics import roc_curve, auc
path= r'C:\Users\Dong\Desktop\DU\GC\roc0.png'
score_svc = model.predict_proba(x_te)[:, 1]
fpr_svc, tpr_svc, threshold_svc = roc_curve(y_te, score_svc)
print("The AUC value of gcforest classifier is ", auc(fpr_svc, tpr_svc))

def plot_roc(fpr_svc, tpr_svc, output_roc_png, m):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr_svc, tpr_svc, linewidth=2,
            label='{} (AUC={})'.format(m, str(round(auc(fpr_svc, tpr_svc), 3))))
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.legend(fontsize=12)
    plt.savefig(output_roc_png)

plot_roc(fpr_svc, tpr_svc, path, m="gc")


