import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier


input_lyr_label = r"label.tif"
input_lyr_mask = r"mask.tif"
input_lyr_1 = r'input1.tif'
input_lyr_2 = r'input2.tif'
input_lyr_3 = r'input3.tif'
output_layer= r'out_layer.tif'

p_list = (input_lyr_mask,
          input_lyr_label,
          input_lyr_1,
          input_lyr_2,
          input_lyr_3)


def data_tf(params_list):
    input_lyr_list = [i for i in params_list]
    input_lyr = [os.path.split(i)[-1] for i in input_lyr_list]
    input_lyr_name = [os.path.splitext(i)[0].title() for i in input_lyr]
    input_layers_array = list(gdal.Open(i) for i in input_lyr)
    data_source = input_layers_array[-1]

    input_layers_array = np.array(list(map(lambda i: i.GetRasterBand(1).ReadAsArray(), input_layers_array)))

    input_layers_array = np.swapaxes(input_layers_array, 0, 2)
    input_layers_array = np.swapaxes(input_layers_array, 0, 1)
    dimension = input_layers_array.shape
    input_layers_array = input_layers_array.reshape((dimension[0] * dimension[1], dimension[2]))

    input_layers_pd = pd.DataFrame(input_layers_array, columns=input_lyr_name)

    mask_t_f = np.where(input_layers_pd['Mask'] == 1, True, False)
    x_label_mask_pd = input_layers_pd[input_layers_pd['Mask'] == 1].copy()
    label_t_f = np.where(x_label_mask_pd['Label'] == 1, True, False)
    x_pd = x_label_mask_pd.drop(labels=['Mask', "Label"], axis=1)
    x_standar_np = preprocessing.StandardScaler().fit_transform(x_pd)
    x_true_np = x_standar_np[label_t_f, :]
    x_false_np = x_standar_np[~label_t_f, :]

    return x_standar_np, x_true_np, x_false_np, mask_t_f, dimension, data_source


def data_split(x_true_np, x_false_np, ratio_value):
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x_t_np, np.ones((x_true_np.shape[0], 1)),
                                                                test_size=0.2, random_state=45)
    x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(x_f_np, np.zeros((x_false_np.shape[0], 1)),
                                                                test_size=0.2, random_state=45)

    np.random.shuffle(x_train_f)
    np.random.shuffle(x_test_f)

    x_train_f = x_train_f[:x_train_t.shape[0] * ratio_value, :]
    y_train_f = y_train_f[:y_train_t.shape[0] * ratio_value, :]

    x_test_f = x_test_f[:x_test_t.shape[0] * ratio_value, :]
    y_test_f = y_test_f[:y_test_t.shape[0] * ratio_value, :]

    x_train = np.vstack((x_train_t, x_train_f))
    y_train = np.vstack((y_train_t, y_train_f))

    x_test = np.vstack((x_test_t, x_test_f))
    y_test = np.vstack((y_test_t, y_test_f))

    print("Size of training set:{}, size of testing set:{}".format(x_train.shape[0], x_test.shape[0]))

    return x_train, y_train, x_test, y_test


def DF_model(x_train, y_train, x_test, x_standar_np):
    model = CascadeForestClassifier(n_estimators=4, n_trees=100, max_depth=8, min_samples_split=20,
                                     min_samples_leaf=10, max_layers=3, backend='sklearn', verbose=1, random_state=90)
    model.fit(x_train, y_train)   # Fitting estimator
    y_value = model.predict(x_test)  # Evaluating estimator
    y_predict_value = model.predict_proba(x_standar_np)[:, 1]
    lay_importance = model.get_layer_feature_importances(2)

    return y_value, y_predict_value, lay_importance


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


x_sc_np, x_t_np, x_f_np, ma_t_f, dim, ds = data_tf(p_list)
ratio = 2

x_tr, y_tr, x_te, y_te = data_split(x_t_np, x_f_np, ratio)

x_tr = x_tr.reshape(x_tr.shape[0],-1)
y_tr = y_tr.ravel()

y, y_predict, importance = DF_model(x_tr, y_tr,x_te,x_sc_np)

acc = accuracy_score(y_te, y) * 100
print("\nTesting Accuracy: {:.3f} %".format(acc))

y_predict_final = np.zeros((dim[0] * dim[1], 1))
y_predict_final[ma_t_f, 0] = y_predict
y_predict_final[~ma_t_f, 0] = np.nan
y_predict_final = y_predict_final.reshape(dim[0], dim[1])
write_geotiff(output_layer, y_predict_final, ds.GetGeoTransform(), ds.GetProjectionRef())
