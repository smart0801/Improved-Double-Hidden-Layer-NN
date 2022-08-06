import itertools
import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
#from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr

# 数据集及数据预处理函数
class DataSets:
    def __init__(self,
                path="./data/data.xlsx", 
                cols=["Year", "Cl", "SO2", "TOW", "Temperature", "Corrosion"], 
                new_cols=["Temperature", "TOW", "SO2", "Cl", "Year", "Corrosion"],
                feature_names=["Temperature", "TOW", "SO2", "Cl", "Year"],
                label_name=["Corrosion"],
                show_detail=False,
                seed=1234):
        self.path = path
        self.cols = cols
        self.new_cols = new_cols
        self.feature_names = feature_names
        self.label_name = label_name
        self.show_detail = show_detail
        self.seed = seed
        self.ori_data = pd.read_excel(path, names=cols)[new_cols]
        # 原论文引用的第12篇论文的数据的单位不正确，需要转换
        self.ori_data.loc[909:, "Corrosion"] = self.ori_data.loc[909:, "Corrosion"] / 7.85
        self.data = self.ori_data.copy()
        self.cleaned = False
        self.is_log1p = False
        self.is_scale = False


    def base_data_clean(self):
        if self.cleaned:
            print("Error, Data has been cleaned.")
            return
        data = self.ori_data.copy()
        # data = data[data["Temperature"] >= -3.1][data["Temperature"] <= 29.3]
        # # data = data[data["TOW"] >= 0.003][data["TOW"] <= 1]   # TOW不需要限定范围
        # data = data[data["SO2"] >= 0][data["SO2"] <= 175]
        # data = data[data["Cl"] >= 0][data["Cl"] <= 260]
        # data = data[data["Year"] >= 0.5][data["Year"] <= 12]
        # data = data[data["Corrosion"] >= 1.7][data["Corrosion"] <= 1040]
        self.data = data
        self.cleaned = True
        if self.show_detail:
            # Baseline论文通过限定取值范围过滤了异常值，具体如下：
            print("原数据取值范围：                 过滤后的数据取值范围：")
            print("Temperature range: -3.1-29.3  current:", min(data["Temperature"]), max(data["Temperature"]))
            print("TOW         range: 0.003-1    current:", min(data["TOW"]), max(data["TOW"]))
            print("SO2         range: 0-175      current:", min(data["SO2"]), max(data["SO2"]))
            print("Cl          range: 0-260      current:", min(data["Cl"]), max(data["Cl"]))
            print("Year        range: 0.5-12     current:", min(data["Year"]), max(data["Year"]))
            print("Corrosion   range: 1.7-1040   current:", min(data["Corrosion"]), max(data["Corrosion"]))
            print("")
            print("Baseline使用了943条数据，当前剩余{}条数据.".format(len(data)))

    def ours_data_clean(self):
        if self.cleaned:
            print("Error, Data has been cleaned.")
            return
        data = self.ori_data.copy()
        data["Corrosion"] = data["Corrosion"] / data["Year"]
        # data = data[data["Temperature"] >= -3.1][data["Temperature"] <= 29.3]
        # # data = data[data["TOW"] >= 0.003][data["TOW"] <= 1]   # TOW不需要限定范围
        # data = data[data["SO2"] >= 0][data["SO2"] <= 175]
        # data = data[data["Cl"] >= 0][data["Cl"] <= 260]
        # data = data[data["Year"] >= 0.5][data["Year"] <= 12]
        # data = data[data["Corrosion"] >= 1.7]
        # data = data[data["Corrosion"] <= 400]
        self.cleaned = True
        self.data = data
        if self.show_detail:
            print("原数据取值范围：                 过滤后的数据取值范围：")
            print("Temperature range: -3.1-29.3  current:", min(data["Temperature"]), max(data["Temperature"]))
            print("TOW         range: 0.003-1    current:", min(data["TOW"]), max(data["TOW"]))
            print("SO2         range: 0-175      current:", min(data["SO2"]), max(data["SO2"]))
            print("Cl          range: 0-260      current:", min(data["Cl"]), max(data["Cl"]))
            print("Year        range: 0.5-12     current:", min(data["Year"]), max(data["Year"]))
            print("Corrosion   range: 1.7-1040   current:", min(data["Corrosion"]), max(data["Corrosion"]))
            print("")
            print("Baseline使用了943条数据，当前剩余{}条数据.".format(len(data)))

    def data_split(self, test_ratio=0.1):
        X = self.data[self.feature_names]
        y = self.data[self.label_name]
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_ratio, random_state=self.seed)
        self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test = \
            X, y, X_train, X_test, y_train, y_test
        
        self.y_test_real = self.y_test.copy()
        train = np.hstack((X_train, y_train))
        test = np.hstack((X_test, y_test))
        self.raw_data = pd.DataFrame(np.vstack((test, train)), columns=self.new_cols)

    def data_log1p(self,
                log1p_cols=["SO2", "Cl"],
                #log1p_cols=["SO2", "Cl", "Year"],
                log1p_label=["Corrosion"]):
        def show_skewness_fig(title='Highest Skewness'):
            sns.set_style('whitegrid')
            fig, axes = plt.subplots(2, 3, figsize=(18, 8))
            plt.subplots_adjust(hspace=0.7, wspace=0.2)
            fig.suptitle(title, fontsize=20)

            for i, col in zip(range(len(self.cols)), self.cols):
                sns.kdeplot(self.data[col], ax=axes[i//3][i % 3], fill=True)
                axes[i//3][i % 3].set_title(col+' Distribution')

            plt.show()
        def show_skewness_data(): 
            skewness = pd.DataFrame()
            skewness[['Positive Columns','Skewness(+v)']] = \
                self.data[self.cols].skew().sort_values(ascending=False)[:10].reset_index()
            skewness[['Negative Columns','Skewness(-v)']] = \
                self.data[self.cols].skew().sort_values(ascending=True)[:10].reset_index()
            skewness.columns = pd.MultiIndex.from_tuples(\
                [('Positive Skewness', 'Columns'), ('Positive Skewness', 'Skewness'),\
                ('Negative Skewness', 'Columns'), ('Negative Skewness', 'Skewness')])
            print(skewness)
        if self.show_detail:
            show_skewness_fig(title='Highest Skewness Before Log1p')
            show_skewness_data()

        self.data[log1p_cols] = np.log1p(self.data[log1p_cols])
        self.data[log1p_label] = np.log1p(self.data[log1p_label])
        self.X_train[log1p_cols] = np.log1p(self.X_train[log1p_cols])
        self.X_test[log1p_cols] = np.log1p(self.X_test[log1p_cols])
        self.y_train[log1p_label] = np.log1p(self.y_train[log1p_label])
        self.y_test[log1p_label] = np.log1p(self.y_test[log1p_label])

        self.is_log1p = True
        if self.show_detail:
            show_skewness_fig(title='Highest Skewness After Log1p')
            show_skewness_data()

    def data_scale(self):
        self.Xscaler = MinMaxScaler(feature_range=(0, 1))
        self.Xscaler.fit(self.X_train)
        self.X_train = self.Xscaler.transform(self.X_train)
        self.X_test = self.Xscaler.transform(self.X_test)
        
        self.yscaler = MinMaxScaler(feature_range=(0, 1))
        self.yscaler.fit(self.y_train)
        self.y_train = self.yscaler.transform(self.y_train)
        self.y_test = self.yscaler.transform(self.y_test)
        self.is_scale = True

    def label_inverse_scale(self, y_pred):
        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=self.label_name)
        if self.is_scale:
            y_pred = self.yscaler.inverse_transform(y_pred)
        if self.is_log1p:
            y_pred = np.expm1(y_pred)
        return pd.DataFrame(y_pred, columns=self.label_name)

    def pipeline(self, name="base"):
        if name == "none":
            self.data_split()
        elif name == 'scale':
            self.data_split()
            self.data_scale()
        elif name == 'base':
            self.base_data_clean()
            self.data_split()
            self.data_scale()
        elif name == 'base_log':
            self.base_data_clean()
            self.data_split()
            self.data_log1p()
            self.data_scale()
        elif name == 'ours':
            self.ours_data_clean()
            self.data_split()
            self.data_scale()
        elif name == 'ours_log':
            self.ours_data_clean()
            self.data_split()
            self.data_log1p()
            self.data_scale()
        else:
            raise ValueError("Wrong pipeline name!")

        return self.X_train, self.y_train, self.X_test, self.y_test, self.raw_data, self.y_test_real

    def inspect_data(self):
        print("\n" + "="*60 + "\nData Detail:")
        print("X      :", self.X.shape, end="")
        print("  \ty      :", self.y.shape)
        print("X_train:", self.X_train.shape, end="")
        print("  \ty_train:", self.y_train.shape)
        print("X_test :", self.X_test.shape, end="")
        print("  \ty_test :", self.y_test.shape)

        print(f"\n             Temp     TOW     SO2      Cl    Year    Corr")
        print(f"Min X    : %6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f" % \
            (*list(np.min(self.X, axis=0)), *list(np.min(self.y, axis=0))))
        print(f"Max X    : %6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f" % \
            (*list(np.max(self.X, axis=0)), *list(np.max(self.y, axis=0))))
        print(f"Min train: %6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f" % \
            (*list(np.min(self.X_train, axis=0)), *list(np.min(self.y_train, axis=0))))
        print(f"Max train: %6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f" % \
            (*list(np.max(self.X_train, axis=0)), *list(np.max(self.y_train, axis=0))))
        print(f"Min test : %6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f" % \
            (*list(np.min(self.X_test, axis=0)), *list(np.min(self.y_test, axis=0))))
        print(f"Max test : %6.2f, %6.2f, %6.2f, %6.2f, %6.2f, %6.2f" % \
            (*list(np.max(self.X_test, axis=0)), *list(np.max(self.y_test, axis=0))))
# Klinesmith算法
class Klinesmith():
    def __init__(self):
        pass
    
    def fit(sef, X, y):
        pass

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            #print("Error: X must be a pandas DataFrame.")
            X = pd.DataFrame(X, columns=["Temperature", "TOW", "SO2", "Cl", "Year"])
        A, B, C, D, E, F, G, H, J, T0 = \
            14.5018, 0.7528, 0.4474, 0.3641, 22.0902, 0.4210, 23.9742, 0.6684, 0.0068, 20
        res = A * pow(X["Year"], B)
        res = res * pow((X["TOW"]/C), D)
        res = res * pow((X["SO2"]/E + 1), F)    
        res = res * pow((X["Cl"]/G + 1), H)
        res = res * np.exp((J * (X["Temperature"] + T0)))
        return res
# 所有用到的模型
class Models:
    def __init__(self, model_name, pipeline="Unknown",save_path="models", seed=1234, force_train=False, show_detail=False):
        self.model_name = model_name
        self.show_detail = show_detail
        self.seed = seed
        self.model = None
        self.model_path = os.path.join(save_path, pipeline + "_" + model_name + ".pkl")
        self.force_train = force_train
        self.pipeline = pipeline

    def model_fit(self, X_train, y_train):
        if self.model_name.startswith('linear'):
            self.model = LinearRegression()
            X_train, X_test, y_train, y_test = \
                train_test_split(X_train, y_train, test_size=0.11, random_state=self.seed)
        elif self.model_name.startswith('ridge'):
            self.model = Ridge(alpha=0.5, max_iter=10000)
            X_train, X_test, y_train, y_test = \
                train_test_split(X_train, y_train, test_size=0.11, random_state=self.seed)
        elif self.model_name.startswith('lasso'):
            self.model = Lasso(alpha=0.5, max_iter=10000)
            X_train, X_test, y_train, y_test = \
                train_test_split(X_train, y_train, test_size=0.11, random_state=self.seed)
        elif self.model_name.startswith('klinesmith'):
            self.model = Klinesmith()
            X_train, X_test, y_train, y_test = \
                train_test_split(X_train, y_train, test_size=0.11, random_state=self.seed)
        elif self.model_name.startswith('elasticnet'):
            self.model = ElasticNet()
        elif self.model_name.startswith('decisiontree'):
            self.model = DecisionTreeRegressor()
        elif self.model_name.startswith('randomforest'):
            self.model = RandomForestRegressor()
        elif self.model_name.startswith('xgboost'):
            self.model = XGBRegressor()
        elif self.model_name.startswith('lightgbm'):
            self.model = LGBMRegressor()
        elif self.model_name.startswith('catboost'):
            self.model = CatBoostRegressor()
        elif self.model_name.startswith('svm'):
            self.model = SVR()
        elif self.model_name.startswith('gaussian'):
            kernel = RationalQuadratic(alpha=0.1, length_scale=1.0)
            self.model = GaussianProcessRegressor(kernel=kernel,
                                                n_restarts_optimizer=9)
        elif self.model_name.startswith('base_mlp'):
            self.model = MLPRegressor(
                hidden_layer_sizes=(32,), max_iter=100000,
                solver='sgd', 
                random_state=self.seed, 
                learning_rate_init=0.001, 
                momentum=0.9,
                learning_rate='adaptive',
                validation_fraction=0.11,
                activation='logistic', # ['identity', 'logistic', 'relu', 'softmax', 'tanh']
                early_stopping=True, 
                batch_size=1)
        elif self.model_name.startswith('basic_double'):
            self.model = MLPRegressor(
                hidden_layer_sizes=(50, 30), max_iter=100000,
                solver='sgd',
                random_state=self.seed,
                learning_rate_init=0.001,
                momentum=0.9,
                learning_rate='adaptive',
                validation_fraction=0.11,
                activation='logistic', # ['identity', 'logistic', 'relu', 'softmax', 'tanh']
                early_stopping=True,
                batch_size=1)
        elif self.model_name.startswith('ours_mlp'):
            self.model = MLPRegressor(
                hidden_layer_sizes=(45, 36),
                max_iter=100000,
                solver='lbfgs', 
                random_state=self.seed, 
                learning_rate_init=0.001, 
                momentum=0.9,
                learning_rate='adaptive',
                validation_fraction=0.11,
                activation='relu', # ['identity', 'logistic', 'relu', 'softmax', 'tanh']
                early_stopping=True, 
                batch_size=1)
        elif self.model_name.startswith('grid_mlp'):
            parametersGrid = {
                'hidden_layer_sizes': \
                    list(itertools.product(range(10, 100, 1), range(5, 50, 1))),
            }
            kfold = KFold(n_splits=9, shuffle=True, random_state=self.seed)
            mlpr = MLPRegressor(
                solver='lbfgs',
                random_state=self.seed,
                learning_rate_init=0.001, 
                momentum=0.9,
                learning_rate='adaptive',
                validation_fraction=0.11,
                activation='relu', # ['identity', 'logistic', 'relu', 'softmax', 'tanh']
                early_stopping=True, 
                batch_size=1,
                max_iter=10000
            )
            self.model = GridSearchCV(mlpr, parametersGrid, scoring='neg_mean_squared_error', cv=kfold, n_jobs=-1)
        else:
            raise ValueError("Unsupported model type")
        if os.path.exists(self.model_path) and self.force_train == False:
            self.model = joblib.load(self.model_path)
            print("loaded====================================")
        else:
            print("traing====================================")

            self.model.fit(X_train, y_train)
            joblib.dump(self.model, self.model_path)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
# 评估标准
class Metrics:
    def __init__(self, metric_name, show_detail=False):
        self.metric_name = metric_name
        self.show_detail = show_detail

    def predict(self, y_true, y_pred):
        ret = None
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        if self.metric_name == 'mean_relative_error' or self.metric_name == 'MRE':
            ret = np.mean(np.abs(y_true - y_pred) / y_true)
        elif self.metric_name == 'root_mean_squared_error' or self.metric_name == 'RMSE':
            ret = mean_squared_error(y_true, y_pred, squared=False)
        elif self.metric_name == 'correlation' or self.metric_name == 'R':
            ret, _ = pearsonr(y_true.squeeze(), y_pred.squeeze())
        else:
            raise ValueError("Unsupported metric type")
        ret = np.array(ret)
        return float(ret)

# 随机种子
seed = 1234
# 数据处理方案
data_pipeline_names = [
    'none',     # 未作处理
    'scale',    # 只做缩放
    'base',     # 基线数据处理方式1  :              按阈值过滤， 缩放
    'base_log', # 基线数据处理方式2  :              按阈值过滤， 缩放，对数变换
    'ours',     # 我们的数据处理方式1: 转换为腐蚀速率，按新阈值过滤，缩放
    'ours_log'  # 我们的数据处理方式2: 转换为腐蚀速率，按新阈值过滤，缩放，对数变换
]
# 使用到的模型
model_names = [
    'linear',
    'ridge',
    'lasso',
    #'klinesmith',
    # 'elasticnet',
    # 'decisiontree',
    # 'randomforest',
    # 'xgboost',
    # 'lightgbm',
    # 'catboost',
    # 'svm',
    #'gaussian',
    'base_mlp',
    'basic_double',
    'ours_mlp',
    'grid_mlp'
]
# 评估标准
metrics = [
    Metrics("RMSE", show_detail=False),
    Metrics("R", show_detail=False),
    Metrics("MRE", show_detail=False)
]

# 写入excel文件
writer = pd.ExcelWriter("data/results.xlsx")

# 如果设置force_train为False
#   如果models文件夹下有训练好的模型，则加载训练好的模型
#   如果models文件夹相应的模型不存在，则训练并保存
#   即：第一次运行会训练并保存所有模型，之后运行则直接加载训练好的模型
# 如果设置force_train为True
#   重新训练所有模型，并保存到models文件夹下，如果已有模型，则覆盖
force_train = False

if not os.path.exists("models"):
    os.mkdir("models")

for pipeline_name in data_pipeline_names: # 外层循环，每次循环都是一个数据处理方案
    # 读取数据，创新数据集
    dataset = DataSets(seed=seed, show_detail=False)
    # 按照pipeline_name处理数据，并获取训练集和测试集。 raw_data, y_test_real主要用于将结果写入excel
    X_train, y_train, X_test, y_test, raw_data, y_test_real = dataset.pipeline(pipeline_name)
    
    # 以下若干行代码适用于将结果写入excel，不用深入了解
    startrow, startcol = len(metrics), 0
    feature_names = ["Temperature", "TOW", "SO2", "Cl", "Year", "Corrosion"]
    raw_data.to_excel(writer, sheet_name=pipeline_name, columns=feature_names,
        startrow=startrow, startcol=startcol, index=False)
    startcol += len(feature_names)
    label_cols = ["Scaled Corrosion"]
    label = pd.DataFrame(y_test, columns=label_cols)
    label.to_excel(writer, sheet_name=pipeline_name, columns=label_cols,
        startrow=startrow, startcol=startcol, index=False)
    startcol += len(label_cols)
    for i, metric in enumerate(metrics):
        pd.DataFrame(columns=[metric.metric_name]).to_excel(writer, sheet_name=pipeline_name,
            startrow=i, startcol=0, index=False)

    for name in model_names: # 内层循环，每次循环都是一个模型。
        for time in range(1, 101):
            split_seed = seed + time*11
            # 创建模型并训练
            model_name = name + "_" + str(time)
            model = Models(model_name, pipeline=pipeline_name, save_path="models", seed=split_seed, force_train=force_train, show_detail=False)
            
            model.model_fit(X_train, y_train)
            # 预测，结果保存在y_pred中
            y_pred = np.array(model.predict(X_test)).reshape(-1, 1)
            # 将预测结果反缩放
            y_pred_inversed = dataset.label_inverse_scale(y_pred)

            # 将反缩放的预测结果和模型输出的预测结果都写入excel
            label_cols_pred = [model_name + " Predict", model_name + " Scaled Predict"]
            label = pd.DataFrame(np.hstack((y_pred_inversed, y_pred)), columns=label_cols_pred)
            label.to_excel(writer, sheet_name=pipeline_name, columns=label_cols_pred,
                startrow=startrow, startcol=startcol, index=False)
            
            # 使用评估标准进行评估
            for i, metric in enumerate(metrics):
                # 计算反缩放过的评估值，以后咱们用该值进行评估
                metric_value = metric.predict(y_test_real, y_pred_inversed)
                pd.DataFrame(columns=[metric_value]).to_excel(writer, sheet_name=pipeline_name,
                    startrow=i, startcol=startcol, index=False)

                    
                print("Inverse Scale:{:8} {:12} {:5} {:.4}".format(pipeline_name, model_name, metric.metric_name, metric_value))
                # 计算缩放过的评估值，学姐用的这个值进行评估，这个评估值只是用来和学姐的结果做对比
                metric_value = metric.predict(y_test, y_pred)
                if pipeline_name == "ours_log" and model_name == "linear" and metric.metric_name=="MRE":
                    print(y_test)
                    print(y_pred)
                    print(metric_value)


                pd.DataFrame(columns=[metric_value]).to_excel(writer, sheet_name=pipeline_name,
                    startrow=i, startcol=startcol + 1, index=False)
                print("Scaled       :{:8} {:12} {:5} {:.4}".format(pipeline_name, model_name, metric.metric_name, metric_value))
                
            startcol += len(label_cols_pred)

# 结果写入excel文件
writer.save()
writer.close()

