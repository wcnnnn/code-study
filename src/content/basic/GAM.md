
# 广义加性模型(GAM)：灵活的非线性建模框架

## 一、引言

### 1.1 问题背景
- 现实世界中的非线性关系普遍存在
- 传统线性模型和GLM的局限性
- 需要在可解释性和灵活性之间取得平衡
- 多维特征的非线性效应难以建模

### 1.2 算法概述
- GAM是GLM的自然扩展
- 将线性项替换为平滑函数
- 保持可加性结构
- 基本形式：$g(E(Y)) = \beta_0 + \sum_{j=1}^p f_j(X_j)$

### 1.3 优势与应用
- 优势：
  - 保持了GLM的可解释性
  - 具有非线性建模能力
  - 可以处理不同类型的响应变量
  - 支持部分特征的线性约束

- 应用领域：
  - 医疗健康：疾病风险预测
  - 环境科学：污染物浓度建模
  - 金融：信用评分模型
  - 生态学：物种分布预测

## 二、理论基础

### 2.1 数学表达

#### 2.1.1 基本形式
对于响应变量Y和预测变量$X_1, ..., X_p$：

$g(E(Y|X)) = \beta_0 + \sum_{j=1}^p f_j(X_j)$

其中：
- $g(\cdot)$ 是链接函数
- $f_j(\cdot)$ 是平滑函数
- $\beta_0$ 是截距项

#### 2.1.2 平滑函数表示
每个平滑函数可以用基函数展开：

$f_j(X_j) = \sum_{k=1}^{K_j} \beta_{jk}b_{jk}(X_j)$

其中：
- $b_{jk}(\cdot)$ 是基函数
- $\beta_{jk}$ 是系数
- $K_j$ 是基函数数量

### 2.2 理论性质

#### 2.2.1 估计方法
目标函数：

$\min_{\beta_0, f_1,...,f_p} \sum_{i=1}^n L(y_i, \beta_0 + \sum_{j=1}^p f_j(x_{ij})) + \sum_{j=1}^p \lambda_j \int [f_j''(t)]^2 dt$

其中：
- $L(\cdot)$ 是损失函数
- $\lambda_j$ 是平滑参数
- 积分项是惩罚项

#### 2.2.2 统计性质
- 一致性：在适当条件下估计量是一致的
- 渐近正态性：估计量渐近服从正态分布
- 收敛速率：依赖于真实函数的光滑程度

### 2.3 算法改进
- 变量选择：
  - LASSO型惩罚
  - 组惩罚方法
  - 自适应惩罚

- 计算优化：
  - 后向拟合算法
  - 块坐标下降
  - 并行计算策略

- 模型扩展：
  - 交互项处理
  - 时变系数
  - 空间效应

## 三、代码实现

### 3.1 基础实现

```python
import numpy as np
from pygam import GAM, s, f, l
from pygam.datasets import wage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

class CustomGAM:
    """
    自定义GAM实现类
    """
    def __init__(self, n_splines=10, lam=0.6):
        """
        初始化GAM模型
        
        参数:
        n_splines: 每个特征的样条基函数数量
        lam: 平滑参数
        """
        self.n_splines = n_splines
        self.lam = lam
        self.gam = GAM(s(0, n_splines=n_splines, basis='cc') + 
                       s(1, n_splines=n_splines, basis='cc') +
                       s(2, n_splines=n_splines, basis='cc'),
                       lam=lam)
    
    def fit(self, X, y):
        """
        训练GAM模型
        """
        self.gam.fit(X, y)
        return self
    
    def predict(self, X):
        """
        预测新数据
        """
        return self.gam.predict(X)
    
    def plot_partial_dependence(self, feature_idx, X, feature_name=None):
        """
        绘制部分依赖图
        """
        XX = self.gam.generate_X_grid(term=feature_idx)
        pdep = self.gam.partial_dependence(term=feature_idx, X=XX)
        
        plt.figure(figsize=(8, 6))
        plt.plot(XX[:, feature_idx], pdep)
        plt.plot(XX[:, feature_idx], pdep + 2*self.gam.standard_errors(term=feature_idx, X=XX), 'r--')
        plt.plot(XX[:, feature_idx], pdep - 2*self.gam.standard_errors(term=feature_idx, X=XX), 'r--')
        plt.scatter(X[:, feature_idx], self.gam.deviance_residuals, c='g', alpha=0.5)
        
        if feature_name:
            plt.xlabel(feature_name)
        plt.ylabel('Partial dependence')
        plt.title(f'Feature {feature_idx} partial dependence')
        plt.grid(True)
        plt.show()
```

### 3.2 进阶功能

```python
class AdvancedGAM:
    """
    高级GAM实现，包含自动化特征选择和交互项
    """
    def __init__(self, max_splines=10, max_interactions=2):
        self.max_splines = max_splines
        self.max_interactions = max_interactions
        self.best_model = None
        self.feature_importance = None
    
    def _generate_interaction_terms(self, n_features):
        """生成可能的交互项组合"""
        from itertools import combinations
        interactions = []
        for i in range(2, self.max_interactions + 1):
            interactions.extend(combinations(range(n_features), i))
        return interactions
    
    def fit(self, X, y, cv=5):
        """
        使用交叉验证选择最佳模型
        """
        n_features = X.shape[1]
        best_score = float('-inf')
        
        # 尝试不同的样条数量和交互项组合
        for n_splines in range(3, self.max_splines + 1):
            # 基本项
            terms = sum(s(i, n_splines=n_splines) for i in range(n_features))
            
            # 添加交互项
            for interaction in self._generate_interaction_terms(n_features):
                current_model = GAM(terms + te(*interaction, n_splines=n_splines))
                scores = []
                
                # 交叉验证
                for train_idx, test_idx in KFold(cv).split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    current_model.fit(X_train, y_train)
                    score = current_model.score(X_test, y_test)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    self.best_model = current_model
        
        # 最终拟合
        self.best_model.fit(X, y)
        return self
```

### 3.3 模型诊断

```python
def model_diagnostics(gam_model, X, y):
    """
    GAM模型诊断函数
    """
    diagnostics = {
        'summary': gam_model.summary(),
        'metrics': {
            'R2': gam_model.score(X, y),
            'GCV': gam_model.gcv_score_,
            'AIC': gam_model.statistics_['AIC'],
            'BIC': gam_model.statistics_['BIC']
        },
        'residuals': {
            'deviance': gam_model.deviance_residuals,
            'pearson': gam_model.pearson_residuals
        }
    }
    
    # 绘制诊断图
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # QQ图
    from scipy.stats import probplot
    probplot(diagnostics['residuals']['deviance'], plot=axes[0,0])
    axes[0,0].set_title('Q-Q Plot')
    
    # 残差vs预测值
    axes[0,1].scatter(gam_model.predict(X), 
                     diagnostics['residuals']['deviance'])
    axes[0,1].set_title('Residuals vs Predicted')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('Residuals')
    
    # 残差直方图
    axes[1,0].hist(diagnostics['residuals']['deviance'], bins=30)
    axes[1,0].set_title('Residuals Distribution')
    
    # 尺度-位置图
    sqrt_abs_res = np.sqrt(np.abs(diagnostics['residuals']['deviance']))
    axes[1,1].scatter(gam_model.predict(X), sqrt_abs_res)
    axes[1,1].set_title('Scale-Location Plot')
    
    plt.tight_layout()
    plt.show()
    
    return diagnostics
   ```
## 四、实验分析

### 4.1 实验设计

#### 4.1.1 数据生成
```python
def generate_synthetic_data(n_samples=1000, noise=0.1):
    """生成合成数据用于实验"""
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (n_samples, 3))
    
    # 生成非线性关系
    y = (np.sin(X[:, 0]) + 
         0.5 * X[:, 1]**2 + 
         2 * np.cos(X[:, 2]) + 
         noise * np.random.normal(0, 1, n_samples))
    
    return X, y

# 生成数据
X, y = generate_synthetic_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### 4.1.2 实验设置
```python
experiment_settings = {
    "模型配置": [
        {
            "name": "GAM_basic",
            "params": {"n_splines": 5, "lam": 0.6}
        },
        {
            "name": "GAM_medium",
            "params": {"n_splines": 10, "lam": 0.3}
        },
        {
            "name": "GAM_complex",
            "params": {"n_splines": 15, "lam": 0.1}
        }
    ],
    "评估指标": ["R²", "MSE", "AIC", "BIC"],
    "交叉验证": {"k_folds": 5}
}
```

### 4.2 实验代码

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

class CustomGAM:
    """
    使用scikit-learn实现的GAM
    """
    def __init__(self, n_splines=10, lam=0.6):
        """
        初始化GAM模型
        
        参数:
        n_splines: 每个特征的样条基函数数量
        lam: 正则化参数（Ridge regression的alpha参数）
        """
        self.n_splines = n_splines
        self.lam = lam
        
        # 为每个特征创建一个SplineTransformer
        self.spline_transformers = [
            SplineTransformer(n_knots=n_splines, degree=3) 
            for _ in range(3)  # 假设有3个特征
        ]
        
        # 创建Pipeline
        self.model = Pipeline([
            ('splines', SplineTransformer(n_knots=n_splines, degree=3)),
            ('ridge', Ridge(alpha=lam))
        ])
    
    def fit(self, X, y):
        """训练模型"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    

def generate_synthetic_data(n_samples=1000, noise=0.1):
    """生成合成数据"""
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (n_samples, 3))
    
    y = (np.sin(X[:, 0]) + 
         0.5 * X[:, 1]**2 + 
         2 * np.cos(X[:, 2]) + 
         noise * np.random.normal(0, 1, n_samples))
    
    return X, y

# 实验设置
experiment_settings = {
    "模型配置": [
        {
            "name": "GAM_basic",
            "params": {"n_splines": 5, "lam": 0.6}
        },
        {
            "name": "GAM_medium",
            "params": {"n_splines": 10, "lam": 0.3}
        },
        {
            "name": "GAM_complex",
            "params": {"n_splines": 15, "lam": 0.1}
        }
    ]
}

def run_gam_experiment():
    """运行GAM实验"""
    # 生成数据
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = {}
    models = {}
    
    for config in experiment_settings["模型配置"]:
        # 训练模型
        model = CustomGAM(**config["params"])
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算性能指标
        results[config["name"]] = {
            "R²": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred)
        }
        
        models[config["name"]] = model
    
    return results, models, X_train, X_test, y_train, y_test
def plot_all_partial_dependence(models, X_train, feature_names=None):
    """
    在一个图中绘制所有特征的部分依赖图
    
    参数:
    models: dict, 模型字典
    X_train: array, 训练数据
    feature_names: list, 特征名称列表
    """
    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]
    
    # 设置颜色方案
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # 创建子图
    fig, axes = plt.subplots(1, n_features, figsize=(15, 5))
    fig.suptitle('Partial Dependence Plots for All Features', fontsize=14)
    
    # 为每个特征绘制部分依赖图
    for feature_idx in range(n_features):
        ax = axes[feature_idx]
        
        # 计算特征的范围
        x_feature = np.linspace(
            X_train[:, feature_idx].min(), 
            X_train[:, feature_idx].max(), 
            100
        )
        
        # 为每个模型绘制曲线
        for (model_name, model), color in zip(models.items(), colors):
            X_temp = X_train.copy()
            predictions = []
            
            # 计算部分依赖
            for x in x_feature:
                X_temp[:, feature_idx] = x
                pred = model.predict(X_temp)
                predictions.append(np.mean(pred))
            
            # 绘制曲线
            ax.plot(x_feature, predictions, color=color, label=model_name, linewidth=2)
            
            # 添加散点图（只为第一个模型添加，避免重复）
            if model_name == list(models.keys())[0]:
                ax.scatter(X_train[:, feature_idx], 
                          model.predict(X_train) - np.mean(model.predict(X_train)),
                          color='gray', alpha=0.1, s=20)
        
        # 设置标签和网格
        ax.set_xlabel(feature_names[feature_idx])
        if feature_idx == 0:
            ax.set_ylabel('Partial dependence')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 只在第一个子图显示图例
        if feature_idx == 0:
            ax.legend()
        
        # 设置标题
        true_functions = {
            0: 'sin(x)',
            1: '0.5x²',
            2: '2cos(x)'
        }
        ax.set_title(f'True function: {true_functions[feature_idx]}')
    
    plt.tight_layout()
    plt.show()

# 在运行实验后调用绘图函数
results, models, X_train, X_test, y_train, y_test = run_gam_experiment()

# 打印性能结果
print("\n模型性能对比：")
performance_df = pd.DataFrame(results).round(4)
print(performance_df)

# 绘制所有特征的部分依赖图
plot_all_partial_dependence(models, X_train)

# 运行实验
results = run_gam_experiment()
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f1261a389e23461988f1f70070f26346.png#pic_center)

### 4.3 结果分析

#### 4.3.1 定量分析

实验结果显示三种复杂度的GAM模型性能：
```python
experiment_results = {
    "GAM_basic": {    # 5个节点
        "R²": 0.9946,
        "MSE": 0.0224,
        "特点": "简单模型已达到很高精度"
    },
    "GAM_medium": {   # 10个节点
        "R²": 0.9965,
        "MSE": 0.0147,
        "特点": "进一步提升性能"
    },
    "GAM_complex": {  # 15个节点
        "R²": 0.9970,
        "MSE": 0.0125,
        "特点": "最佳性能但提升幅度减小"
    }
}
```

从结果可以观察到：
1. **整体性能**：
   - 所有模型都达到了极高的拟合精度（R² > 0.99）
   - MSE值都很小，表明预测误差较低
   - 随着复杂度增加，性能提升但边际效应递减

2. **性能提升趋势**：
   - R²提升：0.9946 → 0.9965 → 0.9970
   - MSE降低：0.0224 → 0.0147 → 0.0125
   - 从basic到medium的提升（ΔR² = 0.0019）大于medium到complex的提升（ΔR² = 0.0005）

#### 4.3.2 定性分析

1. **特征效应分析**：
   - Feature 0 (sin(x))：
     - 所有模型都准确捕捉到正弦波形
     - 复杂模型在波峰和波谷处拟合更精确
     - 散点分布显示较大的随机性

   - Feature 1 (0.5x²)：
     - 完美捕捉二次函数形态
     - 三个模型的表现几乎一致
     - 在数据边界处的拟合尤为准确

   - Feature 2 (2cos(x))：
     - 准确重现余弦函数形态
     - 振幅和周期都被精确建模
     - 在极值点附近的拟合特别准确

2. **模型复杂度权衡**：
   ```python
   complexity_analysis = {
       "GAM_basic": {
           "优势": ["计算效率高", "已达到优秀性能", "避免过拟合"],
           "劣势": ["局部细节可能略有损失"]
       },
       "GAM_medium": {
           "优势": ["较好的性能提升", "平衡复杂度和精度"],
           "劣势": ["计算成本增加", "收益递减开始显现"]
       },
       "GAM_complex": {
           "优势": ["最佳整体性能", "最好的局部拟合"],
           "劣势": ["计算成本最高", "边际效益较小"]
       }
   }
   ```

3. **实践建议**：
   - 对于类似的非线性关系，GAM_basic（5个节点）已经足够
   - 如果需要更精确的局部拟合，可以考虑GAM_medium
   - GAM_complex的边际收益较小，除非对精度有极高要求
   - 在实际应用中，建议从简单模型开始，根据需要逐步增加复杂度

## 五、实践指南

### 5.1 参数调优

#### 5.1.1 关键参数
```python
parameter_guide = {
    "样条数量": {
        "说明": "每个特征的基函数数量",
        "建议范围": "3-15",
        "调优方法": "交叉验证",
        "注意事项": "过多可能导致过拟合"
    },
    "平滑参数": {
        "说明": "控制函数光滑程度",
        "建议范围": "0.1-1.0",
        "调优方法": "GCV或REML",
        "注意事项": "需要权衡偏差和方差"
    },
    "基函数类型": {
        "说明": "样条基函数的选择",
        "选项": ["循环样条", "B样条", "自然样条"],
        "选择依据": "数据特征和边界行为"
    }
}
```

#### 5.1.2 调优流程
1. **数据预处理**：
   - 特征标准化：$X_{scaled} = \frac{X - \mu}{\sigma}$
   - 异常值处理
   - 缺失值处理

2. **模型选择**：
   ```python
   def select_best_model(X, y, cv=5):
       """模型选择函数"""
       best_params = {
           'n_splines': None,
           'lam': None,
           'score': float('-inf')
       }
       
       for n_splines in [5, 10, 15]:
           for lam in [0.1, 0.3, 0.6, 1.0]:
               model = CustomGAM(n_splines=n_splines, lam=lam)
               scores = cross_val_score(model, X, y, cv=cv)
               avg_score = np.mean(scores)
               
               if avg_score > best_params['score']:
                   best_params.update({
                       'n_splines': n_splines,
                       'lam': lam,
                       'score': avg_score
                   })
       
       return best_params
   ```

### 5.2 注意事项

#### 5.2.1 常见问题及解决方案
```python
common_issues = {
    "过拟合": {
        "症状": ["测试集性能差", "函数过于波动"],
        "解决方案": [
            "增加平滑参数",
            "减少样条数量",
            "使用交叉验证"
        ]
    },
    "欠拟合": {
        "症状": ["训练集和测试集性能都差", "函数过于平滑"],
        "解决方案": [
            "减小平滑参数",
            "增加样条数量",
            "考虑添加交互项"
        ]
    },
    "计算效率": {
        "症状": ["训练时间过长", "内存使用过大"],
        "解决方案": [
            "减少样条数量",
            "使用稀疏矩阵",
            "考虑分批训练"
        ]
    }
}
```

#### 5.2.2 最佳实践
1. **模型构建**：
   - 从简单模型开始
   - 逐步增加复杂度
   - 注意监控过拟合

2. **特征工程**：
   - 处理非线性关系
   - 考虑特征交互
   - 注意特征尺度

### 5.3 应用案例

#### 5.3.1 信用评分模型
```python
def credit_scoring_gam():
    """信用评分GAM示例"""
    # 模型定义
    gam = GAM(s(0, basis='cr', constraints='monotonic_inc') +  # 年龄
              s(1) +                                           # 收入
              f(2) +                                          # 教育程度
              s(3, constraints='monotonic_dec'))              # 负债比率
    return gam
```

#### 5.3.2 环境数据分析
```python
def environmental_gam():
    """环境数据分析GAM示例"""
    # 模型定义（包含周期性约束）
    gam = GAM(s(0, basis='cc') +     # 温度
              s(1, basis='cc') +     # 湿度
              s(2, basis='cc') +     # 风速
              te(0, 1))              # 温度-湿度交互
    return gam
```

## 六、进阶探讨

### 6.1 算法优化

#### 6.1.1 计算效率优化
```python
optimization_strategies = {
    "矩阵计算": [
        "使用稀疏矩阵表示",
        "块矩阵分解",
        "并行计算"
    ],
    "模型训练": [
        "分批训练策略",
        "增量更新方法",
        "早停策略"
    ],
    "内存管理": [
        "数据生成器",
        "特征矩阵压缩",
        "模型检查点"
    ]
}
```

#### 6.1.2 高维数据处理
- LASSO型惩罚
- 组稀疏惩罚
- 维度降维技术

### 6.2 模型扩展

#### 6.2.1 时变系数
$g(E(Y|X)) = \beta_0(t) + \sum_{j=1}^p f_j(X_j, t)$

#### 6.2.2 空间效应
$g(E(Y|X)) = \beta_0 + \sum_{j=1}^p f_j(X_j) + s(lat, lon)$

### 6.3 前沿研究
```python
research_topics = {
    "理论发展": [
        "非参数置信区间",
        "变量选择理论",
        "最优收敛速率"
    ],
    "算法创新": [
        "深度GAM",
        "分布式GAM",
        "自适应基函数"
    ],
    "应用拓展": [
        "因果推断",
        "生存分析",
        "图像处理"
    ]
}
```
```