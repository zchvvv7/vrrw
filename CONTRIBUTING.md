## 代码风格规范

本项目Python代码风格严格参照 [PEP 8](https://peps.python.org/pep-0008/)，所有成员提交代码前请遵守以下规范。

### 1. 命名规范
 
| 类型        | 规范        | 示例                                 |
|-----------|-----------|------------------------------------|
| 变量 / 函数   | 全小写 + 下划线 | `detected_object`, `get_data()`    |
| 类名        | 每个单词首字母大写 | `RiskEvaluator`                    |
| 常量        | 全大写 + 下划线 | `MAX_SIZE`     |
| 私有变量 / 函数 | 前面加一个下划线  | `_internal_value`, `_read_image()` |
| 模块 / 文件名  | 全小写 + 下划线 | `camera_reader.py`                 |

### 2. 缩进与空格

- 统一使用 4 个空格缩进，不要使用 Tab
- 逗号后面加空格，例如 `func(a, b, c)` ，不要写 `func(a,b,c)`
- 赋值等号两边加空格，例如 `x = 1`
  - 函数默认参数例外，等号两边不加空格，例如 `def f(x=1):`
- 运算符两边加空格，例如 `a + b` ， `if a == b:`

### 3. 行长度
 
- 每行代码不超过 79 个字符（文档字符串/注释同样遵守）


### 4. 空行
 
- 顶层函数、类定义之间空 2 行
- 类内部的方法之间空 1 行
- 逻辑关系不大的代码块之间可适当空 1 行，提升可读性

### 5. 导入
 
- 每个模块单独一行导入，不要写成一行导入多个：
```python
  # 正确示例
  import os
  import sys
 
  # 错误示例
  import os, sys
```
- 导入顺序分三组，组间空一行：
  - 标准库（如 `os`, `sys`）
  - 第三方库（如 `numpy`, `requests`）
  - 项目内部模块（如 `from .utils import helper`）

### 6. 类型标注

所有函数的参数和返回值都必须加类型标注。
 
```python
def get_average_order_value(user_id: int) -> float:
    orders = fetch_orders(user_id)
    return sum(orders) / len(orders)
```
 
- `user_id: int` 说明参数应传入整数
- `-> float` 说明返回值是浮点数

### 7. 其他规范

- 字符串统一使用双引号 `"` 而非单引号
- 条件判断中 `if is_valid == True:` 简化为 `if is_valid:`
- 不要使用可变对象作为函数默认参数，例如 `def f(x=[]):`

---

## 注释规范

本项目所有python文件都需要注释，所有成员进行标注时请遵守以下规范。

### 1. 文件头注释

每个 `.py` 文件开头必须有一段注释，说明该文件的用途、作者和创建/修改日期，空一行后开始import。

```python
"""
文件名: data_loader.py
用途: 从数据库读取原始数据
作者: 小张
创建日期: 2026-07-15
最后修改日期: 2026-07-15
"""
 
import os
import pandas as pd
```

### 2. 函数注释

每个函数定义前必须有一行简短注释，说明这个函数做什么，不需要逐行解释实现细节。

```python
# 计算用户的平均订单金额
def get_average_order_value(user_id: int) -> float:
    orders = fetch_orders(user_id)
    return sum(orders) / len(orders)
```

### 3. 类注释

类定义下方用 docstring 说明类的用途。
 
```python
class UserProfile:
    """存储并管理单个用户的基础信息与偏好设置"""
 
    def __init__(self, user_id):
        self.user_id = user_id
```

### 4. TODO / FIXME 标记
 
标记未完成或待修复的代码，方便搜索定位。
 
```python
# TODO: 这个函数的功能还未实现，仅占位
# FIXME: 这里代码有bug需要修复
```

---

## Git 协作规范

不要直接在 `main` 分支上改代码，新任务先建一个分支。

### 1. 分支命名规范

命名格式统一为：类型 / 简短描述，描述部分用小写 + 连字符，简单说明这个分支做什么。

| 前缀 | 用途 | 示例 |
|---|---|---|
| `feature/` | 新功能开发 | `feature/login-page` |
| `fix/` | 修复 bug | `fix/login-crash-bug` |
| `refactor/` | 重构代码，不改变功能 | `refactor/data-loader` |
| `docs/` | 只改文档 | `docs/update-readme` |
| `test/` | 增加或修改测试 | `test/user-login` |

### 2. Commit 规范

每次提交都要写清楚这次改了什么，不要写没有意义的宽泛说明，例如`git commit -m "update"`。

Commit统一格式为：类型: 简要叙述。

| 类型 | 说明 |
|---|---|
| `feat` | 新增功能 |
| `fix` | 修复 bug |
| `docs` | 只改动文档 |
| `style` | 代码格式调整（不影响逻辑） |
| `refactor` | 重构代码（不新增功能，也不修 bug） |
| `test` | 增加或修改测试 |
| `chore` | 杂项，如更新依赖、配置文件 |

示例

- feat: 完成用户登录页面
- docs: 更新 README 使用说明

### 3. 提交前检查

每次 git push 前请确认：

- 代码能正常运行，没有报错
- 已经 `git pull` 过，本地是基于最新代码修改的
- 没有遗留的调试代码，例如多余的 `print()`

### 4. 合并流程

1. 在自己的分支上完成开发并推送 `git push origin feature/xxx`
2. 点击 Compare & pull request 发起 PR
3. 确认没有问题后合并进 `main`
4. 合并后可删除已完成的分支，保持仓库整洁