# 双引擎智能代码检索系统

基于关键字+语义混合检索的智能代码分析服务，支持多语言代码解析、功能定位和自动化测试生成。

## 系统介绍

本系统采用**双引擎智能检索**技术，结合关键字检索和语义向量检索的优势，提供高效准确的代码分析服务：

- **关键字检索引擎**：基于代码函数名、变量名等关键信息进行精确匹配
- **语义向量检索引擎**：基于代码语义相似度进行智能匹配
- **混合检索策略**：结合两种检索方式的优势，提供更全面的代码分析结果

### 核心特性

- **智能代码检索**：支持关键字+语义混合检索
- **代码知识图谱**：构建项目代码结构关系图
- **LLM智能分析**：基于大语言模型的代码理解
- **自动化测试**：自动生成和执行测试代码

## 快速开始



## 项目结构

```
code_reviewer_agent/
├── app/                                             # 主应用代码
│   ├── api/                                         # API接口层
│   │   ├── __init__
│   │   └── endpoints.py                             # API端点定义和路由（未编译）
│   ├── core/                                        # 核心功能模块
│   │   ├── __init__.cpython-311-darwin.so
│   │   ├── config.cpython-311-darwin.so             # 配置管理
│   │   ├── database.cpython-311-darwin.so           # 数据库连接和操作
│   │   └── llm_service.py                           # LLM服务接口（未编译）
│   ├── models/                                      # 数据模型层
│   │   ├── __init__.cpython-311-darwin.so
│   │   ├── database_models.cpython-311-darwin.so    # 数据库模型定义
│   │   └── response_models.cpython-311-darwin.so    # API响应模型
│   ├── services/                                    # 业务逻辑服务层
│   │   ├── __init__.cpython-311-darwin.so
│   │   ├── ckg_service.cpython-311-darwin.so        # 代码知识图谱服务
│   │   ├── code_parser.cpython-311-darwin.so        # 代码解析服务
│   │   ├── test_executor.cpython-311-darwin.so      # 测试执行服务
│   │   └── vector_search_service.cpython-311-darwin.so # 向量检索服务
│   └── utils/                                       # 工具类
│       ├── __init__.cpython-311-darwin.so
│       └── file_processor.cpython-311-darwin.so     # 文件处理工具
├── scripts/                                         # 脚本文件
│   └── init_db.cpython-311-darwin.so                # 数据库初始化脚本
├── uploads/                                         # 上传文件临时目录
├── data/                                            # 数据目录
│   └── vector_index/                                # 向量索引存储
├── requirements.txt                                 # Python依赖
├── run.cpython-311-darwin.so                        # 主启动文件
├── start.sh                                         # 启动脚本
├── Dockerfile                    
├── docker-compose.yml            
└── README.md                     
```

## API接口介绍

### 推荐使用接口2：关键字检索+向量检索

结合了关键字检索的精确性和向量检索的语义理解能力。

#### 接口地址
```
POST /analyze-v2
```

#### 请求参数
```python
{
    "problem_description": str,    # 功能需求描述
    "code_zip": UploadFile,        # 包含项目代码的ZIP文件
    "generate_tests": bool,        # 是否生成测试代码 (默认: False)
    "execute_tests": bool          # 是否执行测试代码 (默认: False)
    "use_vector_search": bool      # 是否启用向量搜索增强 (默认: False)
}
```

#### 执行流程
1. **读取ZIP文件** → 接收并验证上传的代码文件
2. **解压和扫描代码文件** → 提取项目结构和代码内容
3. **构建代码知识图谱** → 分析代码依赖关系和结构
4. **LLM获取问题关键字** → 从需求描述中提取关键信息
5. **关键词搜索 + 向量搜索** → 双引擎检索相关代码
6. **LLM分析代码相关性** → 智能分析检索结果的关联性
7. **生成执行计划** → 制定代码分析和测试策略
8. **生成测试代码** → 根据分析结果生成测试用例
9. **执行测试代码** → 运行测试并收集结果
10. **输出结果** → 返回分析报告和测试结果

### 其他检索方案

#### 接口1：/analyze-v1，对每个文件调用LLM分析（已弃用）
- **特点**: 逐个文件分析，准确度高但性能较差
- **适用场景**: 小规模项目或对准确性要求极高的场景

#### 接口3：/analyze-v3，函数名关键字检索+向量检索
- **特点**: 基于函数名进行关键字匹配，结合向量检索
- **适用场景**: 需要快速定位特定功能的场景

#### 接口4：/analyze-v4，纯向量检索
- **特点**: 仅使用语义向量检索
- **适用场景**: 仅测试效果时使用


## 性能优化建议

### 1. 模型优化
- **使用更强大的本地代码模型**：推荐使用CodeBERT等专门针对代码优化的模型，实测效果较好

### 2. 向量索引优化
- **增量更新机制**：与GitHub连接，只有检测到文件改变才会重新构建向量索引

