# GraphRAG Web 问答界面

这是一个基于 GraphRAG 的智能问答 Web 界面，提供了完整的图形化操作界面和 RESTful API 服务。

## 🌟 功能特性

### 🔍 多种搜索方法
- **本地搜索 (Local Search)**: 基于实体的精确查询，适合特定实体相关问题
- **全局搜索 (Global Search)**: 整个数据集的综合分析，适合宏观问题
- **DRIFT搜索 (DRIFT Search)**: 动态递归信息查找，结合全局和本地搜索优势
- **基本搜索 (Basic Search)**: 快速文本检索，适合简单查询

### 💬 智能交互
- **流式输出**: 实时显示AI回答过程，提供更好的用户体验
- **非流式输出**: 一次性返回完整答案
- **上下文预览**: 显示搜索过程中使用的数据源和上下文信息

### ⚙️ 高级参数控制
- **温度 (Temperature)**: 控制回答的创造性 (0-2)
- **Top-p**: 控制回答的多样性 (0-1)
- **最大令牌数**: 限制回答长度 (100-8192)
- **社区层级**: 控制搜索的社区深度 (0-5)
- **K值**: 基本搜索返回的结果数量 (1-20)

### 🎨 现代化界面
- 响应式设计，支持桌面和移动设备
- 实时状态监控
- 优雅的消息气泡设计
- 流畅的动画效果

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：

```bash
pip install fastapi uvicorn python-multipart
```

### 2. 配置环境变量

在项目目录下创建 `.env` 文件：

```env
# DeepSeek API (用于聊天模型)
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# 阿里云API (用于嵌入模型)
ALIYUN_API_KEY=your_aliyun_api_key
ALIYUN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. 准备数据

确保项目目录结构如下：

```
graphrag_demo_01/
├── input/              # 输入文档
│   └── *.txt          # 文本文件
├── output/            # 索引输出 (构建后生成)
├── cache/             # 缓存目录
├── prompts/           # 提示词文件 (自动生成)
├── app.py             # FastAPI 应用
├── graphrag_service.py # GraphRAG 服务
├── index.html         # Web 界面
├── start_server.py    # 启动脚本
└── .env               # 环境变量
```

### 4. 启动服务

#### 方法一：使用启动脚本 (推荐)

```bash
python start_server.py
```

#### 方法二：直接启动

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 访问界面

服务启动后，访问以下地址：

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 📖 使用指南

### 首次使用

1. **初始化服务**: 访问 Web 界面后，系统会自动检查服务状态
2. **构建索引**: 如果索引未构建，需要先构建索引（通过API或命令行）
3. **开始问答**: 选择搜索方法，输入问题，开始智能问答

### 搜索方法选择

| 搜索方法 | 适用场景 | 特点 |
|---------|---------|------|
| 本地搜索 | 特定实体查询 | 精确、快速、基于实体关系 |
| 全局搜索 | 宏观分析问题 | 全面、深入、基于社区报告 |
| DRIFT搜索 | 复杂推理问题 | 智能、递归、结合多种方法 |
| 基本搜索 | 简单文本检索 | 快速、直接、基于文本相似度 |

### 参数调优建议

- **创造性回答**: 提高温度值 (0.8-1.2)
- **准确性回答**: 降低温度值 (0.1-0.5)
- **多样性回答**: 调整 Top-p 值 (0.7-0.95)
- **长回答**: 增加最大令牌数
- **深度分析**: 提高社区层级

## 🔧 API 接口

### 服务管理

- `GET /api/status` - 获取服务状态
- `POST /api/init` - 初始化服务
- `POST /api/index/build` - 构建索引
- `POST /api/index/load` - 加载索引

### 搜索接口

#### 非流式搜索
- `POST /api/search/local` - 本地搜索
- `POST /api/search/global` - 全局搜索
- `POST /api/search/drift` - DRIFT搜索
- `POST /api/search/basic` - 基本搜索

#### 流式搜索
- `POST /api/search/local/stream` - 流式本地搜索
- `POST /api/search/global/stream` - 流式全局搜索
- `POST /api/search/drift/stream` - 流式DRIFT搜索
- `POST /api/search/basic/stream` - 流式基本搜索

### 请求示例

```json
{
  "query": "什么是人工智能？",
  "community_level": 0,
  "temperature": 0.6,
  "top_p": 0.9,
  "max_tokens": 4096
}
```

## 🛠️ 故障排除

### 常见问题

1. **服务无法启动**
   - 检查端口 8000 是否被占用
   - 确认依赖包已正确安装
   - 检查环境变量配置

2. **索引构建失败**
   - 确认 API 密钥配置正确
   - 检查输入目录是否有文档
   - 查看日志错误信息

3. **搜索无结果**
   - 确认索引已成功构建
   - 检查查询语句是否合理
   - 尝试不同的搜索方法

4. **跨域问题**
   - 确认 CORS 中间件已正确配置
   - 检查浏览器控制台错误信息

### 日志查看

服务运行时会输出详细日志，包括：
- 服务启动状态
- API 请求处理
- 搜索执行过程
- 错误信息详情

## 📝 开发说明

### 文件结构

- `app.py`: FastAPI 应用主文件，定义所有 API 接口
- `graphrag_service.py`: GraphRAG 服务封装，处理核心业务逻辑
- `index.html`: Web 前端界面，提供用户交互
- `start_server.py`: 服务启动脚本，简化启动流程

### 扩展开发

如需扩展功能，可以：
1. 在 `app.py` 中添加新的 API 接口
2. 在 `graphrag_service.py` 中扩展业务逻辑
3. 在 `index.html` 中添加新的前端功能

## 📄 许可证

本项目基于 GraphRAG 开源项目开发，遵循相应的开源许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！ 