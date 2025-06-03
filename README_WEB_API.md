# GraphRAG Web API 使用指南

基于 GraphRAG 的知识图谱搜索 Web API 服务，提供完整的索引构建和多种搜索功能。

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 安装Web服务依赖  
pip install -r requirements_web.txt
```

### 2. 准备数据

将要处理的文档放入 `input` 目录：
```bash
mkdir -p input
cp your_documents.txt input/
```

### 3. 配置环境（可选）

如果使用 OpenAI 等 API 服务，创建 `.env` 文件：
```bash
# .env
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 4. 启动服务

```bash
# 使用启动脚本（推荐）
python start_web_service.py

# 或直接启动
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. 访问服务

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs  
- **健康检查**: http://localhost:8000/health

## 📚 API 接口文档

### 配置和状态管理

#### GET /health
健康检查接口
```json
{
  "status": "healthy",
  "service_status": "ready",
  "index_status": "ready"
}
```

#### GET /api/status  
获取服务状态
```json
{
  "status": "success",
  "data": {
    "service_status": "ready",
    "index_status": "ready", 
    "project_dir": "/path/to/project",
    "has_index_data": true
  }
}
```

#### POST /api/init
初始化服务
```json
{
  "project_dir": ".",
  "config_data": {}  // 可选
}
```

### 索引管理

#### POST /api/index/build
构建或更新索引
```json
{
  "method": "standard",  // "standard" 或 "fast"
  "is_update": false     // 是否增量更新
}
```

#### POST /api/index/load
加载现有索引
```json
{
  "status": "success",
  "message": "索引数据加载成功"
}
```

#### GET /api/index/status
获取索引状态
```json
{
  "status": "success",
  "data": {
    "index_status": "ready",
    "has_index_data": true
  }
}
```

### 搜索接口

#### POST /api/search/global
全局搜索 - 基于社区报告的高层次回答
```json
{
  "query": "图形算法有哪些应用？",
  "community_level": 0,
  "temperature": 0.6,
  "top_p": 0.9,
  "max_tokens": 4096
}
```

#### POST /api/search/local  
本地搜索 - 基于实体和关系的详细回答
```json
{
  "query": "图形算法有哪些应用？",
  "community_level": 0,
  "temperature": 0.6,
  "top_p": 0.9,
  "max_tokens": 4096
}
```

#### POST /api/search/drift
DRIFT搜索 - 混合全局和本地的搜索方法
```json
{
  "query": "图形算法有哪些应用？",
  "community_level": 0,
  "temperature": 0.6,
  "top_p": 0.9,
  "max_tokens": 4096
}
```

#### POST /api/search/basic
基本搜索 - 基于向量相似度的简单搜索
```json
{
  "query": "图形算法有哪些应用？",
  "k": 5,
  "temperature": 0.6,
  "top_p": 0.9,
  "max_tokens": 4096
}
```

### 流式搜索接口

所有搜索类型都支持流式响应：
- `POST /api/search/global/stream`
- `POST /api/search/local/stream`  
- `POST /api/search/drift/stream`
- `POST /api/search/basic/stream`

流式响应格式：
```
data: {"type": "chunk", "content": "部分回答内容"}
data: {"type": "chunk", "content": "更多内容"}
data: {"type": "done"}
```

### 其他接口

#### GET /api/context
获取最近搜索的上下文数据
```json
{
  "status": "success",
  "data": {
    "context": {
      "reports": {...},
      "entities": {...}
    }
  }
}
```

## 🧪 测试

### 快速测试
```bash
python test_web_api.py quick
```

### 完整测试
```bash
python test_web_api.py
```

测试包括：
- ✅ 健康检查
- ✅ 服务状态
- ✅ 服务初始化  
- ✅ 索引加载
- ✅ 搜索功能（如果索引可用）
- ✅ 流式搜索
- ✅ 上下文获取

## 🔧 高级配置

### 启动参数

```bash
python start_web_service.py --help

# 自定义端口
python start_web_service.py --port 8080

# 开发模式（自动重载）
python start_web_service.py --reload

# 跳过环境检查
python start_web_service.py --skip-checks
```

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API密钥 | - |
| `OPENAI_BASE_URL` | OpenAI API基础URL | https://api.openai.com/v1 |
| `GRAPHRAG_LOG_LEVEL` | 日志级别 | INFO |

### 配置文件

服务会自动生成 `settings.yaml` 配置文件，可以手动编辑来调整：
- 模型配置
- 输入输出路径
- 索引参数
- 搜索参数

## 📁 项目结构

```
graphrag_demo_01/
├── app.py                 # FastAPI Web服务主文件
├── graphrag_service.py    # GraphRAG核心服务类
├── main.py               # 命令行版本（参考）
├── start_web_service.py  # Web服务启动脚本
├── test_web_api.py       # API测试脚本
├── requirements.txt      # 核心依赖
├── requirements_web.txt  # Web服务依赖
├── input/               # 输入文档目录
├── output/              # 索引输出目录
├── cache/               # 缓存目录
├── logs/                # 日志目录
├── prompts/             # 提示词目录
└── .env                 # 环境配置（可选）
```

## 🔍 使用示例

### Python客户端示例

```python
import httpx
import asyncio

async def example_usage():
    async with httpx.AsyncClient() as client:
        
        # 1. 检查服务状态
        response = await client.get("http://localhost:8000/api/status")
        print(response.json())
        
        # 2. 全局搜索
        search_data = {
            "query": "图形算法的应用",
            "temperature": 0.7
        }
        response = await client.post(
            "http://localhost:8000/api/search/global",
            json=search_data
        )
        result = response.json()
        print(result["data"]["result"])
        
        # 3. 流式搜索
        async with client.stream(
            "POST",
            "http://localhost:8000/api/search/local/stream", 
            json=search_data
        ) as response:
            async for chunk in response.aiter_text():
                print(chunk)

# 运行示例
asyncio.run(example_usage())
```

### JavaScript客户端示例

```javascript
// 普通搜索
async function search(query) {
    const response = await fetch('http://localhost:8000/api/search/global', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query: query,
            temperature: 0.7
        })
    });
    const result = await response.json();
    return result.data.result;
}

// 流式搜索
async function streamSearch(query) {
    const response = await fetch('http://localhost:8000/api/search/global/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query})
    });
    
    const reader = response.body.getReader();
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        
        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'chunk') {
                    console.log(data.content);
                }
            }
        }
    }
}
```

## 🚨 常见问题

### Q: 服务启动失败？
A: 检查依赖是否完整安装，端口是否被占用

### Q: 搜索返回错误？
A: 确保索引已构建且加载成功

### Q: 索引构建失败？
A: 检查input目录是否有文档，.env配置是否正确

### Q: 流式搜索无响应？
A: 检查客户端是否正确处理Server-Sent Events格式

## 📞 支持

- 查看日志：`logs/` 目录下的日志文件
- API文档：http://localhost:8000/docs
- 测试工具：`python test_web_api.py`

## 🎯 下一步

1. **扩展搜索功能**：添加多模态搜索、语义搜索等
2. **性能优化**：缓存、并发、负载均衡
3. **前端界面**：开发完整的Web界面
4. **部署方案**：Docker化、云部署配置