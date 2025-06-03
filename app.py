#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG FastAPI Web服务
提供RESTful API接口
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import os

from graphrag_demo_01.graphrag_service import GraphRAGService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局服务实例
graphrag_service = GraphRAGService()

# Pydantic模型定义
class InitRequest(BaseModel):
    """初始化请求模型"""
    project_dir: str = Field(..., description="项目目录路径")
    config_data: Optional[Dict[str, Any]] = Field(None, description="可选的配置数据")

class IndexRequest(BaseModel):
    """索引构建请求模型"""
    method: str = Field("standard", description="索引方法: standard 或 fast")
    is_update: bool = Field(False, description="是否为增量更新")

class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    community_level: int = Field(0, description="社区层级")
    temperature: float = Field(0.6, description="温度参数")
    top_p: float = Field(0.9, description="Top-p参数")
    max_tokens: int = Field(4096, description="最大令牌数")

class BasicSearchRequest(BaseModel):
    """基本搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    k: int = Field(5, description="返回结果数量")
    temperature: float = Field(0.6, description="温度参数")
    top_p: float = Field(0.9, description="Top-p参数")
    max_tokens: int = Field(4096, description="最大令牌数")

class ApiResponse(BaseModel):
    """API响应模型"""
    status: str = Field(..., description="响应状态")
    message: Optional[str] = Field(None, description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    error: Optional[str] = Field(None, description="错误信息")

# FastAPI应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("GraphRAG Web服务启动")
    
    # 自动初始化GraphRAG服务
    try:
        # 使用当前目录作为默认项目目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"正在初始化GraphRAG服务，项目目录: {current_dir}")
        
        result = await graphrag_service.initialize(project_dir=current_dir)
        
        if result["status"] == "success":
            logger.info(f"GraphRAG服务初始化成功，服务状态: {result['service_status']}, 索引状态: {result['index_status']}")
        else:
            logger.warning(f"GraphRAG服务初始化失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        logger.error(f"GraphRAG服务自动初始化失败: {str(e)}")
        logger.info("服务将继续启动，但需要手动初始化GraphRAG功能")
    
    yield
    logger.info("GraphRAG Web服务关闭")

# 创建FastAPI应用
app = FastAPI(
    title="GraphRAG Web API",
    description="基于GraphRAG的知识图谱搜索API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务（用于提供HTML界面）
app.mount("/static", StaticFiles(directory="."), name="static")

# ========== 配置和状态管理接口 ==========

@app.post("/api/init", response_model=ApiResponse)
async def initialize_service(request: InitRequest):
    """初始化GraphRAG服务"""
    try:
        result = await graphrag_service.initialize(
            project_dir=request.project_dir,
            config_data=request.config_data
        )
        
        if result["status"] == "success":
            return ApiResponse(
                status="success",
                message=result["message"],
                data=result
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"初始化服务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=ApiResponse)
async def get_service_status():
    """获取服务状态"""
    try:
        status = graphrag_service.get_status()
        return ApiResponse(
            status="success",
            data=status
        )
    except Exception as e:
        logger.error(f"获取服务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== 索引管理接口 ==========

@app.post("/api/index/build", response_model=ApiResponse)
async def build_index(request: IndexRequest, background_tasks: BackgroundTasks):
    """构建或更新索引"""
    try:
        # 在后台任务中执行索引构建
        result = await graphrag_service.build_index(
            method=request.method,
            is_update=request.is_update
        )
        
        if result["status"] == "success":
            return ApiResponse(
                status="success",
                message=result["message"],
                data=result
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"构建索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index/load", response_model=ApiResponse)
async def load_index():
    """加载索引数据"""
    try:
        result = await graphrag_service.load_index()
        
        if result["status"] == "success":
            return ApiResponse(
                status="success",
                message=result["message"],
                data=result
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"加载索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/index/status", response_model=ApiResponse)
async def get_index_status():
    """获取索引状态"""
    try:
        status = graphrag_service.get_status()
        return ApiResponse(
            status="success",
            data={
                "index_status": status["index_status"],
                "has_index_data": status["has_index_data"]
            }
        )
    except Exception as e:
        logger.error(f"获取索引状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== 搜索接口 ==========

@app.post("/api/search/global", response_model=ApiResponse)
async def global_search(request: SearchRequest):
    """全局搜索"""
    try:
        result = await graphrag_service.global_search(
            query=request.query,
            community_level=request.community_level,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        if result["status"] == "success":
            return ApiResponse(
                status="success",
                data=result
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"全局搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/local", response_model=ApiResponse)
async def local_search(request: SearchRequest):
    """本地搜索"""
    try:
        result = await graphrag_service.local_search(
            query=request.query,
            community_level=request.community_level,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        if result["status"] == "success":
            return ApiResponse(
                status="success",
                data=result
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"本地搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/drift", response_model=ApiResponse)
async def drift_search(request: SearchRequest):
    """DRIFT搜索"""
    try:
        result = await graphrag_service.drift_search(
            query=request.query,
            community_level=request.community_level,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        if result["status"] == "success":
            return ApiResponse(
                status="success",
                data=result
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"DRIFT搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/basic", response_model=ApiResponse)
async def basic_search(request: BasicSearchRequest):
    """基本搜索"""
    try:
        result = await graphrag_service.basic_search(
            query=request.query,
            k=request.k,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        if result["status"] == "success":
            return ApiResponse(
                status="success",
                data=result
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"基本搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== 流式搜索接口 ==========

@app.post("/api/search/global/stream")
async def global_search_streaming(request: SearchRequest):
    """流式全局搜索"""
    try:
        async def generate():
            try:
                # 创建一个队列来处理流式数据
                import asyncio
                queue = asyncio.Queue()
                
                def callback(chunk: str):
                    """同步回调转异步"""
                    try:
                        loop = asyncio.get_event_loop()
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    except Exception:
                        pass
                
                # 启动搜索任务
                async def search_task():
                    try:
                        full_response, context = await graphrag_service.global_search_streaming(
                            query=request.query,
                            community_level=request.community_level,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            max_tokens=request.max_tokens,
                            callback=callback
                        )
                        # 搜索完成后发送结束标记
                        await queue.put("[DONE]")
                    except Exception as e:
                        await queue.put(f"ERROR: {str(e)}")
                
                # 启动搜索任务
                task = asyncio.create_task(search_task())
                
                # 实时发送数据
                while True:
                    try:
                        # 等待数据，设置超时避免无限等待
                        chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                        
                        if chunk == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        elif chunk.startswith("ERROR:"):
                            yield f"data: {chunk}\n\n"
                            break
                        else:
                            yield f"data: {chunk}\n\n"
                            
                    except asyncio.TimeoutError:
                        # 超时时发送心跳
                        yield "data: \n\n"
                        continue
                    except Exception as e:
                        yield f"data: ERROR: {str(e)}\n\n"
                        break
                
                # 确保任务完成
                if not task.done():
                    task.cancel()
                    
            except Exception as e:
                yield f"data: ERROR: {str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"流式全局搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/local/stream")
async def local_search_streaming(request: SearchRequest):
    """流式本地搜索"""
    try:
        async def generate():
            try:
                import asyncio
                queue = asyncio.Queue()
                
                def callback(chunk: str):
                    try:
                        loop = asyncio.get_event_loop()
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    except Exception:
                        pass
                
                async def search_task():
                    try:
                        full_response, context = await graphrag_service.local_search_streaming(
                            query=request.query,
                            community_level=request.community_level,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            max_tokens=request.max_tokens,
                            callback=callback
                        )
                        await queue.put("[DONE]")
                    except Exception as e:
                        await queue.put(f"ERROR: {str(e)}")
                
                task = asyncio.create_task(search_task())
                
                while True:
                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                        
                        if chunk == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        elif chunk.startswith("ERROR:"):
                            yield f"data: {chunk}\n\n"
                            break
                        else:
                            yield f"data: {chunk}\n\n"
                            
                    except asyncio.TimeoutError:
                        yield "data: \n\n"
                        continue
                    except Exception as e:
                        yield f"data: ERROR: {str(e)}\n\n"
                        break
                
                if not task.done():
                    task.cancel()
                    
            except Exception as e:
                yield f"data: ERROR: {str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"流式本地搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/drift/stream")
async def drift_search_streaming(request: SearchRequest):
    """流式DRIFT搜索"""
    try:
        async def generate():
            try:
                import asyncio
                queue = asyncio.Queue()
                
                def callback(chunk: str):
                    try:
                        loop = asyncio.get_event_loop()
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    except Exception:
                        pass
                
                async def search_task():
                    try:
                        full_response, context = await graphrag_service.drift_search_streaming(
                            query=request.query,
                            community_level=request.community_level,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            max_tokens=request.max_tokens,
                            callback=callback
                        )
                        await queue.put("[DONE]")
                    except Exception as e:
                        await queue.put(f"ERROR: {str(e)}")
                
                task = asyncio.create_task(search_task())
                
                while True:
                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                        
                        if chunk == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        elif chunk.startswith("ERROR:"):
                            yield f"data: {chunk}\n\n"
                            break
                        else:
                            yield f"data: {chunk}\n\n"
                            
                    except asyncio.TimeoutError:
                        yield "data: \n\n"
                        continue
                    except Exception as e:
                        yield f"data: ERROR: {str(e)}\n\n"
                        break
                
                if not task.done():
                    task.cancel()
                    
            except Exception as e:
                yield f"data: ERROR: {str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"流式DRIFT搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/basic/stream")
async def basic_search_streaming(request: BasicSearchRequest):
    """流式基本搜索"""
    try:
        async def generate():
            try:
                import asyncio
                queue = asyncio.Queue()
                
                def callback(chunk: str):
                    try:
                        loop = asyncio.get_event_loop()
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    except Exception:
                        pass
                
                async def search_task():
                    try:
                        full_response, context = await graphrag_service.basic_search_streaming(
                            query=request.query,
                            k=request.k,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            max_tokens=request.max_tokens,
                            callback=callback
                        )
                        await queue.put("[DONE]")
                    except Exception as e:
                        await queue.put(f"ERROR: {str(e)}")
                
                task = asyncio.create_task(search_task())
                
                while True:
                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                        
                        if chunk == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        elif chunk.startswith("ERROR:"):
                            yield f"data: {chunk}\n\n"
                            break
                        else:
                            yield f"data: {chunk}\n\n"
                            
                    except asyncio.TimeoutError:
                        yield "data: \n\n"
                        continue
                    except Exception as e:
                        yield f"data: ERROR: {str(e)}\n\n"
                        break
                
                if not task.done():
                    task.cancel()
                    
            except Exception as e:
                yield f"data: ERROR: {str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"流式基本搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== 工具接口 ==========

@app.get("/api/context", response_model=ApiResponse)
async def get_context():
    """获取最近查询的上下文数据"""
    try:
        context = graphrag_service.get_context()
        return ApiResponse(
            status="success",
            data={"context": context}
        )
    except Exception as e:
        logger.error(f"获取上下文失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== 健康检查接口 ==========

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "GraphRAG Web API"}

@app.get("/")
async def root():
    """根路径 - 提供HTML界面"""
    # 检查HTML文件是否存在
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {
            "message": "GraphRAG Web API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "note": "HTML界面文件不存在，请确保index.html文件在正确位置"
        }

@app.get("/api")
async def api_info():
    """API信息"""
    return {
        "message": "GraphRAG Web API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# 启动服务
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 