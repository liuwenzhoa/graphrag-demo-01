#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG Web API 测试脚本
测试所有API接口的功能
"""

import asyncio
import json
import time
import os
from typing import Dict, Any, Optional
import httpx
import pytest

# 基础配置
BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0

class GraphRAGAPITester:
    """GraphRAG API测试类"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.client:
            await self.client.aclose()
    
    async def test_health(self) -> Dict[str, Any]:
        """测试健康检查"""
        print("🔍 测试健康检查...")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            result = {
                "test_name": "health_check",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": None
            }
            print(f"✅ 健康检查成功: {result['data']}")
            return result
        except Exception as e:
            result = {
                "test_name": "health_check", 
                "success": False,
                "error": str(e)
            }
            print(f"❌ 健康检查失败: {e}")
            return result
    
    async def test_service_status(self) -> Dict[str, Any]:
        """测试获取服务状态"""
        print("🔍 测试获取服务状态...")
        try:
            response = await self.client.get(f"{self.base_url}/api/status")
            result = {
                "test_name": "service_status",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": None
            }
            if result["success"]:
                print(f"✅ 服务状态获取成功: {result['data']['data']}")
            else:
                print(f"❌ 服务状态获取失败: {response.text}")
            return result
        except Exception as e:
            result = {
                "test_name": "service_status",
                "success": False, 
                "error": str(e)
            }
            print(f"❌ 服务状态获取异常: {e}")
            return result
    
    async def test_initialize(self, project_dir: str = ".") -> Dict[str, Any]:
        """测试初始化服务"""
        print(f"🔍 测试初始化服务 (项目目录: {project_dir})...")
        try:
            payload = {"project_dir": project_dir}
            response = await self.client.post(
                f"{self.base_url}/api/init",
                json=payload
            )
            result = {
                "test_name": "initialize",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": None
            }
            if result["success"]:
                print(f"✅ 服务初始化成功: {result['data']['data']}")
            else:
                print(f"❌ 服务初始化失败: {response.text}")
            return result
        except Exception as e:
            result = {
                "test_name": "initialize",
                "success": False,
                "error": str(e)
            }
            print(f"❌ 服务初始化异常: {e}")
            return result
    
    async def test_load_index(self) -> Dict[str, Any]:
        """测试加载索引"""
        print("🔍 测试加载索引...")
        try:
            response = await self.client.post(f"{self.base_url}/api/index/load")
            result = {
                "test_name": "load_index",
                "status_code": response.status_code, 
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": None
            }
            if result["success"]:
                print(f"✅ 索引加载成功: {result['data']['message']}")
            else:
                print(f"⚠️ 索引加载失败（可能尚未构建）: {response.text}")
            return result
        except Exception as e:
            result = {
                "test_name": "load_index",
                "success": False,
                "error": str(e)
            }
            print(f"❌ 索引加载异常: {e}")
            return result
    
    async def test_index_status(self) -> Dict[str, Any]:
        """测试获取索引状态"""
        print("🔍 测试获取索引状态...")
        try:
            response = await self.client.get(f"{self.base_url}/api/index/status")
            result = {
                "test_name": "index_status",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": None
            }
            if result["success"]:
                print(f"✅ 索引状态获取成功: {result['data']['data']}")
            else:
                print(f"❌ 索引状态获取失败: {response.text}")
            return result
        except Exception as e:
            result = {
                "test_name": "index_status",
                "success": False,
                "error": str(e)
            }
            print(f"❌ 索引状态获取异常: {e}")
            return result
    
    async def test_build_index(self, method: str = "standard", is_update: bool = False) -> Dict[str, Any]:
        """测试构建索引"""
        operation = "更新索引" if is_update else "构建索引"
        print(f"🔍 测试{operation} (方法: {method})...")
        try:
            payload = {
                "method": method,
                "is_update": is_update
            }
            response = await self.client.post(
                f"{self.base_url}/api/index/build",
                json=payload,
                timeout=300.0  # 索引构建可能需要较长时间
            )
            result = {
                "test_name": "build_index",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": None
            }
            if result["success"]:
                print(f"✅ {operation}成功: {result['data']['message']}")
            else:
                print(f"❌ {operation}失败: {response.text}")
            return result
        except Exception as e:
            result = {
                "test_name": "build_index",
                "success": False,
                "error": str(e)
            }
            print(f"❌ {operation}异常: {e}")
            return result
    
    async def test_search(self, search_type: str, query: str, **kwargs) -> Dict[str, Any]:
        """测试搜索功能"""
        print(f"🔍 测试{search_type}搜索: {query}")
        try:
            if search_type == "basic":
                payload = {
                    "query": query,
                    "k": kwargs.get("k", 5),
                    "temperature": kwargs.get("temperature", 0.6),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", 4096)
                }
            else:
                payload = {
                    "query": query,
                    "community_level": kwargs.get("community_level", 0),
                    "temperature": kwargs.get("temperature", 0.6),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", 4096)
                }
            
            response = await self.client.post(
                f"{self.base_url}/api/search/{search_type}",
                json=payload,
                timeout=120.0
            )
            result = {
                "test_name": f"{search_type}_search",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": None,
                "query": query
            }
            if result["success"]:
                answer = result["data"]["data"]["result"]
                print(f"✅ {search_type}搜索成功:")
                print(f"   查询: {query}")
                print(f"   回答: {answer[:200]}..." if len(answer) > 200 else f"   回答: {answer}")
            else:
                print(f"❌ {search_type}搜索失败: {response.text}")
            return result
        except Exception as e:
            result = {
                "test_name": f"{search_type}_search",
                "success": False,
                "error": str(e),
                "query": query
            }
            print(f"❌ {search_type}搜索异常: {e}")
            return result
    
    async def test_streaming_search(self, search_type: str, query: str, **kwargs) -> Dict[str, Any]:
        """测试流式搜索功能"""
        print(f"🔍 测试{search_type}流式搜索: {query}")
        try:
            if search_type == "basic":
                payload = {
                    "query": query,
                    "k": kwargs.get("k", 5),
                    "temperature": kwargs.get("temperature", 0.6),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", 4096)
                }
            else:
                payload = {
                    "query": query,
                    "community_level": kwargs.get("community_level", 0),
                    "temperature": kwargs.get("temperature", 0.6),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", 4096)
                }
            
            full_response = ""
            chunk_count = 0
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/search/{search_type}/stream",
                json=payload,
                timeout=120.0
            ) as response:
                
                if response.status_code != 200:
                    return {
                        "test_name": f"{search_type}_streaming_search",
                        "success": False,
                        "error": f"HTTP {response.status_code}: {await response.aread()}",
                        "query": query
                    }
                
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        chunk_count += 1
                        try:
                            # 处理Server-Sent Events格式
                            if chunk.startswith("data: "):
                                data_str = chunk[6:].strip()
                                if data_str:
                                    data = json.loads(data_str)
                                    if data.get("type") == "chunk":
                                        content = data.get("content", "")
                                        full_response += content
                                        print(f"📡 接收流式数据块 {chunk_count}: {content[:50]}...")
                                    elif data.get("type") == "done":
                                        print("✅ 流式搜索完成")
                                        break
                                    elif data.get("type") == "error":
                                        return {
                                            "test_name": f"{search_type}_streaming_search",
                                            "success": False,
                                            "error": data.get("content", "未知错误"),
                                            "query": query
                                        }
                        except json.JSONDecodeError:
                            continue  # 忽略非JSON数据
            
            result = {
                "test_name": f"{search_type}_streaming_search",
                "success": True,
                "data": {
                    "full_response": full_response,
                    "chunk_count": chunk_count
                },
                "query": query
            }
            
            print(f"✅ {search_type}流式搜索成功:")
            print(f"   查询: {query}")
            print(f"   接收了 {chunk_count} 个数据块")
            print(f"   完整回答: {full_response[:200]}..." if len(full_response) > 200 else f"   完整回答: {full_response}")
            
            return result
            
        except Exception as e:
            result = {
                "test_name": f"{search_type}_streaming_search",
                "success": False,
                "error": str(e),
                "query": query
            }
            print(f"❌ {search_type}流式搜索异常: {e}")
            return result
    
    async def test_context(self) -> Dict[str, Any]:
        """测试获取上下文数据"""
        print("🔍 测试获取上下文数据...")
        try:
            response = await self.client.get(f"{self.base_url}/api/context")
            result = {
                "test_name": "context",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": None
            }
            if result["success"]:
                context = result["data"]["data"]["context"]
                print(f"✅ 上下文数据获取成功: {len(context)} 个键")
            else:
                print(f"❌ 上下文数据获取失败: {response.text}")
            return result
        except Exception as e:
            result = {
                "test_name": "context",
                "success": False,
                "error": str(e)
            }
            print(f"❌ 上下文数据获取异常: {e}")
            return result

async def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 开始 GraphRAG Web API 综合测试")
    print("=" * 60)
    
    results = []
    
    async with GraphRAGAPITester() as tester:
        
        # 1. 基础功能测试
        print("\n📋 第一阶段：基础功能测试")
        print("-" * 40)
        
        # 健康检查
        results.append(await tester.test_health())
        
        # 服务状态
        results.append(await tester.test_service_status())
        
        # 初始化服务
        results.append(await tester.test_initialize())
        
        # 索引状态
        results.append(await tester.test_index_status())
        
        # 尝试加载索引
        results.append(await tester.test_load_index())
        
        # 2. 搜索功能测试（如果索引可用）
        print("\n📋 第二阶段：搜索功能测试")
        print("-" * 40)
        
        # 检查是否有可用的索引
        status_result = await tester.test_service_status()
        if status_result["success"] and status_result["data"]["data"]["index_status"] == "ready":
            
            test_query = "图形算法的应用"
            
            # 测试各种搜索类型
            for search_type in ["global", "local", "drift", "basic"]:
                results.append(await tester.test_search(search_type, test_query))
            
            # 3. 流式搜索测试
            print("\n📋 第三阶段：流式搜索测试")
            print("-" * 40)
            
            # 测试流式搜索
            for search_type in ["global", "local", "drift", "basic"]:
                results.append(await tester.test_streaming_search(search_type, test_query))
            
            # 获取上下文数据
            results.append(await tester.test_context())
            
        else:
            print("⚠️ 索引未就绪，跳过搜索功能测试")
            print("💡 提示：运行索引构建测试或手动构建索引后再测试搜索功能")
            
            # 提供索引构建测试选项
            print("\n📋 可选：索引构建测试")
            print("-" * 40)
            print("⚠️ 注意：索引构建需要在input目录中有文档文件，且可能需要较长时间")
            
            # 检查input目录
            input_dir = "input"
            if os.path.exists(input_dir) and os.listdir(input_dir):
                print(f"✅ 发现input目录中有文件: {os.listdir(input_dir)}")
                print("🔄 可以尝试运行索引构建...")
                # 如果需要自动构建，取消下面的注释
                # results.append(await tester.test_build_index())
            else:
                print(f"❌ input目录为空或不存在，无法构建索引")
    
    # 4. 结果汇总
    print("\n📊 测试结果汇总")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {failed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n📋 详细结果:")
    for result in results:
        status = "✅ 通过" if result["success"] else "❌ 失败"
        print(f"  {status} - {result['test_name']}")
        if not result["success"] and "error" in result:
            print(f"    错误: {result['error']}")
    
    # 保存测试结果
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 测试结果已保存到 test_results.json")
    
    return results

async def run_quick_test():
    """运行快速测试"""
    print("🚀 开始 GraphRAG Web API 快速测试")
    print("=" * 60)
    
    async with GraphRAGAPITester() as tester:
        
        # 基础功能测试
        print("🔍 测试基础功能...")
        health_result = await tester.test_health()
        status_result = await tester.test_service_status()
        
        if health_result["success"] and status_result["success"]:
            print("✅ Web服务运行正常")
            
            # 如果索引就绪，测试搜索
            if status_result["data"]["data"]["index_status"] == "ready":
                print("✅ 索引已就绪，测试搜索功能...")
                search_result = await tester.test_search("global", "测试查询")
                if search_result["success"]:
                    print("✅ 搜索功能正常")
                else:
                    print("❌ 搜索功能异常")
            else:
                print("⚠️ 索引未就绪，需要先构建索引")
        else:
            print("❌ Web服务异常")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 快速测试
        asyncio.run(run_quick_test())
    else:
        # 综合测试
        asyncio.run(run_comprehensive_test())