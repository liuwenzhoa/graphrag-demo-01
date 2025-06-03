#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG Web API 使用演示
展示如何使用各种API接口
"""

import asyncio
import json
import time
import httpx
from typing import Optional

class GraphRAGDemo:
    """GraphRAG API演示类"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    async def demo_basic_operations(self):
        """演示基础操作"""
        print("🚀 GraphRAG Web API 演示")
        print("=" * 50)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            
            # 1. 健康检查
            print("\n1. 🔍 健康检查")
            try:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ 服务状态: {data.get('status')}")
                    print(f"   服务状态: {data.get('service_status')}")
                    print(f"   索引状态: {data.get('index_status')}")
                else:
                    print(f"❌ 健康检查失败: {response.status_code}")
                    return
            except Exception as e:
                print(f"❌ 无法连接到服务: {e}")
                print("💡 请确保服务已启动: python start_web_service.py")
                return
            
            # 2. 获取服务状态
            print("\n2. 📊 获取服务状态")
            try:
                response = await client.get(f"{self.base_url}/api/status")
                if response.status_code == 200:
                    data = response.json()["data"]
                    print(f"✅ 服务状态: {data.get('service_status')}")
                    print(f"   索引状态: {data.get('index_status')}")
                    print(f"   项目目录: {data.get('project_dir')}")
                    print(f"   有索引数据: {data.get('has_index_data')}")
                    
                    # 检查是否可以进行搜索
                    if data.get('index_status') == 'ready':
                        await self.demo_search_operations(client)
                    else:
                        print("\n⚠️ 索引未就绪，跳过搜索演示")
                        print("💡 请先构建索引或加载现有索引")
                        await self.demo_index_operations(client)
                else:
                    print(f"❌ 获取状态失败: {response.status_code}")
            except Exception as e:
                print(f"❌ 获取状态异常: {e}")
    
    async def demo_index_operations(self, client: httpx.AsyncClient):
        """演示索引操作"""
        print("\n3. 🔧 索引操作演示")
        
        # 尝试加载现有索引
        print("   📥 尝试加载现有索引...")
        try:
            response = await client.post(f"{self.base_url}/api/index/load")
            if response.status_code == 200:
                print("✅ 索引加载成功")
                return True
            else:
                print("⚠️ 索引加载失败（可能尚未构建）")
        except Exception as e:
            print(f"❌ 索引加载异常: {e}")
        
        # 检查输入文件
        print("   📁 检查输入文件...")
        import os
        if os.path.exists("input") and os.listdir("input"):
            files = os.listdir("input")
            print(f"✅ 发现 {len(files)} 个输入文件")
            
            print("   🔄 可以尝试构建索引...")
            print("   💡 提示：索引构建需要时间，这里仅作演示不实际执行")
            # 实际构建需要较长时间，这里仅作说明
            # response = await client.post(f"{self.base_url}/api/index/build", 
            #                            json={"method": "standard"})
        else:
            print("❌ input目录为空或不存在")
            print("💡 请将文档文件放入input目录后重新尝试")
        
        return False
    
    async def demo_search_operations(self, client: httpx.AsyncClient):
        """演示搜索操作"""
        print("\n3. 🔍 搜索功能演示")
        
        # 示例查询
        test_queries = [
            "图形算法有哪些应用？",
            "什么是机器学习？",
            "人工智能的发展历程"
        ]
        
        # 选择一个查询进行演示
        query = test_queries[0]
        print(f"   查询: {query}")
        
        # 演示不同类型的搜索
        search_types = [
            ("global", "全局搜索"),
            ("local", "本地搜索"), 
            ("drift", "DRIFT搜索"),
            ("basic", "基本搜索")
        ]
        
        for search_type, description in search_types:
            print(f"\n   🔍 {description} ({search_type})")
            try:
                # 准备请求数据
                if search_type == "basic":
                    payload = {
                        "query": query,
                        "k": 3,
                        "temperature": 0.7
                    }
                else:
                    payload = {
                        "query": query,
                        "community_level": 0,
                        "temperature": 0.7
                    }
                
                # 发送搜索请求
                response = await client.post(
                    f"{self.base_url}/api/search/{search_type}",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["data"]["result"]
                    print(f"✅ 搜索成功")
                    print(f"   回答: {answer[:200]}..." if len(answer) > 200 else f"   回答: {answer}")
                else:
                    print(f"❌ 搜索失败: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 搜索异常: {e}")
            
            # 添加延迟避免请求过快
            await asyncio.sleep(1)
    
    async def demo_streaming_search(self):
        """演示流式搜索"""
        print("\n4. 📡 流式搜索演示")
        
        query = "解释一下深度学习的基本概念"
        print(f"   查询: {query}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            
            print("   🔄 执行流式全局搜索...")
            
            try:
                payload = {
                    "query": query,
                    "temperature": 0.7
                }
                
                full_response = ""
                chunk_count = 0
                
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/search/global/stream",
                    json=payload
                ) as response:
                    
                    if response.status_code != 200:
                        print(f"❌ 流式搜索失败: {response.status_code}")
                        return
                    
                    print("   📡 接收流式数据:")
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            try:
                                if chunk.startswith("data: "):
                                    data_str = chunk[6:].strip()
                                    if data_str:
                                        data = json.loads(data_str)
                                        if data.get("type") == "chunk":
                                            content = data.get("content", "")
                                            full_response += content
                                            chunk_count += 1
                                            print(f"      块 {chunk_count}: {content[:80]}...")
                                        elif data.get("type") == "done":
                                            print("   ✅ 流式搜索完成")
                                            break
                                        elif data.get("type") == "error":
                                            print(f"   ❌ 流式搜索错误: {data.get('content')}")
                                            return
                            except json.JSONDecodeError:
                                continue
                
                print(f"\n   📊 流式搜索结果:")
                print(f"   总共接收 {chunk_count} 个数据块")
                print(f"   完整回答: {full_response[:300]}..." if len(full_response) > 300 else f"   完整回答: {full_response}")
                
            except Exception as e:
                print(f"❌ 流式搜索异常: {e}")
    
    async def demo_context_data(self):
        """演示上下文数据获取"""
        print("\n5. 📋 上下文数据演示")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{self.base_url}/api/context")
                if response.status_code == 200:
                    data = response.json()
                    context = data["data"]["context"]
                    print(f"✅ 上下文数据获取成功")
                    print(f"   上下文键数量: {len(context)}")
                    if context:
                        print(f"   包含的键: {list(context.keys())}")
                    else:
                        print("   暂无上下文数据（需要先执行搜索）")
                else:
                    print(f"❌ 上下文数据获取失败: {response.status_code}")
            except Exception as e:
                print(f"❌ 上下文数据获取异常: {e}")

async def main():
    """主演示函数"""
    demo = GraphRAGDemo()
    
    print("🎯 GraphRAG Web API 完整演示")
    print("本演示将展示以下功能:")
    print("1. 健康检查和服务状态")
    print("2. 索引操作（加载/构建）")
    print("3. 各种搜索功能")
    print("4. 流式搜索")
    print("5. 上下文数据获取")
    print("\n" + "=" * 60)
    
    # 基础操作演示
    await demo.demo_basic_operations()
    
    # 流式搜索演示
    await demo.demo_streaming_search()
    
    # 上下文数据演示
    await demo.demo_context_data()
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("\n💡 接下来你可以:")
    print("1. 访问 http://localhost:8000/docs 查看完整API文档")
    print("2. 运行 python test_web_api.py 进行完整测试")
    print("3. 将自己的文档放入input目录并构建索引")
    print("4. 使用各种搜索功能探索你的数据")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 演示已取消")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("💡 请确保GraphRAG Web服务已启动")