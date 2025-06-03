#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG API测试脚本
测试所有API接口功能
"""

import asyncio
import json
import time
import os
from typing import Dict, Any

import httpx
import pytest

# API基础URL
BASE_URL = "http://localhost:8000"

class GraphRAGAPITester:
    """GraphRAG API测试类"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5分钟超时
        self.project_dir = "."  # 使用当前目录作为项目目录
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def test_health_check(self):
        """测试健康检查"""
        print("\n=== 测试健康检查 ===")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            print(f"状态码: {response.status_code}")
            print(f"响应: {response.json()}")
            assert response.status_code == 200
            print("✅ 健康检查通过")
            return True
        except Exception as e:
            print(f"❌ 健康检查失败: {str(e)}")
            return False
    
    async def test_root_endpoint(self):
        """测试根路径"""
        print("\n=== 测试根路径 ===")
        try:
            response = await self.client.get(f"{self.base_url}/")
            print(f"状态码: {response.status_code}")
            print(f"响应: {response.json()}")
            assert response.status_code == 200
            print("✅ 根路径测试通过")
            return True
        except Exception as e:
            print(f"❌ 根路径测试失败: {str(e)}")
            return False
    
    async def test_service_initialization(self):
        """测试服务初始化"""
        print("\n=== 测试服务初始化 ===")
        try:
            # 检查初始状态
            response = await self.client.get(f"{self.base_url}/api/status")
            print(f"初始状态: {response.json()}")
            
            # 初始化服务
            init_data = {
                "project_dir": self.project_dir
            }
            response = await self.client.post(
                f"{self.base_url}/api/init",
                json=init_data
            )
            print(f"初始化状态码: {response.status_code}")
            print(f"初始化响应: {response.json()}")
            
            if response.status_code == 200:
                print("✅ 服务初始化成功")
                return True
            else:
                print(f"❌ 服务初始化失败: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ 服务初始化异常: {str(e)}")
            return False
    
    async def test_service_status(self):
        """测试服务状态查询"""
        print("\n=== 测试服务状态查询 ===")
        try:
            response = await self.client.get(f"{self.base_url}/api/status")
            print(f"状态码: {response.status_code}")
            print(f"服务状态: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            assert response.status_code == 200
            print("✅ 服务状态查询成功")
            return True
        except Exception as e:
            print(f"❌ 服务状态查询失败: {str(e)}")
            return False
    
    async def test_index_status(self):
        """测试索引状态查询"""
        print("\n=== 测试索引状态查询 ===")
        try:
            response = await self.client.get(f"{self.base_url}/api/index/status")
            print(f"状态码: {response.status_code}")
            print(f"索引状态: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            assert response.status_code == 200
            print("✅ 索引状态查询成功")
            return True
        except Exception as e:
            print(f"❌ 索引状态查询失败: {str(e)}")
            return False
    
    async def test_index_building(self):
        """测试索引构建"""
        print("\n=== 测试索引构建 ===")
        
        # 检查输入目录是否有文件
        input_dir = os.path.join(self.project_dir, "input")
        if not os.path.exists(input_dir):
            print("⚠️  输入目录不存在，创建示例输入文件进行测试")
            os.makedirs(input_dir, exist_ok=True)
            # 创建一个示例文本文件
            sample_text = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习。深度学习是机器学习的一个子集，它基于人工神经网络的表示学习。

自然语言处理（NLP）是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言。

计算机视觉是一门研究如何使机器"看"的科学，更进一步的说，就是是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图形处理。
            """.strip()
            
            with open(os.path.join(input_dir, "ai_introduction.txt"), "w", encoding="utf-8") as f:
                f.write(sample_text)
            print("✅ 已创建示例输入文件")
        
        elif len(os.listdir(input_dir)) == 0:
            print("⚠️  输入目录为空，创建示例输入文件进行测试")
            # 创建一个示例文本文件
            sample_text = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习。深度学习是机器学习的一个子集，它基于人工神经网络的表示学习。

自然语言处理（NLP）是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言。

计算机视觉是一门研究如何使机器"看"的科学，更进一步的说，就是是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图形处理。
            """.strip()
            
            with open(os.path.join(input_dir, "ai_introduction.txt"), "w", encoding="utf-8") as f:
                f.write(sample_text)
            print("✅ 已创建示例输入文件")
        
        try:
            # 构建索引
            build_data = {
                "method": "standard",
                "is_update": False
            }
            print("开始构建索引...")
            print("⚠️  注意：索引构建可能需要几分钟时间，请耐心等待...")
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/api/index/build",
                json=build_data
            )
            end_time = time.time()
            
            print(f"构建耗时: {end_time - start_time:.2f} 秒")
            print(f"构建状态码: {response.status_code}")
            print(f"构建响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200:
                print("✅ 索引构建成功")
                return True
            else:
                print(f"❌ 索引构建失败: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ 索引构建异常: {str(e)}")
            return False
    
    async def test_index_loading(self):
        """测试索引加载"""
        print("\n=== 测试索引加载 ===")
        try:
            response = await self.client.post(f"{self.base_url}/api/index/load")
            print(f"状态码: {response.status_code}")
            print(f"加载响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200:
                print("✅ 索引加载成功")
                return True
            else:
                print(f"❌ 索引加载失败: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ 索引加载异常: {str(e)}")
            return False
    
    async def test_global_search(self):
        """测试全局搜索"""
        print("\n=== 测试全局搜索 ===")
        try:
            search_data = {
                "query": "什么是人工智能？",
                "community_level": 0,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1000
            }
            response = await self.client.post(
                f"{self.base_url}/api/search/global",
                json=search_data
            )
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"搜索结果: {result['data']['result'][:200]}...")
                print("✅ 全局搜索成功")
                return True
            else:
                print(f"❌ 全局搜索失败: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ 全局搜索异常: {str(e)}")
            return False
    
    async def test_local_search(self):
        """测试本地搜索"""
        print("\n=== 测试本地搜索 ===")
        try:
            search_data = {
                "query": "张明个人情况",
                "community_level": 1,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 4096
            }
            response = await self.client.post(
                f"{self.base_url}/api/search/local",
                json=search_data
            )
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"搜索结果: {result['data']['result'][:200]}...")
                print("✅ 本地搜索成功")
                return True
            else:
                print(f"❌ 本地搜索失败: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ 本地搜索异常: {str(e)}")
            return False
    
    async def test_drift_search(self):
        """测试DRIFT搜索"""
        print("\n=== 测试DRIFT搜索 ===")
        try:
            search_data = {
                "query": "深度学习技术",
                "community_level": 0,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1000
            }
            response = await self.client.post(
                f"{self.base_url}/api/search/drift",
                json=search_data
            )
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"搜索结果: {result['data']['result'][:200]}...")
                print("✅ DRIFT搜索成功")
                return True
            else:
                print(f"❌ DRIFT搜索失败: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ DRIFT搜索异常: {str(e)}")
            return False
    
    async def test_basic_search(self):
        """测试基本搜索"""
        print("\n=== 测试基本搜索 ===")
        try:
            search_data = {
                "query": "数据科学",
                "k": 5,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1000
            }
            response = await self.client.post(
                f"{self.base_url}/api/search/basic",
                json=search_data
            )
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"搜索结果: {result['data']['result'][:200]}...")
                print("✅ 基本搜索成功")
                return True
            else:
                print(f"❌ 基本搜索失败: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ 基本搜索异常: {str(e)}")
            return False
    
    async def test_streaming_search(self):
        """测试流式搜索"""
        print("\n=== 测试流式搜索 ===")
        try:
            search_data = {
                "query": "人工智能的未来发展",
                "community_level": 0,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 500
            }
            
            print("测试流式全局搜索...")
            start_time = time.time()
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/search/global/stream",
                json=search_data
            ) as response:
                print(f"状态码: {response.status_code}")
                if response.status_code == 200:
                    chunks = []
                    chunk_count = 0
                    first_chunk_time = None
                    
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            chunk_count += 1
                            if first_chunk_time is None:
                                first_chunk_time = time.time()
                                print(f"首个数据块到达时间: {first_chunk_time - start_time:.2f} 秒")
                            
                            chunks.append(chunk)
                            # 只显示前几个块的内容，避免输出过多
                            if chunk_count <= 3:
                                print(f"数据块 {chunk_count}: {chunk[:100]}...")
                            elif chunk_count == 4:
                                print("...")
                            
                            # 检查是否是结束标记
                            if "[DONE]" in chunk:
                                print("收到结束标记")
                                break
                    
                    end_time = time.time()
                    print(f"✅ 流式全局搜索成功")
                    print(f"总耗时: {end_time - start_time:.2f} 秒")
                    print(f"收到数据块数量: {len(chunks)}")
                    print(f"首块延迟: {first_chunk_time - start_time:.2f} 秒" if first_chunk_time else "未收到数据块")
                    
                    # 测试其他流式搜索接口
                    print("\n测试流式本地搜索...")
                    async with self.client.stream(
                        "POST",
                        f"{self.base_url}/api/search/local/stream",
                        json=search_data
                    ) as local_response:
                        if local_response.status_code == 200:
                            local_chunks = 0
                            async for chunk in local_response.aiter_text():
                                if chunk.strip():
                                    local_chunks += 1
                                    if "[DONE]" in chunk:
                                        break
                            print(f"✅ 流式本地搜索成功，收到 {local_chunks} 个数据块")
                        else:
                            print(f"❌ 流式本地搜索失败: {local_response.status_code}")
                    
                    return True
                else:
                    print(f"❌ 流式搜索失败: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ 流式搜索异常: {str(e)}")
            return False
    
    async def test_context_retrieval(self):
        """测试上下文获取"""
        print("\n=== 测试上下文获取 ===")
        try:
            response = await self.client.get(f"{self.base_url}/api/context")
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"上下文数据: {json.dumps(result, indent=2, ensure_ascii=False)[:300]}...")
                print("✅ 上下文获取成功")
                return True
            else:
                print(f"❌ 上下文获取失败: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ 上下文获取异常: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始GraphRAG API完整测试")
        print("=" * 60)
        
        test_results = []
        
        # 基础测试
        test_results.append(await self.test_health_check())
        test_results.append(await self.test_root_endpoint())
        
        # 服务管理测试
        test_results.append(await self.test_service_initialization())
        test_results.append(await self.test_service_status())
        
        # 索引管理测试
        test_results.append(await self.test_index_status())
        # test_results.append(await self.test_index_building())
        test_results.append(await self.test_index_loading())

        # test_results.append(await self.test_local_search())
        
        # # 搜索功能测试（只有在索引构建成功后才执行）
        # if test_results[-1]:  # 如果索引加载成功
        #     test_results.append(await self.test_global_search())
        #     test_results.append(await self.test_local_search())
        #     test_results.append(await self.test_drift_search())
        #     test_results.append(await self.test_basic_search())
        #     test_results.append(await self.test_streaming_search())
        #     test_results.append(await self.test_context_retrieval())
        # else:
        #     print("\n⚠️  索引未就绪，跳过搜索功能测试")
        
        # 测试结果汇总
        print("\n" + "=" * 60)
        print("📊 测试结果汇总")
        print("=" * 60)
        
        passed = sum(test_results)
        total = len(test_results)
        
        # 测试项目名称
        test_names = [
            "健康检查",
            "根路径",
            "服务初始化", 
            "服务状态查询",
            "索引状态查询",
            "索引构建",
            "索引加载"
        ]
        
        # 如果索引加载成功，添加搜索测试项目
        if len(test_results) > 7:
            test_names.extend([
                "全局搜索",
                "本地搜索", 
                "DRIFT搜索",
                "基本搜索",
                "流式搜索",
                "上下文获取"
            ])
        
        # 显示详细结果
        print("详细测试结果:")
        for i, (name, result) in enumerate(zip(test_names, test_results)):
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {i+1:2d}. {name:<12} - {status}")
        
        print(f"\n统计信息:")
        print(f"  总测试数: {total}")
        print(f"  通过数: {passed}")
        print(f"  失败数: {total - passed}")
        print(f"  通过率: {passed/total*100:.1f}%")
        
        if passed == total:
            print("\n🎉 所有测试通过！GraphRAG API服务运行正常")
        else:
            print(f"\n⚠️  有 {total - passed} 个测试失败，请检查以下内容:")
            print("  1. API服务是否正常启动")
            print("  2. 环境变量是否正确配置")
            print("  3. 依赖包是否正确安装")
            print("  4. 网络连接是否正常")
        
        return passed == total

async def main():
    """主函数"""
    print("GraphRAG API测试工具")
    print("请确保API服务已启动 (python app.py)")
    print("服务地址: http://localhost:8000")
    
    # 等待用户确认
    input("\n按回车键开始测试...")
    
    async with GraphRAGAPITester() as tester:
        success = await tester.run_all_tests()
        return success

def test_api_endpoints():
    """pytest测试函数"""
    return asyncio.run(main())

if __name__ == "__main__":
    # 直接运行测试
    success = asyncio.run(main())
    exit(0 if success else 1) 