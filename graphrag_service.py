#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG 服务类
基于GraphRAGInteractive重构，用于Web服务
"""

import os
import asyncio
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv
import json
from typing import Any, Dict, Optional, Callable, Tuple
from enum import Enum

import pandas as pd
import tiktoken

# 导入GraphRAG API和其他必要组件
import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.enums import IndexingMethod, ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
from graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks

# 导入提示词常量
from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
from graphrag.prompts.index.extract_claims import EXTRACT_CLAIMS_PROMPT
from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT
from graphrag.prompts.index.community_report_text_units import COMMUNITY_REPORT_TEXT_PROMPT
from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_reduce_system_prompt import REDUCE_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_knowledge_system_prompt import GENERAL_KNOWLEDGE_INSTRUCTION
from graphrag.prompts.query.drift_search_system_prompt import DRIFT_LOCAL_SYSTEM_PROMPT, DRIFT_REDUCE_PROMPT
from graphrag.prompts.query.basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT
from graphrag.prompts.query.question_gen_system_prompt import QUESTION_SYSTEM_PROMPT

from graphrag.vector_stores.lancedb import LanceDBVectorStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """服务状态枚举"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    INDEXING = "indexing"
    ERROR = "error"

class IndexStatus(Enum):
    """索引状态枚举"""
    NOT_BUILT = "not_built"
    BUILDING = "building"
    READY = "ready"
    UPDATING = "updating"
    ERROR = "error"

class GraphRAGService:
    """GraphRAG服务类
    
    提供GraphRAG的所有核心功能，包括：
    - 配置管理
    - 索引构建和更新
    - 多种查询方法（全局、本地、DRIFT、基本）
    - 流式和非流式搜索
    """
    
    def __init__(self):
        """初始化GraphRAG服务"""
        self.service_status = ServiceStatus.NOT_INITIALIZED
        self.index_status = IndexStatus.NOT_BUILT
        self.error_message = None
        
        # 项目配置
        self.project_dir = None
        self.output_dir = None
        self.config = None
        self.model_manager = ModelManager()
        self.chat_model = None
        self.embedding_model = None
        self.token_encoder = None
        
        # 索引数据
        self.entity_df = None
        self.community_df = None
        self.report_df = None
        self.text_unit_df = None
        self.relationship_df = None
        self.covariate_df = None
        self.document_df = None
        
        # 向量存储
        self.description_embedding_store = None
        self.text_embedding_store = None
        self.community_content_embedding_store = None
        
        # 上下文数据
        self.context_data = {}
    
    async def initialize(self, project_dir: str, config_data: Optional[Dict] = None) -> Dict[str, Any]:
        """初始化服务
        
        Args:
            project_dir: 项目目录路径
            config_data: 可选的配置数据，如果不提供则使用默认配置
            
        Returns:
            初始化结果
        """
        try:
            self.service_status = ServiceStatus.INITIALIZING
            logger.info(f"开始初始化GraphRAG服务，项目目录: {project_dir}")
            
            # 设置项目目录
            await self._setup_project_directory(project_dir)
            
            # 设置配置
            if config_data:
                await self._setup_config_from_data(config_data)
            else:
                await self._setup_config()
            
            # 检查并生成提示词文件（只在需要时生成）
            await self._generate_prompts()
            
            # 检查索引状态
            self._check_index_status()
            
            # 如果索引已存在且有效，直接加载索引数据
            if self.index_status == IndexStatus.READY:
                logger.info("检测到现有索引文件，正在加载...")
                load_result = await self.load_index()
                if load_result["status"] != "success":
                    logger.warning("加载现有索引失败，索引状态重置为未构建")
                    self.index_status = IndexStatus.NOT_BUILT
                else:
                    logger.info("现有索引加载成功")
            
            self.service_status = ServiceStatus.READY
            logger.info("GraphRAG服务初始化完成")
            
            return {
                "status": "success",
                "service_status": self.service_status.value,
                "index_status": self.index_status.value,
                "project_dir": self.project_dir,
                "message": "服务初始化成功",
                "index_loaded": self.index_status == IndexStatus.READY
            }
            
        except Exception as e:
            self.service_status = ServiceStatus.ERROR
            self.error_message = str(e)
            logger.error(f"服务初始化失败: {str(e)}")
            return {
                "status": "error",
                "service_status": self.service_status.value,
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "service_status": self.service_status.value,
            "index_status": self.index_status.value,
            "project_dir": self.project_dir,
            "error_message": self.error_message,
            "has_index_data": self._has_index_data()
        }
    
    async def build_index(self, method: str = "standard", is_update: bool = False) -> Dict[str, Any]:
        """构建或更新索引
        
        Args:
            method: 索引方法，"standard" 或 "fast"
            is_update: 是否为增量更新
            
        Returns:
            构建结果
        """
        if self.service_status != ServiceStatus.READY:
            return {
                "status": "error",
                "error": "服务未就绪，请先初始化服务"
            }
        
        try:
            self.index_status = IndexStatus.UPDATING if is_update else IndexStatus.BUILDING
            
            indexing_method = IndexingMethod.Fast if method.lower() == "fast" else IndexingMethod.Standard
            operation_type = "增量更新" if is_update else "全新索引构建"
            
            logger.info(f"开始{operation_type}，使用方法: {method}")
            
            # 检查输入文件
            input_dir = os.path.join(self.project_dir, "input")
            if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
                self.index_status = IndexStatus.ERROR
                return {
                    "status": "error",
                    "error": "输入目录为空，无法执行索引构建"
                }
            
            # 如果是全新构建，清理现有文件
            if not is_update:
                await self._clean_output_directories()
            
            # 执行索引构建
            await self._build_index(indexing_method, is_update)
            
            # 加载索引数据
            load_result = await self.load_index()
            if load_result["status"] != "success":
                self.index_status = IndexStatus.ERROR
                return load_result
            
            self.index_status = IndexStatus.READY
            logger.info(f"{operation_type}完成")
            
            return {
                "status": "success",
                "index_status": self.index_status.value,
                "message": f"{operation_type}完成"
            }
            
        except Exception as e:
            self.index_status = IndexStatus.ERROR
            self.error_message = str(e)
            logger.error(f"索引构建失败: {str(e)}")
            return {
                "status": "error",
                "index_status": self.index_status.value,
                "error": str(e)
            }
    
    async def load_index(self) -> Dict[str, Any]:
        """加载索引数据"""
        try:
            logger.info("开始加载索引数据...")
            
            # 加载索引数据
            self._load_index_data()
            
            # 连接向量存储
            self._connect_vector_stores()
            
            self.index_status = IndexStatus.READY
            logger.info("索引数据加载完成")
            
            return {
                "status": "success",
                "index_status": self.index_status.value,
                "message": "索引数据加载成功"
            }
            
        except Exception as e:
            self.index_status = IndexStatus.ERROR
            self.error_message = str(e)
            logger.error(f"索引数据加载失败: {str(e)}")
            return {
                "status": "error",
                "index_status": self.index_status.value,
                "error": str(e)
            }
    
    async def global_search(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                           top_p: float = 0.9, max_tokens: int = 4096) -> Dict[str, Any]:
        """全局搜索"""
        if not self._check_search_ready():
            return {
                "status": "error",
                "error": "搜索功能未就绪，请先构建索引"
            }
        
        try:
            logger.info(f"执行全局搜索: {query}")
            
            # 保存原始配置的状态，以便在执行完搜索后恢复
            original_params = {}
            
            # 通过修改模型配置对象调整温度等参数（参考main.py实现）
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 记录原始参数值并设置新值
                for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                    if hasattr(model_config, param):
                        original_params[param] = getattr(model_config, param)
                        setattr(model_config, param, value)
            
            # 准备收集上下文数据
            context_data = {}
            
            def on_context(context):
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            user_query = query.strip()
            
            # 使用正确的API调用方式（参考main.py实现）
            response = await api.global_search(
                config=self.config,
                entities=self.entity_df,
                communities=self.community_df,
                community_reports=self.report_df,
                community_level=community_level,
                dynamic_community_selection=False,
                response_type="详细的中文回答",
                query=user_query,
                callbacks=callbacks,
            )
            
            # 处理global_search特有的context_data
            processed_context = {}
            if 'reports' in context_data:
                reports = context_data['reports']
                if isinstance(reports, pd.DataFrame):
                    # 将DataFrame转换为可序列化的格式
                    reports_list = []
                    for _, report in reports.iterrows():
                        report_dict = {
                            'title': report.get('title', '未命名社区'),
                            'summary': report.get('summary', report.get('content', '无摘要')),
                            'rank': report.get('rank', '未知'),
                            'id': report.get('id', ''),
                            'content': report.get('content', '')
                        }
                        reports_list.append(report_dict)
                    
                    processed_context['reports'] = {
                        'count': len(reports),
                        'data': reports_list
                    }
                    logger.info(f"全局搜索使用了 {len(reports)} 个社区报告生成答案")
                else:
                    processed_context['reports'] = str(reports)
            
            return {
                "status": "success",
                "result": response,
                "context": processed_context
            }
            
        except Exception as e:
            logger.error(f"全局搜索失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def local_search(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                          top_p: float = 0.9, max_tokens: int = 4096) -> Dict[str, Any]:
        """本地搜索"""
        if not self._check_search_ready():
            return {
                "status": "error",
                "error": "搜索功能未就绪，请先构建索引"
            }
        
        try:
            logger.info(f"执行本地搜索: {query}")
            
            # 保存原始配置的状态，以便在执行完搜索后恢复
            original_params = {}
            
            # 通过修改模型配置对象调整温度等参数（参考main.py实现）
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 记录原始参数值并设置新值
                for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                    if hasattr(model_config, param):
                        original_params[param] = getattr(model_config, param)
                        setattr(model_config, param, value)
            
            # 准备收集上下文数据
            context_data = {}
            
            def on_context(context):
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            user_query = query.strip()
            
            # 使用正确的API调用方式（参考main.py实现）
            response = await api.local_search(
                config=self.config,
                entities=self.entity_df,
                communities=self.community_df,
                community_reports=self.report_df,
                text_units=self.text_unit_df,
                relationships=self.relationship_df,
                covariates=self.covariate_df,
                community_level=community_level,
                response_type="详细的中文回答",
                query=user_query,
                callbacks=callbacks,
            )
            
            # 处理local_search特有的context_data
            # 根据实际测试，local_search返回的context_data中包含DataFrame对象，需要转换为可序列化格式
            processed_context = {}
            
            # 处理local_search的多个数据键
            for key in ['entities', 'relationships', 'reports', 'sources', 'claims']:
                if key in context_data:
                    data = context_data[key]
                    
                    if isinstance(data, pd.DataFrame):
                        # DataFrame对象转换为可序列化的格式
                        if len(data) > 0:
                            # 转换为字典列表格式
                            data_list = []
                            for _, row in data.iterrows():
                                row_dict = row.to_dict()
                                # 确保所有值都是可序列化的
                                for k, v in row_dict.items():
                                    if pd.isna(v):
                                        row_dict[k] = None
                                    elif isinstance(v, (pd.Timestamp, pd.NaT.__class__)):
                                        row_dict[k] = str(v)
                                    elif hasattr(v, 'item'):  # numpy类型
                                        row_dict[k] = v.item()
                                data_list.append(row_dict)
                            
                            processed_context[key] = {
                                'type': 'DataFrame',
                                'count': len(data),
                                'columns': list(data.columns),
                                'data': data_list[:10] if len(data_list) > 10 else data_list  # 限制返回数量
                            }
                        else:
                            # 空DataFrame
                            processed_context[key] = {
                                'type': 'DataFrame',
                                'count': 0,
                                'columns': list(data.columns) if hasattr(data, 'columns') else [],
                                'data': []
                            }
                    elif isinstance(data, str):
                        processed_context[key] = {
                            'type': 'string',
                            'content': data[:1000] if len(data) > 1000 else data  # 限制长度避免过大
                        }
                    elif isinstance(data, list):
                        processed_context[key] = {
                            'type': 'list',
                            'count': len(data),
                            'data': data[:10] if len(data) > 10 else data  # 限制返回数量
                        }
                    else:
                        processed_context[key] = {
                            'type': str(type(data)),
                            'content': str(data)[:1000] if len(str(data)) > 1000 else str(data)
                        }
            
            logger.info(f"本地搜索处理了 {len(processed_context)} 种类型的上下文数据")
            logger.info(f"本地搜索的context_data: {processed_context}")

            result_data = response
            # 处理本地搜索返回结果
            if isinstance(response, tuple) and len(response) > 0:
                # 如果是元组，第一个元素是Markdown文本
                result_data = result_data[0]
            else:
                # 如果不是元组，则整个结果就是Markdown文本
                result_data = result_data

            print(json.dumps(result_data, indent=2, ensure_ascii=False, default=str))
            return {
                "status": "success",
                "result": result_data,
                "context": processed_context
            }
            
        except Exception as e:
            logger.error(f"本地搜索失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def drift_search(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                          top_p: float = 0.9, max_tokens: int = 4096) -> Dict[str, Any]:
        """DRIFT搜索"""
        if not self._check_search_ready():
            return {
                "status": "error",
                "error": "搜索功能未就绪，请先构建索引"
            }
        
        try:
            logger.info(f"执行DRIFT搜索: {query}")
            
            # 保存原始配置的状态，以便在执行完搜索后恢复
            original_params = {}
            
            # 通过修改模型配置对象调整温度等参数（参考main.py实现）
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 记录原始参数值并设置新值
                for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                    if hasattr(model_config, param):
                        original_params[param] = getattr(model_config, param)
                        setattr(model_config, param, value)
            
            # 准备收集上下文数据
            context_data = {}
            
            def on_context(context):
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            user_query = query.strip()
            
            # 使用正确的API调用方式（参考main.py实现）
            response = await api.drift_search(
                config=self.config,
                entities=self.entity_df,
                communities=self.community_df,
                community_reports=self.report_df,
                text_units=self.text_unit_df,
                relationships=self.relationship_df,
                community_level=community_level,
                response_type="详细的中文回答",
                query=user_query,
                callbacks=callbacks,
            )
            
            # 处理drift_search特有的context_data
            processed_context = {}
            
            # 处理drift_search的数据键，根据实际情况，可能包含DataFrame对象
            for key in ['reports', 'entities', 'sources']:
                if key in context_data:
                    data = context_data[key]
                    
                    if isinstance(data, pd.DataFrame):
                        # DataFrame对象转换为可序列化的格式
                        if len(data) > 0:
                            # 转换为字典列表格式
                            data_list = []
                            for _, row in data.iterrows():
                                row_dict = row.to_dict()
                                # 确保所有值都是可序列化的
                                for k, v in row_dict.items():
                                    if pd.isna(v):
                                        row_dict[k] = None
                                    elif isinstance(v, (pd.Timestamp, pd.NaT.__class__)):
                                        row_dict[k] = str(v)
                                    elif hasattr(v, 'item'):  # numpy类型
                                        row_dict[k] = v.item()
                                data_list.append(row_dict)
                            
                            processed_context[key] = {
                                'type': 'DataFrame',
                                'count': len(data),
                                'columns': list(data.columns),
                                'data': data_list[:10] if len(data_list) > 10 else data_list  # 限制返回数量
                            }
                        else:
                            # 空DataFrame
                            processed_context[key] = {
                                'type': 'DataFrame',
                                'count': 0,
                                'columns': list(data.columns) if hasattr(data, 'columns') else [],
                                'data': []
                            }
                    elif isinstance(data, str):
                        processed_context[key] = {
                            'type': 'string',
                            'content': data[:1000] if len(data) > 1000 else data
                        }
                    elif isinstance(data, list):
                        processed_context[key] = {
                            'type': 'list',
                            'count': len(data),
                            'data': data[:10] if len(data) > 10 else data  # 限制返回数量
                        }
                    else:
                        processed_context[key] = {
                            'type': str(type(data)),
                            'content': str(data)[:1000] if len(str(data)) > 1000 else str(data)
                        }
            
            logger.info(f"DRIFT搜索处理了 {len(processed_context)} 种类型的上下文数据")
            
            return {
                "status": "success",
                "result": response,
                "context": processed_context
            }
            
        except Exception as e:
            logger.error(f"DRIFT搜索失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def basic_search(self, query: str, k: int = 5, temperature: float = 0.6, 
                          top_p: float = 0.9, max_tokens: int = 4096) -> Dict[str, Any]:
        """基本搜索"""
        if not self._check_search_ready():
            return {
                "status": "error",
                "error": "搜索功能未就绪，请先构建索引"
            }
        
        try:
            logger.info(f"执行基本搜索: {query}")
            
            # 保存原始配置的状态，以便在执行完搜索后恢复
            original_params = {}
            
            # 通过修改模型配置对象调整温度等参数（参考main.py实现）
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 记录原始参数值并设置新值
                for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                    if hasattr(model_config, param):
                        original_params[param] = getattr(model_config, param)
                        setattr(model_config, param, value)
            
            # 准备收集上下文数据
            context_data = {}
            
            def on_context(context):
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            user_query = query.strip()
            
            # 使用正确的API调用方式（参考main.py实现）
            response = await api.basic_search(
                config=self.config,
                entities=self.entity_df,
                text_units=self.text_unit_df,
                response_type="详细的中文回答",
                query=user_query,
                k=k,
                callbacks=callbacks,
            )
            
            # 处理basic_search特有的context_data
            processed_context = {}
            
            # 处理Sources数据（注意大写S）
            if 'Sources' in context_data:
                sources_data = context_data['Sources']
                if isinstance(sources_data, pd.DataFrame):
                    # 将DataFrame转换为可序列化的格式
                    sources_list = []
                    for _, source in sources_data.iterrows():
                        source_dict = {
                            'source_id': source.get('source_id', '未知'),
                            'text': source.get('text', '无内容'),
                            'id': source.get('id', ''),
                            'chunk_id': source.get('chunk_id', '')
                        }
                        sources_list.append(source_dict)
                    
                    processed_context['Sources'] = {
                        'count': len(sources_data),
                        'data': sources_list
                    }
                    logger.info(f"基本搜索检索到 {len(sources_data)} 个最相关的文本片段")
                else:
                    processed_context['Sources'] = str(sources_data)
            
            return {
                "status": "success",
                "result": response,
                "context": processed_context
            }
            
        except Exception as e:
            logger.error(f"基本搜索失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def global_search_streaming(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                                     top_p: float = 0.9, max_tokens: int = 4096, 
                                     callback: Optional[Callable] = None) -> Tuple[str, Dict]:
        """流式全局搜索"""
        if not self._check_search_ready():
            raise Exception("搜索功能未就绪，请先构建索引")
        
        try:
            logger.info(f"执行流式全局搜索: {query}")
            
            # 保存原始配置的状态，以便在执行完搜索后恢复
            original_params = {}
            
            # 通过修改模型配置对象调整温度等参数（参考main.py实现）
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 记录原始参数值并设置新值
                for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                    if hasattr(model_config, param):
                        original_params[param] = getattr(model_config, param)
                        setattr(model_config, param, value)
            
            # 准备收集上下文数据
            context_data = {}
            
            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            user_query = query.strip()
            
            # 使用正确的API调用方式（参考main.py实现）
            result = await api.global_search(
                config=self.config,
                entities=self.entity_df,
                communities=self.community_df,
                community_reports=self.report_df,
                community_level=community_level,
                dynamic_community_selection=False,
                response_type="stream",
                query=user_query,
                callbacks=callbacks,
            )
            
            full_response = ""
            async for chunk in result.response:
                if callback:
                    callback(chunk)
                full_response += chunk
            
            # 处理global_search_streaming特有的context_data
            processed_context = {}
            if 'reports' in context_data:
                reports = context_data['reports']
                if isinstance(reports, pd.DataFrame):
                    # 将DataFrame转换为可序列化的格式
                    reports_list = []
                    for _, report in reports.iterrows():
                        report_dict = {
                            'title': report.get('title', '未命名社区'),
                            'summary': report.get('summary', report.get('content', '无摘要')),
                            'rank': report.get('rank', '未知'),
                            'id': report.get('id', ''),
                            'content': report.get('content', '')
                        }
                        reports_list.append(report_dict)
                    
                    processed_context['reports'] = {
                        'count': len(reports),
                        'data': reports_list
                    }
                    logger.info(f"流式全局搜索使用了 {len(reports)} 个社区报告生成答案")
                else:
                    processed_context['reports'] = str(reports)
            
            return full_response, processed_context
            
        except Exception as e:
            logger.error(f"流式全局搜索失败: {str(e)}")
            raise e
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def local_search_streaming(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                                    top_p: float = 0.9, max_tokens: int = 4096, 
                                    callback: Optional[Callable] = None) -> Tuple[str, Dict]:
        """流式本地搜索"""
        if not self._check_search_ready():
            raise Exception("搜索功能未就绪，请先构建索引")
        
        try:
            logger.info(f"执行流式本地搜索: {query}")
            
            # 保存原始配置的状态，以便在执行完搜索后恢复
            original_params = {}
            
            # 通过修改模型配置对象调整温度等参数（参考main.py实现）
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 记录原始参数值并设置新值
                for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                    if hasattr(model_config, param):
                        original_params[param] = getattr(model_config, param)
                        setattr(model_config, param, value)
            
            # 准备收集上下文数据
            context_data = {}
            
            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            user_query = query.strip()
            
            # 使用正确的API调用方式（参考main.py实现）
            result = await api.local_search(
                config=self.config,
                entities=self.entity_df,
                communities=self.community_df,
                community_reports=self.report_df,
                text_units=self.text_unit_df,
                relationships=self.relationship_df,
                covariates=self.covariate_df,
                community_level=community_level,
                response_type="stream",
                query=user_query,
                callbacks=callbacks,
            )
            
            full_response = ""
            async for chunk in result.response:
                if callback:
                    callback(chunk)
                full_response += chunk
            
            # 处理local_search_streaming特有的context_data
            processed_context = {}
            
            # 处理local_search的多个数据键
            for key in ['entities', 'relationships', 'reports', 'sources', 'claims']:
                if key in context_data:
                    data = context_data[key]
                    
                    if isinstance(data, pd.DataFrame):
                        # DataFrame对象转换为可序列化的格式
                        if len(data) > 0:
                            # 转换为字典列表格式
                            data_list = []
                            for _, row in data.iterrows():
                                row_dict = row.to_dict()
                                # 确保所有值都是可序列化的
                                for k, v in row_dict.items():
                                    if pd.isna(v):
                                        row_dict[k] = None
                                    elif isinstance(v, (pd.Timestamp, pd.NaT.__class__)):
                                        row_dict[k] = str(v)
                                    elif hasattr(v, 'item'):  # numpy类型
                                        row_dict[k] = v.item()
                                data_list.append(row_dict)
                            
                            processed_context[key] = {
                                'type': 'DataFrame',
                                'count': len(data),
                                'columns': list(data.columns),
                                'data': data_list[:10] if len(data_list) > 10 else data_list  # 限制返回数量
                            }
                        else:
                            # 空DataFrame
                            processed_context[key] = {
                                'type': 'DataFrame',
                                'count': 0,
                                'columns': list(data.columns) if hasattr(data, 'columns') else [],
                                'data': []
                            }
                    elif isinstance(data, str):
                        processed_context[key] = {
                            'type': 'string',
                            'content': data[:1000] if len(data) > 1000 else data  # 限制长度避免过大
                        }
                    elif isinstance(data, list):
                        processed_context[key] = {
                            'type': 'list',
                            'count': len(data),
                            'data': data[:10] if len(data) > 10 else data  # 限制返回数量
                        }
                    else:
                        processed_context[key] = {
                            'type': str(type(data)),
                            'content': str(data)[:1000] if len(str(data)) > 1000 else str(data)
                        }
            
            logger.info(f"流式本地搜索处理了 {len(processed_context)} 种类型的上下文数据")
            
            return full_response, processed_context
            
        except Exception as e:
            logger.error(f"流式本地搜索失败: {str(e)}")
            raise e
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def drift_search_streaming(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                                    top_p: float = 0.9, max_tokens: int = 4096, 
                                    callback: Optional[Callable] = None) -> Tuple[str, Dict]:
        """流式DRIFT搜索"""
        if not self._check_search_ready():
            raise Exception("搜索功能未就绪，请先构建索引")
        
        try:
            logger.info(f"执行流式DRIFT搜索: {query}")
            
            # 保存原始配置的状态，以便在执行完搜索后恢复
            original_params = {}
            
            # 通过修改模型配置对象调整温度等参数（参考main.py实现）
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 记录原始参数值并设置新值
                for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                    if hasattr(model_config, param):
                        original_params[param] = getattr(model_config, param)
                        setattr(model_config, param, value)
            
            # 准备收集上下文数据
            context_data = {}
            
            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            user_query = query.strip()
            
            # 使用正确的API调用方式（参考main.py实现）
            result = await api.drift_search(
                config=self.config,
                entities=self.entity_df,
                communities=self.community_df,
                community_reports=self.report_df,
                text_units=self.text_unit_df,
                relationships=self.relationship_df,
                community_level=community_level,
                response_type="stream",
                query=user_query,
                callbacks=callbacks,
            )
            
            full_response = ""
            async for chunk in result.response:
                if callback:
                    callback(chunk)
                full_response += chunk
            
            # 处理drift_search_streaming特有的context_data
            processed_context = {}
            
            # 处理drift_search的数据键，根据实际情况，可能包含DataFrame对象
            for key in ['reports', 'entities', 'sources']:
                if key in context_data:
                    data = context_data[key]
                    
                    if isinstance(data, pd.DataFrame):
                        # DataFrame对象转换为可序列化的格式
                        if len(data) > 0:
                            # 转换为字典列表格式
                            data_list = []
                            for _, row in data.iterrows():
                                row_dict = row.to_dict()
                                # 确保所有值都是可序列化的
                                for k, v in row_dict.items():
                                    if pd.isna(v):
                                        row_dict[k] = None
                                    elif isinstance(v, (pd.Timestamp, pd.NaT.__class__)):
                                        row_dict[k] = str(v)
                                    elif hasattr(v, 'item'):  # numpy类型
                                        row_dict[k] = v.item()
                                data_list.append(row_dict)
                            
                            processed_context[key] = {
                                'type': 'DataFrame',
                                'count': len(data),
                                'columns': list(data.columns),
                                'data': data_list[:10] if len(data_list) > 10 else data_list  # 限制返回数量
                            }
                        else:
                            # 空DataFrame
                            processed_context[key] = {
                                'type': 'DataFrame',
                                'count': 0,
                                'columns': list(data.columns) if hasattr(data, 'columns') else [],
                                'data': []
                            }
                    elif isinstance(data, str):
                        processed_context[key] = {
                            'type': 'string',
                            'content': data[:1000] if len(data) > 1000 else data
                        }
                    elif isinstance(data, list):
                        processed_context[key] = {
                            'type': 'list',
                            'count': len(data),
                            'data': data[:10] if len(data) > 10 else data  # 限制返回数量
                        }
                    else:
                        processed_context[key] = {
                            'type': str(type(data)),
                            'content': str(data)[:1000] if len(str(data)) > 1000 else str(data)
                        }
            
            logger.info(f"流式DRIFT搜索处理了 {len(processed_context)} 种类型的上下文数据")
            
            return full_response, processed_context
            
        except Exception as e:
            logger.error(f"流式DRIFT搜索失败: {str(e)}")
            raise e
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def basic_search_streaming(self, query: str, k: int = 5, temperature: float = 0.6, 
                                    top_p: float = 0.9, max_tokens: int = 4096,
                                    callback: Optional[Callable] = None) -> Tuple[str, Dict]:
        """流式基本搜索"""
        if not self._check_search_ready():
            raise Exception("搜索功能未就绪，请先构建索引")
        
        try:
            logger.info(f"执行流式基本搜索: {query}")
            
            # 保存原始配置的状态，以便在执行完搜索后恢复
            original_params = {}
            
            # 通过修改模型配置对象调整温度等参数（参考main.py实现）
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 记录原始参数值并设置新值
                for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                    if hasattr(model_config, param):
                        original_params[param] = getattr(model_config, param)
                        setattr(model_config, param, value)
            
            # 准备收集上下文数据
            context_data = {}
            
            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            user_query = query.strip()
            
            # 使用正确的API调用方式（参考main.py实现）
            result = await api.basic_search(
                config=self.config,
                entities=self.entity_df,
                text_units=self.text_unit_df,
                response_type="stream",
                query=user_query,
                k=k,
                callbacks=callbacks,
            )
            
            full_response = ""
            async for chunk in result.response:
                if callback:
                    callback(chunk)
                full_response += chunk
            
            # 处理basic_search_streaming特有的context_data
            processed_context = {}
            
            # 处理Sources数据（注意大写S）
            if 'Sources' in context_data:
                sources_data = context_data['Sources']
                if isinstance(sources_data, pd.DataFrame):
                    # 将DataFrame转换为可序列化的格式
                    sources_list = []
                    for _, source in sources_data.iterrows():
                        source_dict = {
                            'source_id': source.get('source_id', '未知'),
                            'text': source.get('text', '无内容'),
                            'id': source.get('id', ''),
                            'chunk_id': source.get('chunk_id', '')
                        }
                        sources_list.append(source_dict)
                    
                    processed_context['Sources'] = {
                        'count': len(sources_data),
                        'data': sources_list
                    }
                    logger.info(f"流式基本搜索检索到 {len(sources_data)} 个最相关的文本片段")
                else:
                    processed_context['Sources'] = str(sources_data)
            
            return full_response, processed_context
            
        except Exception as e:
            logger.error(f"流式基本搜索失败: {str(e)}")
            raise e
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    # ========== 私有辅助方法 ==========
    
    async def _setup_project_directory(self, project_dir: str):
        """设置项目目录"""
        self.project_dir = project_dir
        
        # 创建项目目录（如果不存在）
        os.makedirs(self.project_dir, exist_ok=True)
        
        # 创建必要的子目录
        self.output_dir = os.path.join(self.project_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        input_dir = os.path.join(self.project_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        cache_dir = os.path.join(self.project_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        prompts_dir = os.path.join(self.project_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        
        # 处理.env文件
        env_path = os.path.join(self.project_dir, ".env")
        if os.path.exists(env_path):
            try:
                load_dotenv(env_path)
                logger.info(f"已加载环境变量配置文件: {env_path}")
            except Exception as e:
                logger.warning(f"加载.env文件时出错: {str(e)}")
    
    async def _setup_config(self):
        """设置配置"""
        config_path = Path(self.project_dir) / "settings.yaml"
        
        if config_path.exists():
            try:
                self.config = load_config(Path(self.project_dir))
                logger.info(f"已从YAML文件加载配置: {config_path}")
            except Exception as e:
                logger.warning(f"加载配置时出错: {str(e)}")
                self._create_config()
        else:
            self._create_config()
    
    async def _setup_config_from_data(self, config_data: Dict):
        """从数据创建配置"""
        try:
            self.config = create_graphrag_config(config_data, Path(self.project_dir))
            logger.info("已从提供的数据创建配置")
        except Exception as e:
            logger.error(f"从数据创建配置失败: {str(e)}")
            raise e
    
    def _create_config(self):
        """创建默认配置"""
        logger.info("创建默认GraphRAG配置...")
        
        # 获取模型配置
        chat_model_config, embedding_model_config = self._create_model_configs()
        
        # 创建配置字典
        config_data = {
            "models": {
                "default_chat_model": chat_model_config,
                "default_embedding_model": embedding_model_config
            },
            "input": {
                "type": "file",
                "file_type": "text",
                "base_dir": os.path.join(self.project_dir, "input"),
            },
            "chunks": {
                "size": 1200,
                "overlap": 100,
                "strategy": "tokens",
            },
            "output": {
                "type": "file",
                "base_dir": os.path.join(self.project_dir, "output"),
            },
            "update_index_output": {
                "type": "file",
                "base_dir": os.path.join(self.project_dir, "output"),
            },
            "cache": {
                "type": "file",
                "base_dir": os.path.join(self.project_dir, "cache"),
            },
            "reporting": {
                "type": "file",
                "base_dir": os.path.join(self.project_dir, "logs"),
            },
            "vector_store": {
                "default_vector_store": {
                    "type": "lancedb",
                    "db_uri": os.path.join(self.project_dir, "output", "lancedb"),
                    "container_name": "default",
                    "overwrite": True,
                }
            },
            "workflows": [
                "create_base_text_units",
                "create_final_documents",
                "extract_graph",
                "finalize_graph",
                "create_communities",
                "create_final_text_units",
                "create_community_reports",
                "generate_text_embeddings",
            ],
            "embed_text": {
                "enabled": True,
                "model_id": "default_embedding_model",
                "vector_store_id": "default_vector_store",
                "batch_size": 10,
                "batch_max_tokens": 8000,
            },
            "extract_graph": {
                "model_id": "default_chat_model",
                "prompt": "prompts/extract_graph.txt",
                "entity_types": ["organization", "person", "geo", "event"],
                "max_gleanings": 2,
            },
            "summarize_descriptions": {
                "model_id": "default_chat_model",
                "prompt": "prompts/summarize_descriptions.txt",
                "max_length": 1000,
                "max_input_length": 12000,
            },
        }
        
        # 创建GraphRAG配置对象
        self.config = create_graphrag_config(config_data, Path(self.project_dir))
        
        # 更新提示词路径
        self._update_prompt_paths_in_config()
        
        logger.info("默认配置创建完成")
    
    def _create_model_configs(self):
        """创建模型配置（参考main.py实现）"""
        # 获取API密钥
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        deepseek_api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        aliyun_api_key = os.environ.get("ALIYUN_API_KEY")
        aliyun_api_base = os.environ.get("ALIYUN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 如果缺少必要的API密钥，使用默认值（在Web服务中不进行交互式输入）
        if not deepseek_api_key:
            logger.warning("未找到DEEPSEEK_API_KEY环境变量")
            deepseek_api_key = ""
        
        if not aliyun_api_key:
            logger.warning("未找到ALIYUN_API_KEY环境变量")
            aliyun_api_key = ""
        
        # 配置聊天模型（使用LanguageModelConfig对象，参考main.py实现）
        chat_config = LanguageModelConfig(
            type=ModelType.OpenAIChat,
            api_base=deepseek_api_base,
            auth_type="api_key",
            api_key=deepseek_api_key,
            model="qwen3-235b-a22b",  # 使用与main.py相同的模型
            encoding_model="cl100k_base",
            model_supports_json=True,
            async_mode="threaded",
            max_retries=2,
            parallelization_num_threads=50,
            parallelization_stagger=0.3,
        )
        
        # 配置嵌入模型（使用LanguageModelConfig对象，参考main.py实现）
        embedding_config = LanguageModelConfig(
            type=ModelType.OpenAIEmbedding,
            api_base=aliyun_api_base,
            auth_type="api_key",
            api_key=aliyun_api_key,
            model="text-embedding-v3",
            encoding_model="cl100k_base",
            model_supports_json=True,
            async_mode="threaded",
            max_retries=2,
            parallelization_num_threads=50,
            parallelization_stagger=0.3,
        )
        
        return chat_config, embedding_config
    
    async def _generate_prompts(self):
        """生成提示词文件（只在需要时生成）"""
        prompts_dir = os.path.join(self.project_dir, "prompts")
        
        # 提示词文件映射（使用与main.py相同的文件名）
        prompt_files = {
            "extract_graph.txt": GRAPH_EXTRACTION_PROMPT,
            "summarize_descriptions.txt": SUMMARIZE_PROMPT,
            "extract_claims.txt": EXTRACT_CLAIMS_PROMPT,
            "community_report_graph.txt": COMMUNITY_REPORT_PROMPT,
            "community_report_text.txt": COMMUNITY_REPORT_TEXT_PROMPT,
            "drift_search_system_prompt.txt": DRIFT_LOCAL_SYSTEM_PROMPT,
            "drift_reduce_prompt.txt": DRIFT_REDUCE_PROMPT,
            "global_search_map_system_prompt.txt": MAP_SYSTEM_PROMPT,
            "global_search_reduce_system_prompt.txt": REDUCE_SYSTEM_PROMPT,
            "global_search_knowledge_system_prompt.txt": GENERAL_KNOWLEDGE_INSTRUCTION,
            "local_search_system_prompt.txt": LOCAL_SEARCH_SYSTEM_PROMPT,
            "basic_search_system_prompt.txt": BASIC_SEARCH_SYSTEM_PROMPT,
            "question_gen_system_prompt.txt": QUESTION_SYSTEM_PROMPT,
        }
        
        # 检查提示词目录是否存在
        if not os.path.exists(prompts_dir):
            logger.info("提示词目录不存在，创建目录并生成所有提示词文件")
            os.makedirs(prompts_dir, exist_ok=True)
            need_generate_all = True
        else:
            # 检查哪些文件需要生成
            missing_files = []
            for filename in prompt_files.keys():
                file_path = os.path.join(prompts_dir, filename)
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    missing_files.append(filename)
            
            if missing_files:
                logger.info(f"检测到 {len(missing_files)} 个提示词文件缺失或为空，将重新生成: {missing_files}")
                need_generate_all = False
            else:
                logger.info("所有提示词文件已存在且有效，跳过生成步骤")
                return
        
        # 生成缺失的提示词文件
        generated_count = 0
        if need_generate_all:
            # 生成所有文件
            for filename, content in prompt_files.items():
                file_path = os.path.join(prompts_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                generated_count += 1
        else:
            # 只生成缺失的文件
            for filename in missing_files:
                if filename in prompt_files:
                    file_path = os.path.join(prompts_dir, filename)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(prompt_files[filename])
                    generated_count += 1
        
        logger.info(f"已生成 {generated_count} 个提示词文件")
    
    def _update_prompt_paths_in_config(self):
        """更新配置对象中的提示词路径（参考main.py实现）"""
        if not hasattr(self, "config") or self.config is None:
            logger.warning("配置对象未初始化，无法更新提示词路径")
            return
        
        # 更新索引相关提示词路径
        if hasattr(self.config, "extract_graph"):
            self.config.extract_graph.prompt = "prompts/extract_graph.txt"
        
        if hasattr(self.config, "summarize_descriptions"):
            self.config.summarize_descriptions.prompt = "prompts/summarize_descriptions.txt"
        
        if hasattr(self.config, "extract_claims"):
            self.config.extract_claims.prompt = "prompts/extract_claims.txt"
        
        if hasattr(self.config, "community_reports"):
            self.config.community_reports.graph_prompt = "prompts/community_report_graph.txt"
            self.config.community_reports.text_prompt = "prompts/community_report_text.txt"
        
        # 更新查询相关提示词路径
        if hasattr(self.config, "local_search"):
            self.config.local_search.prompt = "prompts/local_search_system_prompt.txt"
        
        if hasattr(self.config, "global_search"):
            self.config.global_search.map_prompt = "prompts/global_search_map_system_prompt.txt"
            self.config.global_search.reduce_prompt = "prompts/global_search_reduce_system_prompt.txt"
            self.config.global_search.knowledge_prompt = "prompts/global_search_knowledge_system_prompt.txt"
        
        if hasattr(self.config, "drift_search"):
            self.config.drift_search.prompt = "prompts/drift_search_system_prompt.txt"
            self.config.drift_search.reduce_prompt = "prompts/drift_reduce_prompt.txt"
    
    async def _build_index(self, method: IndexingMethod = IndexingMethod.Standard, is_update: bool = False):
        """构建索引"""
        try:
            # 如果是增量更新，需要添加更新工作流
            if is_update and hasattr(self.config, 'workflows'):
                update_workflows = [
                    "update_final_documents",
                    "update_entities_relationships", 
                    "update_text_units",
                    "update_covariates",
                    "update_communities",
                    "update_community_reports",
                    "update_text_embeddings",
                    "update_clean_state",
                ]
                
                # 检查当前配置的workflows中是否已有增量更新工作流
                existing_update_workflows = [w for w in self.config.workflows if w.startswith("update_")]
                
                # 只添加当前不存在的增量更新工作流
                if not existing_update_workflows:
                    logger.info("为增量更新模式自动追加update工作流到工作流配置中")
                    self.config.workflows.extend(update_workflows)
            
            # 创建进度回调对象（参考main.py实现）
            progress_logger = ConsoleWorkflowCallbacks()
            callbacks = [progress_logger]
            
            # 使用正确的API调用方式（参考main.py实现）
            results = await api.build_index(
                config=self.config,
                method=method,
                is_update_run=is_update,
                callbacks=callbacks,
                progress_logger=None
            )
            
            # 检查索引过程是否成功完成（参考main.py实现）
            if results:
                success = True
                errors = []
                
                logger.info("索引构建完成，工作流结果:")
                for result in results:
                    status = "error" if result.errors else "success"
                    logger.info(f"工作流名称: {result.workflow}\t状态: {status}")
                    if result.errors and len(result.errors) > 0:
                        success = False
                        errors.extend(result.errors)
                        logger.error(f"错误详情:")
                        for error in result.errors:
                            logger.error(f"  - {error}")
                
                if success:
                    logger.info("索引构建成功完成")
                else:
                    logger.error("索引构建过程中有错误:")
                    for error in errors:
                        logger.error(f"- {error}")
                    raise Exception(f"索引构建失败: {'; '.join(str(e) for e in errors)}")
            else:
                raise Exception("索引构建未返回结果，可能是由于配置问题或其他错误导致")
            
        except Exception as e:
            logger.error(f"索引构建失败: {str(e)}")
            raise e
    
    async def _clean_output_directories(self):
        """清理输出目录"""
        try:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info("已清理输出目录")
        except Exception as e:
            logger.warning(f"清理输出目录时出错: {str(e)}")
    
    def _load_index_data(self):
        """加载索引数据"""
        try:
            logger.info("开始加载索引数据...")
            
            # 加载各种数据表（使用正确的文件名）
            output_path = Path(self.output_dir)
            
            # 核心索引数据
            self.community_df = pd.read_parquet(output_path / "communities.parquet")
            self.entity_df = pd.read_parquet(output_path / "entities.parquet")
            self.report_df = pd.read_parquet(output_path / "community_reports.parquet")
            self.text_unit_df = pd.read_parquet(output_path / "text_units.parquet")
            self.relationship_df = pd.read_parquet(output_path / "relationships.parquet")
            self.document_df = pd.read_parquet(output_path / "documents.parquet")
            
            logger.info(f"已加载核心索引数据:")
            logger.info(f"- 社区数: {len(self.community_df)}")
            logger.info(f"- 实体数: {len(self.entity_df)}")
            logger.info(f"- 社区报告数: {len(self.report_df)}")
            logger.info(f"- 文本单元数: {len(self.text_unit_df)}")
            logger.info(f"- 关系数: {len(self.relationship_df)}")
            logger.info(f"- 文档数: {len(self.document_df)}")
            
            # 尝试读取协变量表(如果存在)
            # 注意：协变量表是可选的，在默认索引工作流中通常不会生成
            try:
                covariate_path = output_path / "covariates.parquet"
                if covariate_path.exists():
                    self.covariate_df = pd.read_parquet(covariate_path)
                    logger.info(f"- 协变量数: {len(self.covariate_df)}")
                else:
                    logger.info("- 协变量表不存在，跳过加载")
                    self.covariate_df = None
            except Exception as e:
                logger.warning(f"加载协变量表时出错: {str(e)}")
                self.covariate_df = None
            
        except Exception as e:
            logger.error(f"加载索引数据失败: {str(e)}")
            raise e
    
    def _connect_vector_stores(self):
        """连接向量存储"""
        try:
            if not self.config or not hasattr(self.config, 'vector_store'):
                logger.warning("配置中未找到向量存储配置")
                return
            
            # 获取向量存储配置
            vector_store_config = self.config.vector_store.get("default_vector_store")
            if not vector_store_config:
                logger.warning("未找到默认向量存储配置")
                return
            
            # 检查向量存储目录是否存在
            lance_db_path = os.path.join(self.output_dir, "lancedb")
            if not os.path.exists(lance_db_path):
                logger.warning(f"向量存储目录不存在: {lance_db_path}")
                return
            
            logger.info("开始连接向量存储...")
            
            # 容器名称
            container_name = vector_store_config.container_name or "default"
            
            # 连接实体描述向量存储
            try:
                self.description_embedding_store = LanceDBVectorStore(
                    collection_name=f"{container_name}-entity-description"
                )
                self.description_embedding_store.connect(db_uri=lance_db_path)
                logger.info("已连接实体描述向量存储")
            except Exception as e:
                logger.warning(f"连接实体描述向量存储失败: {str(e)}")
                self.description_embedding_store = None
            
            # 连接文本单元向量存储
            try:
                self.text_embedding_store = LanceDBVectorStore(
                    collection_name=f"{container_name}-text_unit-text"
                )
                self.text_embedding_store.connect(db_uri=lance_db_path)
                logger.info("已连接文本单元向量存储")
            except Exception as e:
                logger.warning(f"连接文本单元向量存储失败: {str(e)}")
                self.text_embedding_store = None
            
            # 连接社区内容向量存储
            try:
                self.community_content_embedding_store = LanceDBVectorStore(
                    collection_name=f"{container_name}-community-full_content"
                )
                self.community_content_embedding_store.connect(db_uri=lance_db_path)
                logger.info("已连接社区内容向量存储")
            except Exception as e:
                logger.warning(f"连接社区内容向量存储失败: {str(e)}")
                self.community_content_embedding_store = None
                
        except Exception as e:
            logger.error(f"连接向量存储失败: {str(e)}")
            # 确保在失败时重置所有向量存储
            self.description_embedding_store = None
            self.text_embedding_store = None
            self.community_content_embedding_store = None
            raise e
    
    def _check_search_ready(self) -> bool:
        """检查搜索功能是否就绪"""
        return (self.service_status == ServiceStatus.READY and 
                self.index_status == IndexStatus.READY and 
                self._has_index_data())
    
    def _has_index_data(self) -> bool:
        """检查是否有索引数据"""
        return (self.entity_df is not None and 
                self.community_df is not None and 
                self.report_df is not None)
    
    def _check_index_status(self):
        """检查索引状态"""
        output_path = Path(self.output_dir)
        
        # 检查关键文件是否存在且有效（使用正确的文件名）
        key_files = [
            "entities.parquet",
            "communities.parquet", 
            "community_reports.parquet",
            "text_units.parquet"
        ]
        
        # 检查输出目录是否存在
        if not output_path.exists():
            self.index_status = IndexStatus.NOT_BUILT
            logger.info("输出目录不存在，索引状态设置为未构建")
            return
        
        # 检查关键文件
        missing_files = []
        invalid_files = []
        
        for file in key_files:
            file_path = output_path / file
            if not file_path.exists():
                missing_files.append(file)
            else:
                # 检查文件是否有效（非空且可读）
                try:
                    if file_path.stat().st_size == 0:
                        invalid_files.append(f"{file} (空文件)")
                    else:
                        # 尝试读取parquet文件以验证文件完整性
                        try:
                            df = pd.read_parquet(file_path)
                            if len(df) == 0:
                                invalid_files.append(f"{file} (无数据)")
                        except Exception as read_error:
                            invalid_files.append(f"{file} (读取错误: {str(read_error)})")
                except Exception as e:
                    invalid_files.append(f"{file} (文件错误: {str(e)})")
        
        # 检查向量存储目录
        vector_store_path = output_path / "lancedb"
        vector_store_valid = False
        if vector_store_path.exists():
            # 检查关键的向量存储表（使用正确的命名格式）
            vector_tables = [
                "default-entity-description.lance",
                "default-text_unit-text.lance"
            ]
            existing_tables = [table for table in vector_tables 
                             if (vector_store_path / table).exists()]
            if len(existing_tables) >= 1:  # 至少有一个向量表存在
                vector_store_valid = True
        
        # 根据检查结果设置状态
        if missing_files or invalid_files:
            self.index_status = IndexStatus.NOT_BUILT
            if missing_files:
                logger.info(f"缺失关键索引文件: {missing_files}")
            if invalid_files:
                logger.info(f"无效索引文件: {invalid_files}")
            logger.info("索引状态设置为未构建")
        elif not vector_store_valid:
            self.index_status = IndexStatus.NOT_BUILT
            logger.info("向量存储不完整，索引状态设置为未构建")
        else:
            self.index_status = IndexStatus.READY
            logger.info("检测到完整的索引文件，索引状态设置为就绪") 