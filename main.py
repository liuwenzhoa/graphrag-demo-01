#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG 交互式控制台界面
提供循环对话框，通过Y/N输入控制程序流程
"""

import os
import asyncio
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv  # 添加dotenv导入
import json
from typing import Any

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

# 导入提示词常量，参考graphrag/graphrag/cli/initialize.py
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

class GraphRAGInteractive:
    """GraphRAG交互式控制台界面
    
    这个类提供了一个交互式控制台界面，通过Y/N输入进行控制，
    包括：
    - 配置管理
    - 模型管理
    - 索引构建
    - 多种查询方法（全局、本地、DRIFT、基本）
    """
    
    def __init__(self):
        """初始化GraphRAG交互式界面类"""
        self.project_dir = None  # 项目根目录路径，存储所有项目文件的基础目录
        self.output_dir = None  # 索引输出目录路径，存储GraphRAG生成的索引文件
        self.config = None  # GraphRAG配置对象，包含所有模型、输入、输出、向量存储等配置信息
        self.model_manager = ModelManager()  # 模型管理器实例，负责创建和管理LLM模型实例
        self.chat_model = None  # 聊天完成模型实例，用于生成文本响应
        self.embedding_model = None  # 嵌入模型实例，用于生成文本向量表示
        self.token_encoder = None  # 令牌编码器，用于计算和限制输入/输出令牌数量
        
        # 索引数据：以下变量存储从Parquet文件加载的原始DataFrame
        self.entity_df = None  # 存储所有提取的实体及其属性的DataFrame
        self.community_df = None  # 存储实体社区层次结构的DataFrame
        self.report_df = None  # 存储每个社区的摘要报告的DataFrame
        self.text_unit_df = None  # 存储文本块的DataFrame
        self.relationship_df = None  # 存储实体间的关系的DataFrame
        self.covariate_df = None  # 存储从文本中提取的协变量(声明、事实等)的DataFrame，可能为None
        self.document_df = None  # 存储原始文档的元数据和引用信息的DataFrame
        
        # 向量存储：以下变量是连接到LanceDB向量存储的实例
        self.description_embedding_store = None  # 实体描述向量存储，用于本地搜索和DRIFT搜索
        self.text_embedding_store = None  # 文本单元向量存储，用于基本搜索
        self.community_content_embedding_store = None  # 社区报告内容向量存储，用于DRIFT搜索
        
        # 添加上下文数据成员变量，用于存储最近一次查询的上下文数据
        self.context_data = {}
    
    async def start(self):
        """启动交互式界面"""
        print("\n" + "="*60)
        print("欢迎使用 GraphRAG 交互式控制台!")
        print("="*60)
        
        # 设置项目目录
        await self._setup_project_directory()
        
        # 设置配置
        await self._setup_config()
        
        # 生成提示词文件
        await self._generate_prompts()
        
        # 设置模型实例，当前版本不设置模型实例，使用API的方式查询会根据graphrag配置提取模型信息并自行创建模型实例。
        # await self._setup_models()
        
        # 构建或加载索引
        await self._handle_indexing()
        
        # 进入查询循环
        await self._query_loop()
    
    async def _setup_project_directory(self):
      """设置项目目录
      
      此函数用于交互式设置GraphRAG项目的工作目录。它会提示用户输入目录路径，
      处理默认值，验证目录是否存在，并在需要时创建目录。
      """
      while True:
          # 定义默认目录为当前目录
          default_dir = '.'
          # 提示用户输入目录路径，提示中包含默认值
          dir_input = input(f"\n请输入项目目录路径 [按回车使用默认值: {default_dir}]: ")
          
          # 如果用户输入了有效路径则使用用户输入，否则使用默认值
          self.project_dir = dir_input.strip() if dir_input.strip() else default_dir
          
          # 验证目录是否存在
          if not os.path.exists(self.project_dir):
              # 目录不存在时询问用户是否创建新目录
              create_dir = input(f"目录 '{self.project_dir}' 不存在，是否创建? (Y/N): ")
              if create_dir.upper() == 'Y':
                  # 用户同意创建目录，使用os.makedirs创建
                  # exist_ok=True确保在目录已存在时不会引发错误
                  os.makedirs(self.project_dir, exist_ok=True)
                  print(f"已创建目录: {self.project_dir}")
                  break  # 目录已创建，退出循环
              else:
                  # 用户不同意创建目录，终止程序
                  print("您选择不创建目录，程序即将退出。")
                  import sys
                  sys.exit(0)
          else:
              # 目录已存在，无需创建
              print(f"使用项目目录: {self.project_dir}")
              break  # 目录已确认，退出循环
      
      # 创建必要的子目录
      # 创建输出目录：在项目目录下创建output子目录用于存储处理结果
      self.output_dir = os.path.join(self.project_dir, "output")
      os.makedirs(self.output_dir, exist_ok=True)
      print(f"已创建输出目录: {self.output_dir}")
      
      # 创建输入目录
      input_dir = os.path.join(self.project_dir, "input")
      os.makedirs(input_dir, exist_ok=True)
      print(f"已创建输入目录: {input_dir}")
      
      # 创建缓存目录
      cache_dir = os.path.join(self.project_dir, "cache")
      os.makedirs(cache_dir, exist_ok=True)
      print(f"已创建缓存目录: {cache_dir}")
      
      # 创建提示词目录
      prompts_dir = os.path.join(self.project_dir, "prompts")
      os.makedirs(prompts_dir, exist_ok=True)
      print(f"已创建提示词目录: {prompts_dir}")
        
      # 处理.env文件
      env_path = os.path.join(self.project_dir, ".env")
      # 如果.env文件存在，直接加载环境变量
      if os.path.exists(env_path):
          try:
              load_dotenv(env_path)
              print(f"已加载环境变量配置文件: {env_path}")
          except Exception as e:
              print(f"加载.env文件时出错: {str(e)}")
    
    async def _setup_config(self):
        """设置配置，包括从YAML文件加载或创建新配置"""
        # 构建配置文件的完整路径（项目目录下的settings.yaml文件）
        config_path = Path(self.project_dir) / "settings.yaml"
        
        # 检查配置文件是否已存在
        if config_path.exists():
            # 如果配置文件存在，询问用户是否要使用该文件
            load_from_yaml = input("\n检测到settings.yaml文件，是否从YAML文件加载配置? (Y/N): ")
            # 如果用户选择加载YAML文件
            if load_from_yaml.upper() == 'Y':
                # 从YAML文件加载GraphRAG配置
                try:
                    self.config = load_config(Path(self.project_dir))
                    print(f"已从YAML文件加载配置: {config_path}")
                except Exception as e:
                    print(f"加载配置时出错: {str(e)}")
                    # 如果加载失败，则创建默认配置
                    self._create_config()
                # 加载完成后返回，不再执行后续创建配置的步骤
                return
        
        # 如果配置文件不存在，或者用户选择不使用现有配置文件,调用创建默认配置的方法
        self._create_config()
    
    def _create_config(self):
        """创建GraphRAG配置对象
        
        创建包含完整模型、输入、分块、输出、向量存储、工作流和搜索配置的GraphRAG配置对象。
        """
        print("\n创建GraphRAG配置...")
        # 获取模型配置
        chat_model_config, embedding_model_config = self._create_model_configs()
        
        # 创建配置字典，参考 D:\graphrag_project\graphrag\docs_zh\config\yaml.md
        config_data = {
            # 模型配置
            "models": {
                "default_chat_model": chat_model_config,
                "default_embedding_model": embedding_model_config
            },
            
            # 输入配置
            "input": {
                "type": "file",  # 输入类型：可以是"file"(本地文件)或"blob"(Azure Blob存储)
                "file_type": "text",  # 文件类型：可以是"text"(纯文本)、"csv"(表格数据)或"json"(结构化数据)
                "base_dir": os.path.join(self.project_dir, "input"),  # 输入文件的基础目录路径
                # 以下是可选参数，根据不同输入类型和需求可以启用
                # "text_column": "content",  # (CSV/JSON)包含主要文本内容的列名
                # "title_column": "title",  # (CSV/JSON)包含文档标题的列名
                # "metadata": ["author", "date"],  # (CSV/JSON)要保留为元数据的其他列名列表
                # "file_pattern": ".*\\.txt$",  # 匹配输入文件的正则表达式，默认基于file_type自动设置
                # "file_filter": {},  # 用于过滤输入文件的键值对
                # "encoding": "utf-8",  # 输入文件的编码，默认为UTF-8
                # "connection_string": "",  # (仅Blob)Azure存储连接字符串
                # "storage_account_blob_url": "",  # (仅Blob)存储账户Blob URL
                # "container_name": "",  # (仅Blob)Azure存储容器名称
            },
            
            # 分块配置
            "chunks": {
                "size": 1200,  # 标记中的最大块大小。这是块的令牌长度，适合大多数LLM的上下文窗口大小
                "overlap": 100,  # 标记中的块重叠。重叠可以确保相邻块之间的上下文连续性，防止概念被分割
                "strategy": "tokens",  # 分块策略：可以是"tokens"（按令牌数分割）或"sentences"（按句子分割）
                # 以下是可选参数，根据文档提供完整配置选项
                # "group_by_columns": [],  # 在分块前按这些字段对文档进行分组，确保相关数据保持在一起
                # "encoding_model": "cl100k_base",  # 用于在标记边界上分割的文本编码模型
                # "prepend_metadata": False,  # 确定是否应在每个块的开头添加元数据值
                # "chunk_size_includes_metadata": False,  # 指定块大小计算是否应包括元数据标记
            },
            
            # 输出配置
            "output": {
                "type": "file",  # 存储类型：可以是"file"(本地文件)、"memory"(内存)、"blob"(Azure Blob)或"cosmosdb"(CosmosDB)
                "base_dir": os.path.join(self.project_dir, "output"),  # 相对于根目录的输出工件写入基础目录
                # 以下是可选参数，根据存储类型可能需要
                # "connection_string": "",  # (仅限blob/cosmosdb)Azure存储连接字符串
                # "container_name": "",  # (仅限blob/cosmosdb)Azure存储容器名称
                # "storage_account_blob_url": "",  # (仅限blob)要使用的存储账户blob URL
                # "cosmosdb_account_blob_url": "",  # (仅限cosmosdb)要使用的CosmosDB账户blob URL
            },

            # 更新索引输出配置
            "update_index_output": {
                "type": "file",  # 存储类型：可以是"file"(本地文件)、"memory"(内存)、"blob"(Azure Blob)或"cosmosdb"(CosmosDB)
                "base_dir": os.path.join(self.project_dir, "output"),  # 相对于根目录的输出工件写入基础目录
                # 以下是可选参数，根据存储类型可能需要
                # "connection_string": "",  # (仅限blob/cosmosdb)Azure存储连接字符串
                # "container_name": "",  # (仅限blob/cosmosdb)Azure存储容器名称
                # "storage_account_blob_url": "",  # (仅限blob)要使用的存储账户blob URL
                # "cosmosdb_account_blob_url": "",  # (仅限cosmosdb)要使用的CosmosDB账户blob URL
            },

            # 缓存配置
            "cache": {
                "type": "file",  # 存储类型：可以是"file"(本地文件)、"memory"(内存)、"blob"(Azure Blob)或"cosmosdb"(CosmosDB)
                "base_dir": os.path.join(self.project_dir, "cache"),  # 缓存文件写入位置，用于存储LLM调用结果以提高性能
                # 以下是可选参数，根据存储类型可能需要
                # "connection_string": "",  # (仅限blob/cosmosdb)Azure存储连接字符串
                # "container_name": "",  # (仅限blob/cosmosdb)Azure存储容器名称
                # "storage_account_blob_url": "",  # (仅限blob)要使用的存储账户blob URL
                # "cosmosdb_account_blob_url": "",  # (仅限cosmosdb)要使用的CosmosDB账户blob URL
            },

            # 报告配置
            "reporting": {
                "type": "file",  # 报告类型：可以是"file"(写入文件)、"console"(输出到控制台)或"blob"(Azure Blob存储)
                "base_dir": os.path.join(self.project_dir, "logs"),  # 报告文件的写入位置，用于记录常见事件和错误消息
                # 以下是可选参数，根据报告类型可能需要
                # "connection_string": "",  # (仅限blob)Azure存储连接字符串
                # "container_name": "",  # (仅限blob)Azure存储容器名称
                # "storage_account_blob_url": "",  # (仅限blob)要使用的存储账户blob URL
            },
            
            # 向量存储配置
            "vector_store": {
                "default_vector_store": {  # 索引标识符，用于在其他配置中引用此向量存储，如embed_text.vector_store_id
                    "type": "lancedb",  # 向量存储类型：可以是"lancedb"、"faiss"、"chromadb"、"azure_search"等
                    "db_uri": os.path.join(self.project_dir, "output", "lancedb"),  # LanceDB数据库URI路径
                    "container_name": "default",  # 容器名称，存储给定数据集所有索引的表
                    "overwrite": True,  # 如果容器已存在，是否覆盖它，默认为False
                    # 以下是可选参数，根据存储类型可能需要
                    # "url": "",  # (仅限AI Search)AI Search端点URL
                    # "api_key": "",  # (仅限AI Search)AI Search API密钥
                    # "audience": "",  # (仅限AI Search)如果使用托管身份认证，则为托管身份令牌的受众
                    # "database_name": "",  # (仅限cosmosdb)数据库名称
                }
                # 多索引配置示例：
                # "secondary_vector_store": {
                #     "type": "lancedb", 
                #     "db_uri": os.path.join(self.project_dir, "output", "lancedb2"),
                #     "container_name": "default",
                #     "overwrite": True,
                #     "index_name": "domain2"  # 多索引搜索时使用，标识索引名称
                # }
            },
            
            # 工作流配置
            "workflows": [
                # 以下是标准GraphRAG索引流水线的核心工作流（Standard方法）
                "create_base_text_units",  # 创建基础文本单元（文本分块），将文档分割成适合上下文窗口的小块
                "create_final_documents",  # 创建最终文档，处理文档元数据并链接到文本单元
                "extract_graph",  # 使用LLM从文本单元中提取实体和关系，构建知识图谱
                "finalize_graph",  # 完成图的处理，包括计算度、布局和可选的GraphML快照
                "create_communities",  # 使用Leiden算法创建社区层次结构，识别实体群组
                "create_final_text_units",  # 创建最终的文本单元，完成处理流程
                "create_community_reports",  # 生成每个社区的摘要报告，综合其实体和关系描述
                "generate_text_embeddings",  # 生成文本嵌入向量，用于向量检索
                
                # 如果我们不定义workflows配置的话，GraphRAG API的build_index函数会根据is_update_run参数来决定是否追加更新工作流
                # 如果我们定义了workflows配置的话，需要在_build_index函数中显式追加增量更新工作流
                # "update_final_documents",  # 更新最终文档
                # "update_entities_relationships",  # 更新实体和关系
                # "update_text_units",  # 更新文本单元
                # "update_covariates",  # 更新协变量
                # "update_communities",  # 更新社区
                # "update_community_reports",  # 更新社区报告
                # "update_text_embeddings",  # 更新文本嵌入
                # "update_clean_state",  # 清理更新状态
                
                # 以下是可选工作流，可以根据需要启用
                # "extract_covariates",  # 提取协变量（声明、事实等），需要在配置中启用extract_claims
                # "prune_graph",  # 剪枝图，移除噪音和低质量连接，通常用于Fast方法
                # "extract_graph_nlp",  # 使用NLP方法（而非LLM）提取图，用于Fast方法
                # "create_community_reports_text",  # 使用文本单元（而非实体/关系描述）创建社区报告，用于FastGraphRAG
                # 特定用例的简化工作流配置（参考byog.md文档）
                # 如果只需要全局搜索功能: ["create_communities", "create_community_reports"]
                # 如果需要本地搜索、DRIFT搜索或基本搜索: ["create_communities", "create_community_reports", "generate_text_embeddings"]
                # 如果使用FastGraphRAG（NLP+LLM混合方法）: ["create_base_text_units", "create_final_documents", "extract_graph_nlp", "prune_graph", "finalize_graph", "create_communities", "create_final_text_units", "create_community_reports_text", "generate_text_embeddings"]
            ],
            
            # ======= 以下是操作特定的配置 =======
            
            # 文本嵌入配置
            "embed_text": {
                "enabled": True,  # 是否启用文本嵌入，默认为True
                "model_id": "default_embedding_model",  # 使用的嵌入模型ID，必须已在models部分定义
                "vector_store_id": "default_vector_store",  # 使用的向量存储ID，必须已在vector_store部分定义
                "batch_size": 10,  # 批处理大小，阿里云API要求不超过10
                "batch_max_tokens": 8000,  # 单个批次的最大标记总数，防止超出API限制
                # "names": ["title", "text", "description", "content"],  # 要运行的嵌入名称列表（必须在支持的列表中）
            },
            
            # 实体图谱提取配置
            "extract_graph": {
                "model_id": "default_chat_model",  # 使用的聊天模型ID，必须已在models部分定义
                "prompt": "prompts/extract_graph.txt",  # 实体提取提示文件路径
                "entity_types": ["organization", "person", "geo", "event"],  # 要提取的实体类型列表:"organization"：组织，"person"：人，"geo"：地理位置，"event"：事件
                # 控制每个文本单元的后续提取次数，增加可以提高召回率但降低精度
                # 会给模型发送后续请求，要求它找出之前可能遗漏的实体和关系
                # 增加max_gleanings会让系统有更多机会要求模型找出更多实体
                # 如果第一次模型只返回<|COMPLETE|>，后续的提取尝试可能会成功获取实体
                "max_gleanings": 2,
            },
            
            # 描述摘要配置
            "summarize_descriptions": {
                "model_id": "default_chat_model",  # 使用的聊天模型ID，必须已在models部分定义
                "prompt": "prompts/summarize_descriptions.txt",  # 摘要提示文件路径
                "max_length": 1000,  # 生成摘要的最大长度（字符数）
                "max_input_length": 12000,  # 输入文本的最大长度，超过将被截断
            },
            
            # NLP图谱提取配置（用于Fast方法）
            "extract_graph_nlp": {
                "normalize_edge_weights": True,  # 是否标准化边权重
                "text_analyzer": {
                    "extractor_type": "regex_english",  # 提取器类型，可选: "regex_english", "syntactic_parser", "cfg"
                    # 以下是可选参数，根据需要取消注释
                    # "model_name": "en_core_web_md",  # NLP模型名称（用于基于SpaCy的模型）
                    # "max_word_length": 15,  # 允许的最长单词长度
                    # "word_delimiter": " ",  # 分割单词的分隔符
                    # "include_named_entities": True,  # 是否在名词短语中包含命名实体
                    # "exclude_nouns": None,  # 要排除的名词列表，None表示使用内部停用词列表
                    # "exclude_entity_tags": ["DATE"],  # 要忽略的实体标签列表
                    # "exclude_pos_tags": ["DET", "PRON", "INTJ", "X"],  # 要忽略的词性标签列表
                    # "noun_phrase_tags": ["PROPN", "NOUNS"],  # 名词短语标签列表
                    # 下面是CFG语法配置，用于匹配名词短语（仅当extractor_type为"cfg"时使用）
                    # "noun_phrase_grammars": {
                    #     "PROPN,PROPN": "PROPN",
                    #     "NOUN,NOUN": "NOUNS",
                    #     "NOUNS,NOUN": "NOUNS",
                    #     "ADJ,ADJ": "ADJ",
                    #     "ADJ,NOUN": "NOUNS",
                    # },
                },
            },
            
            # 图剪枝配置
            # "prune_graph": {
            #     "min_node_freq": 2,  # 允许的最小节点频率，低于此值的节点将被删除
            #     "max_node_freq_std": None,  # 允许的最大节点频率标准偏差，设为None表示不限制
            #     "min_node_degree": 1,  # 允许的最小节点度，低于此值的节点将被删除
            #     "max_node_degree_std": 3.0,  # 允许的最大节点度标准偏差，高于此值的节点将被修剪
            #     "min_edge_weight_pct": 0.1,  # 允许的最小边权重百分位数，低于此值的边将被删除
            #     "remove_ego_nodes": False,  # 是否移除自我节点（指向自身的节点）
            #     "lcc_only": True,  # 是否仅使用最大连通分量（Largest Connected Component）
            # },
            
            # 控制社区聚类过程的重要设置，这对知识图谱的组织和后续查询质量有重大影响。
            # max_cluster_size: 10
            # 这个参数控制每个社区的最大节点数，当社区大小超过此值时会被进一步分解为子社区。
            # 作用：平衡社区的凝聚力和覆盖范围
            # 合理设置：
            # 小型图谱（数百节点）：5-8
            # 中型图谱（数千节点）：10-15（当前值适合）
            # 大型图谱（数万节点）：15-25
            # 影响：较小的值产生更精确但更窄的社区报告，较大的值提供更全面但可能更宽泛的信息
            # use_lcc: True
            # 决定是否只使用图中的最大连通分量（Largest Connected Component）进行社区划分。
            # 作用：专注于主要知识网络，忽略孤立的小部分
            # 合理设置：
            # True：适合大多数情况，特别是当你希望专注于主要知识结构
            # False：仅当图谱有多个重要且相当大小的独立部分时
            # 影响：True设置可确保社区划分集中在最相关的信息网络上，提高查询结果一致性
            # seed: 42
            # 控制聚类算法的随机性，确保多次运行产生相同的结果。
            # 作用：保证结果可重现性
            # 合理设置：任何固定整数值都可以，42是常见的默认选择
            # 影响：当现有社区划分效果不理想时，可以尝试更改此值获得不同的划分结果
            # 运行一次默认配置后，检查生成的社区报告质量
            # 如果报告太宽泛或包含过多不相关信息，减小max_cluster_size
            # 如果报告太分散或无法建立更广泛的关联，增大max_cluster_size
            # 多次尝试不同的seed值可能会产生更好的社区划分
            "cluster_graph": {
                "max_cluster_size": 10,  # 最大集群大小，超过此值将创建子集群
                "use_lcc": True,  # 是否仅使用最大连通子图
                "seed": 42,  # 随机种子，确保结果可重现
            },
            
            # 协变量提取配置
            "extract_claims": {
                "enabled": False,  # 是否启用协变量提取（默认禁用，需要用户自定义调优）
                "model_id": "default_chat_model",  # 使用的聊天模型ID
                "prompt": "prompts/extract_claims.txt",  # 协变量提取提示文件路径
                "description": "提取与时间相关的事实声明、决策和行动",  # 描述要提取的声明类型
                "max_gleanings": 3,  # 要使用的最大收集周期数（默认值）
            },
            
            # 社区报告生成配置
            "community_reports": {
                "model_id": "default_chat_model",  # 用于API调用的模型定义名称
                "graph_prompt": "prompts/community_report_graph.txt",  # 基于图生成报告的专用提示
                "text_prompt": "prompts/community_report_text.txt",  # 基于文本生成报告的专用提示
                "max_length": 4000,  # 每个报告的最大输出标记数
                "max_input_length": 8000,  # 生成报告时使用的最大输入标记数
                # 以下是settings.yaml中存在但yaml.md文档中未明确列出的参数
                # "prompt": "prompts/community_report.txt",  # 要使用的提示文件路径
            },
            
            # 图嵌入配置（用于可视化）
            "embed_graph": {
                "enabled": False,  # 是否启用图嵌入（主要用于可视化，默认不启用）
                # "dimensions": 128,  # 产生的向量维度数
                # "num_walks": 10,  # node2vec步行数
                # "walk_length": 80,  # node2vec步行长度
                # "window_size": 10,  # node2vec窗口大小
                # "iterations": 1,  # node2vec迭代次数
                # "random_seed": 42,  # node2vec随机种子，保证结果可复现
                # "strategy": {},  # 完全覆盖嵌入图策略的高级选项，除非有特殊需求否则不需要设置
            },
            
            # UMAP配置（用于降维和可视化）
            "umap": {
                "enabled": False,  # 是否启用UMAP布局，用于为图节点提供适合可视化的x/y坐标
            },
            
            # 图谱快照配置
            "snapshots": {
                "graphml": False,  # 是否将图快照导出到GraphML格式
                "embeddings": False,  # 是否将嵌入快照导出到parquet
            },
            
            # 本地搜索配置
            "local_search": {
                "chat_model_id": "default_chat_model",  # 用于聊天完成调用的模型定义名称
                "embedding_model_id": "default_embedding_model",  # 用于嵌入调用的模型定义名称
                "prompt": "prompts/local_search_system_prompt.txt",  # 要使用的提示文件路径
                # "text_unit_prop": 0.5,  # 文本单元比例，控制上下文中分配给文本单元的令牌比例
                # "community_prop": 0.15,  # 社区比例，控制上下文中分配给社区报告的令牌比例
                # "conversation_history_max_turns": 5,  # 对话历史最大回合数
                # "top_k_entities": 10,  # 前k个映射实体，控制检索的顶级实体数量
                # "top_k_relationships": 10,  # 前k个映射关系，控制检索的顶级关系数量
                # "max_context_tokens": 8192,  # 构建请求上下文使用的最大令牌数
            },
            
            # 全局搜索配置
            "global_search": {
                "chat_model_id": "default_chat_model",  # 用于聊天完成调用的模型定义名称
                "map_prompt": "prompts/global_search_map_system_prompt.txt",  # 全局搜索映射器提示文件路径
                "reduce_prompt": "prompts/global_search_reduce_system_prompt.txt",  # 全局搜索归约器提示文件路径
                "knowledge_prompt": "prompts/global_search_knowledge_system_prompt.txt",  # 全局搜索通用提示文件路径
                # "max_context_tokens": 12000,  # 创建的最大上下文大小，以标记为单位
                # "data_max_tokens": 96000,  # 从归约的响应构建最终响应时使用的最大标记数
                # "map_max_length": 500,  # 映射阶段响应请求的最大长度，以单词为单位
                # "reduce_max_length": 1000,  # 归约阶段响应请求的最大长度，以单词为单位
                # # 动态社区选择相关配置
                # "dynamic_search_threshold": 7,  # 包含社区报告的评级阈值
                # "dynamic_search_keep_parent": True,  # 如果任何子社区相关，则保留父社区
                # "dynamic_search_num_repeats": 1,  # 对同一社区报告进行评级的次数
                # "dynamic_search_use_summary": False,  # 使用社区摘要而不是full_context
                # "dynamic_search_max_level": 2,  # 如果处理的社区都不相关，要考虑的社区层次结构的最大级别
            },
            
            # DRIFT搜索配置
            "drift_search": {
                "chat_model_id": "default_chat_model",  # 用于聊天完成调用的模型定义名称
                "embedding_model_id": "default_embedding_model",  # 用于嵌入调用的模型定义名称
                "prompt": "prompts/drift_search_system_prompt.txt",  # 要使用的提示文件路径
                "reduce_prompt": "prompts/drift_reduce_prompt.txt",  # 要使用的归约器提示文件路径
                # "data_max_tokens": 96000,  # 数据llm最大标记数
                # "reduce_max_tokens": 4000,  # 归约阶段的最大标记数（非o系列模型使用）
                # "reduce_max_completion_tokens": 4000,  # 归约阶段的最大标记数（仅用于o系列模型）
                # "concurrency": 1,  # 并发请求数，控制搜索并发度，防止速率限制
                # 实现参考：D:\graphrag_project\graphrag\graphrag\query\structured_search\drift_search\search.py
                "drift_k_followups": 5,  # 检索的顶级全局结果数，控制搜索广度(默认 20)
                # 实现参考：D:\graphrag_project\graphrag\graphrag\query\structured_search\drift_search\search.py
                "primer_folds": 3,  # 搜索启动的折数(默认 3)
                # "primer_llm_max_tokens": 4000,  # 启动器中LLM的最大标记数
                # 实现参考：D:\graphrag_project\graphrag\graphrag\query\structured_search\drift_search\search.py
                "n_depth": 2,  # 要采取的漂移搜索步骤数，控制搜索深度(默认 3)
                # 实现参考：D:\graphrag_project\graphrag\graphrag\query\structured_search\local_search\search.py
                # "local_search_text_unit_prop": 0.5,  # 专用于文本单元的搜索比例
                # "local_search_community_prop": 0.25,  # 专用于社区属性的搜索比例
                # "local_search_top_k_mapped_entities": 8,  # 在本地搜索期间映射的前K个实体数（原配置中的max_entities）
                # "local_search_top_k_relationships": 10,  # 在本地搜索期间映射的前K个关系数
                "local_search_max_data_tokens": 8000,  # 本地搜索的最大上下文大小，以标记为单位 (text-embedding-v3 模型限制8192)
                # "local_search_temperature": 0.5,  # 本地搜索的温度
                # "local_search_top_p": 1.0,  # 本地搜索的top-p值
                # "local_search_n": 1,  # 本地搜索的生成数量
                # "local_search_llm_max_gen_tokens": 4000,  # 本地搜索中LLM的最大生成标记数（非o系列模型使用）
                # "local_search_llm_max_gen_completion_tokens": 4000,  # 本地搜索中LLM的最大生成标记数（仅用于o系列模型）
            },
            
            # 基本搜索配置
            "basic_search": {
                "chat_model_id": "default_chat_model",  # 用于聊天完成调用的模型ID，必须已在models部分定义
                "embedding_model_id": "default_embedding_model",  # 用于嵌入调用的模型ID，必须已在models部分定义
                "prompt": "prompts/basic_search_system_prompt.txt",  # 基本搜索使用的提示文件
                # "k": 10,  # 从向量存储中检索用于上下文构建的文本单元数量
            },
        }
        
        # 创建GraphRAG配置对象
        try:
            self.config = create_graphrag_config(config_data, root_dir=self.project_dir)
            print("已创建完整的GraphRAG配置对象")
        except Exception as e:
            print(f"创建配置对象时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_model_configs(self):
        """创建默认的模型配置对象
        
        创建包含完整参数的模型配置对象，包括API密钥、模型类型、并行化参数、异步模式等。
        这些配置对象用于初始化GraphRAG的聊天模型和嵌入模型，支持各种高级功能如速率限制和缓存。
        
        返回:
            tuple: (chat_model_config, embedding_model_config) 元组，包含聊天模型和嵌入模型的配置
        """
        # 获取API密钥
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        deepseek_api_base = os.environ.get("DEEPSEEK_API_BASE")
        aliyun_api_key = os.environ.get("ALIYUN_API_KEY")
        aliyun_api_base = os.environ.get("ALIYUN_API_BASE")
        
        # 标记需要更新的环境变量
        updated_env_vars = {}
        
        if not deepseek_api_key:
            deepseek_api_key = input("请输入您的DeepSeek API密钥: ")
            os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
            updated_env_vars["DEEPSEEK_API_KEY"] = deepseek_api_key
        
        if not deepseek_api_base:
            deepseek_api_base = input("请输入DeepSeek API基础URL [默认: https://api.deepseek.com/v1]: ")
            if not deepseek_api_base:
                deepseek_api_base = "https://api.deepseek.com/v1"
            os.environ["DEEPSEEK_API_BASE"] = deepseek_api_base
            updated_env_vars["DEEPSEEK_API_BASE"] = deepseek_api_base
        
        if not aliyun_api_key:
            aliyun_api_key = input("请输入您的Aliyun API密钥: ")
            os.environ["ALIYUN_API_KEY"] = aliyun_api_key
            updated_env_vars["ALIYUN_API_KEY"] = aliyun_api_key
        
        if not aliyun_api_base:
            aliyun_api_base = input("请输入Aliyun API基础URL [默认: https://dashscope.aliyuncs.com/compatible-mode/v1]: ")
            if not aliyun_api_base:
                aliyun_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            os.environ["ALIYUN_API_BASE"] = aliyun_api_base
            updated_env_vars["ALIYUN_API_BASE"] = aliyun_api_base
        
        # 如果有新输入的环境变量，询问是否保存到.env文件
        if updated_env_vars:
            save_to_env = input("是否将API密钥保存到.env文件中? (Y/N): ")
            if save_to_env.upper() == 'Y':
                env_path = os.path.join(self.project_dir, ".env")
                try:
                    # 如果.env文件不存在或为空，先创建带有注释的文件
                    if not os.path.exists(env_path) or os.path.getsize(env_path) == 0:
                        with open(env_path, 'w', encoding='utf-8') as f:
                            f.write("# GraphRAG 环境变量配置文件\n")
                            f.write("# 在此处添加API密钥和其他配置\n")
                    
                    # 追加新的环境变量（不修改已有的）
                    with open(env_path, 'a', encoding='utf-8') as f:
                        for key, value in updated_env_vars.items():
                            f.write(f"{key}={value}\n")
                    
                    print(f"新的API密钥已追加到: {env_path}")
                except Exception as e:
                    print(f"保存API密钥到.env文件时出错: {str(e)}")
                    print("将继续使用内存中的环境变量")
        
        # 配置聊天模型
        # LanguageModelConfig是GraphRAG框架中用于配置语言模型的核心类
        # 这个类在GraphRAG 2.2.0及以上版本中得到了完善，特别是增加了对OpenAI o系列模型的支持。
        chat_config = LanguageModelConfig(
            type=ModelType.OpenAIChat,  # 模型类型，指定使用OpenAI兼容的聊天模型
            api_base=deepseek_api_base,  # API基础URL，指定DeepSeek API服务器地址
            auth_type="api_key",  # 认证类型，支持api_key或azure_managed_identity
            api_key=deepseek_api_key,  # API密钥，用于认证请求
            # 使用的具体模型名称：deepseek-chat qwen3-235b-a22b qwen-max qwen-max-latest
            # 如果是 qwen-max-latest：
            #   那么问的问题就需要精准一些，否则Map阶段返回的所有响应评分为0，表示系统认为数据集中没有相关信息
            # 如果是 qwen3-235b-a22b：
            #   修改源码：\graphrag\query\structured_search\global_search\search.py
            #   GlobalSearch类中添加如下代码：self.map_llm_params["extra_body"] = {"enable_thinking": False}
            #   修改源码：\graphrag\language_model\providers\fnllm\models.py
            #   添加如下代码：kwargs["extra_body"]["enable_thinking"] = False
            #   注意：对于查询来说其实只改 models.py 就可以。因为 search.py 最终数据会传输到 models.py 中。
            model="qwen3-235b-a22b",
            encoding_model="cl100k_base", # 指定用于token编码的模型，如"cl100k_base"(针对非OpenAI原生模型)
            model_supports_json=True,  # 模型是否支持返回JSON格式
            async_mode="threaded", # 异步处理模式，可选asyncio或threaded
            max_retries=2, # 调用大模型进行实体和关系提取时最大重试次数，推荐使用10次，-1的动态最大重试支持已在2.3.0版本中移除
            parallelization_num_threads=50, # 控制向模型API发送请求的并行程度，从而设置线程数量，通常建议不超过可用CPU核心数的2-4倍
            parallelization_stagger=0.3, # 并行请求之间的时间间隔，单位为秒，用于防止API过载或速率限制错误
        )
        
        # 配置嵌入模型
        embedding_config = LanguageModelConfig(
            type=ModelType.OpenAIEmbedding,  # 模型类型，指定使用OpenAI兼容的嵌入模型
            api_base=aliyun_api_base,  # API基础URL，指定阿里云API服务器地址
            auth_type="api_key",  # 认证类型，支持api_key或azure_managed_identity
            api_key=aliyun_api_key,  # API密钥，用于认证请求
            model="text-embedding-v3",  # 使用的具体嵌入模型名称
            encoding_model="cl100k_base", # 显式指定编码器，针对非OpenAI原生模型
            model_supports_json=True,  # 模型是否支持返回JSON格式
            async_mode="threaded",  # 异步处理模式，可选asyncio或threaded
            max_retries=2,  # 调用大模型进行实体和关系提取时最大重试次数，推荐使用10次
            parallelization_num_threads=50,  # 控制向模型API发送请求的并行程度，从而设置线程数量，通常建议不超过可用CPU核心数的2-4倍
            parallelization_stagger=0.3,  # 并行请求之间的时间间隔，单位为秒，用于防止API调用过于频繁
        )
        
        return chat_config, embedding_config
    
    async def _generate_prompts(self):
        """为prompts目录生成所需的提示词文件并更新配置中的路径"""
        print("\n正在检查提示词文件...")
        
        # 获取prompts目录路径（已在_setup_project_directory中创建）
        prompts_dir = Path(self.project_dir) / "prompts"
        
        # 参考：graphrag/graphrag/cli/initialize.py
        # 定义要生成的提示词文件及其内容
        prompts = {
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
        
        # 是否重新创建所有提示词文件
        recreate_all = False
        # 遍历prompts目录，检查是否存在对应的提示词文件
        existing_files = [filename for filename in prompts.keys() if (prompts_dir / filename).exists()]
        # 如果存在提示词文件，询问用户是否要重新创建
        if existing_files:
            print(f"在 {prompts_dir} 目录中发现 {len(existing_files)} 个已存在的提示词文件:")
            for file in existing_files:
                print(f"- {file}")
                
            recreate_choice = input("是否要重新创建这些提示词文件? 这将覆盖任何现有的自定义内容。 (Y/N): ")
            recreate_all = recreate_choice.upper() == 'Y'
        
        created_count = 0 # 创建的提示词文件数量
        skipped_count = 0 # 跳过的提示词文件数量
        # 遍历prompts目录，写入提示词文件
        for filename, content in prompts.items():
            # 获取提示词文件路径
            file_path = prompts_dir / filename
            # 如果提示词文件不存在，或者重新创建所有提示词文件，则写入提示词文件
            should_write = not file_path.exists() or recreate_all
            # 写入提示词文件
            if should_write:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"已创建提示词文件: {file_path}")
                created_count += 1
            else:
                print(f"已跳过提示词文件(已存在): {filename}")
                skipped_count += 1
        
        # 更新 graphrag 配置中的提示词路径
        self._update_prompt_paths_in_config()
        
        print(f"提示词文件处理完成: 创建了 {created_count} 个文件, 跳过了 {skipped_count} 个文件")
    
    def _update_prompt_paths_in_config(self):
        """更新配置对象中的提示词路径"""
        if not hasattr(self, "config") or self.config is None:
            print("警告: 配置对象未初始化，无法更新提示词路径")
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
        
        if hasattr(self.config, "basic_search"):
            self.config.basic_search.prompt = "prompts/basic_search_system_prompt.txt"
        
        if hasattr(self.config, "question_gen"):
            self.config.question_gen.prompt = "prompts/question_gen_system_prompt.txt"
    
    async def _setup_models(self):
        """设置模型实例"""
        # 如果配置存在并且有模型设置，则使用配置中的模型
        if self.config and self.config.models:
            print("\n使用配置文件中的模型设置模型实例...")
            
            # 从配置中获取模型配置
            chat_model_config = None # 聊天模型配置
            embedding_model_config = None # 嵌入模型配置
            chat_model_key = None # 聊天模型键名
            embedding_model_key = None # 嵌入模型键名
            
            # 查找配置中的聊天模型和嵌入模型配置
            for model_key, model_config in self.config.models.items():
                if model_config.type == ModelType.OpenAIChat:
                    chat_model_key = model_key
                    chat_model_config = model_config
                elif model_config.type == ModelType.OpenAIEmbedding:
                    embedding_model_key = model_key
                    embedding_model_config = model_config
        
            # 根据模型配置使用ModelManager初始化模型实例
            # ModelManager负责管理这些模型实例，它会使用ModelFactory来根据配置创建适当的模型实现（如OpenAIChat或OpenAIEmbedding）。
            self.chat_model = self.model_manager.get_or_create_chat_model(
                        name=chat_model_key,  # 使用配置中的键名
                        model_type=chat_model_config.type, # 模型类型
                        config=chat_model_config # 模型配置
            )
            self.embedding_model = self.model_manager.get_or_create_embedding_model(
                        name=embedding_model_key,  # 使用配置中的键名
                        model_type=embedding_model_config.type,
                        config=embedding_model_config
            )

            # 创建token编码器
            # 从chat_model_config中获取encoding_model
            encoding_model = chat_model_config.encoding_model
            # 如果配置中指定了编码器，则使用它
            if encoding_model:
                try:
                    self.token_encoder = tiktoken.get_encoding(encoding_model)
                    print(f"使用模型 {chat_model_key} 配置中的编码器: {self.token_encoder.name}")
                except Exception as e:
                    print(f"无法使用配置的编码器 {encoding_model}: {e}")
                    self.token_encoder = tiktoken.get_encoding("cl100k_base")
                    print("回退到默认编码器: cl100k_base")
            else:
                # 尝试根据模型名称获取编码器，如果失败则使用cl100k_base
                try:
                    model_name = chat_model_config.model
                    # OpenAI模型通常可以用tiktoken.encoding_for_model识别
                    # 非OpenAI模型通常使用cl100k_base
                    self.token_encoder = tiktoken.encoding_for_model(model_name)
                    print(f"根据模型 {chat_model_key} 匹配的编码器: {self.token_encoder.name}")
                except Exception as e:
                    print(f"设置编码器时出错: {e} 无法为模型 {model_name} 获取编码器")
                    self.token_encoder = tiktoken.get_encoding("cl100k_base")
                    print("回退到默认编码器: cl100k_base")
        else:
            # 如果没有从配置文件加载模型配置，则使用默认设置
            print("\n配置文件没有模型设置，请合理配置，程序即将退出。")
            import sys
            sys.exit(0)
        
        print("语言模型和嵌入模型设置完成")
        
    async def _handle_indexing(self):
        """处理索引构建或加载
        
        该方法管理GraphRAG索引的创建、加载和更新流程:
        - 检测现有索引数据的存在
        - 提供用户交互选项(加载现有索引/增量更新/新建索引)
        - 根据用户选择调用相应处理函数
        - 处理用户输入异常情况
        """
        # 检查是否已有索引数据 - 查找entities.parquet作为判断依据
        # entities.parquet是GraphRAG索引的核心输出文件之一，包含知识图谱中的所有实体
        entities_path = os.path.join(self.output_dir, "entities.parquet")
        
        # 如果索引数据存在，则询问用户是否加载现有索引或执行增量更新
        # GraphRAG支持三种索引操作模式：加载现有索引/增量更新现有索引/创建全新索引
        if os.path.exists(entities_path):
            load_choice = input("\n检测到现有索引数据，请选择操作:\n1. 加载现有索引\n2. 执行增量更新\n3. 构建全新索引\n请选择 [1-3]: ")
            
            if load_choice == '1':
                # 选项1：直接加载现有索引数据而不进行任何重建
                # _load_index_data方法将从parquet文件加载实体、关系、社区等数据到内存
                self._load_index_data()
                return
            elif load_choice == '2':
                # 选项2：执行增量更新，保留现有索引数据并添加新内容
                # 增量更新会处理input目录中的新文件，并更新现有索引而非重建
                method_input = input("选择增量更新方法 (1: Standard, 2: Fast) [默认: 1]: ")
                # Standard方法更全面但更慢，Fast方法更快但可能失去一些深度处理
                method = IndexingMethod.Fast if method_input == "2" else IndexingMethod.Standard
                # 调用_build_index但设置is_update=True表示这是增量更新而非全新构建
                await self._build_index(method, is_update=True)
                # 增量更新完成后，加载新创建的索引数据
                self._load_index_data()
                return
            elif load_choice == '3':
                # 选项3：构建全新索引，清理之前的索引文件
                try:
                    # 清理output目录中的索引文件
                    if os.path.exists(self.output_dir):
                        # 列出output目录中的所有文件和子目录
                        items = os.listdir(self.output_dir)
                        print(f"正在清理output目录中的 {len(items)} 个项目...")
                        for item in items:
                            item_path = os.path.join(self.output_dir, item)
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                                print(f"已删除目录: {item}")
                            else:
                                os.remove(item_path)
                                print(f"已删除文件: {item}")
                        print("索引文件清理完成")
                    # 清理缓存目录
                    cache_dir = os.path.join(self.project_dir, "cache")
                    if os.path.exists(cache_dir):
                        # 列出cache目录中的所有文件和子目录
                        items = os.listdir(cache_dir)
                        print(f"正在清理cache目录中的 {len(items)} 个项目...")
                        for item in items:
                            item_path = os.path.join(cache_dir, item)
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                                print(f"已删除目录: {item}")
                            else:
                                os.remove(item_path)
                                print(f"已删除文件: {item}")
                        print("缓存文件清理完成")
                except Exception as e:
                    print(f"清理文件时出错: {str(e)}")
                    import sys
                    sys.exit(0)
            else:
                # 处理无效输入
                print("无效选择，将询问是否构建新索引")
        
        # 没有现有索引或用户选择不加载或重新构建时，询问是否构建新索引
        build_index = input("\n是否构建新索引? (Y/N): ")
        if build_index.upper() == 'Y':
            # 用户选择构建新索引
            method_input = input("选择索引方法 (1: Standard, 2: Fast) [默认: 1]: ")
            method = IndexingMethod.Fast if method_input == "2" else IndexingMethod.Standard
            # 调用_build_index并设置is_update=False表示这是全新构建
            await self._build_index(method, is_update=False)
            # 构建完成后，加载新创建的索引数据
            self._load_index_data()
        else:
            # 用户选择不构建新索引，程序无法继续，退出
            print("警告: 没有加载索引数据，程序即将退出。")
            import sys
            sys.exit(0)
    
    async def _build_index(self, method: IndexingMethod = IndexingMethod.Standard, is_update: bool = False):
        """构建GraphRAG索引
        
        此方法调用GraphRAG API执行索引构建过程，这是RAG系统的核心预处理步骤。
        索引过程会处理输入文档，构建知识图谱，并生成用于检索的矢量嵌入。
        
        Args:
            method: 索引构建方法，可选值:
                - IndexingMethod.Standard: 标准方法，执行完整的处理流程
                   包括文本分块、实体提取、关系识别、社区检测等全部步骤
                - IndexingMethod.Fast: 快速方法，简化部分处理步骤
                   适用于快速原型或小型数据集，牺牲部分质量换取速度
            
            is_update: 指定是否为增量更新模式:
                - True: 增量更新模式，仅处理新添加的文档
                       保留现有索引并添加新内容，通常比全量构建快
                - False: 全新构建模式，处理所有输入文档
                        忽略任何现有索引数据，从头开始构建
        
        索引构建过程将生成多个核心Parquet文件:
            - documents.parquet: 原始输入文档的元数据及引用信息
            - text_units.parquet: 文本分块后的单元数据，包含实际文本内容
            - entities.parquet: 从文本中提取的实体及其描述
            - relationships.parquet: 实体之间的关系信息
            - communities.parquet: 社区检测结构，包含层次结构
            - community_reports.parquet: 每个社区的摘要报告
            - covariates.parquet (可选): 存储额外的协变量信息(如果启用了声明提取)
        
        向量数据库目录(output/lancedb/)将包含以下向量索引:
            - default-text_unit-text.lance: 文本单元的向量嵌入
            - default-entity-description.lance: 实体描述的向量嵌入
            - default-community-full_content.lance: 社区报告的向量嵌入
        
        其他辅助文件:
            - stats.json: 索引过程的统计信息
            - context.json: 索引上下文信息
            - graph.graphml (如果启用): 用于可视化的图谱文件
        """

        # 检查输入目录
        input_dir = os.path.join(self.project_dir, "input")
        input_files = os.listdir(input_dir)
        print(f"input目录中输入文件数量: {len(input_files)}")
        if len(input_files) == 0:
            print("错误: 输入目录为空，无法执行索引构建。请确保input目录中有文件后再试。")
            import sys
            sys.exit(0)
            
        # 显示操作类型
        operation_type = "增量更新" if is_update else "全新索引构建"
        print(f"开始【{operation_type}】，使用索引方法: {method}")
        
        # 如果是增量更新模式，且配置中显式定义了workflows
        if is_update and hasattr(self.config, 'workflows'):
            # 默认的增量更新工作流列表
            update_workflows = [
                "update_final_documents",  # 更新最终文档
                "update_entities_relationships",  # 更新实体和关系
                "update_text_units",  # 更新文本单元
                "update_covariates",  # 更新协变量
                "update_communities",  # 更新社区
                "update_community_reports",  # 更新社区报告
                "update_text_embeddings",  # 更新文本嵌入
                "update_clean_state",  # 清理更新状态
            ]
            
            # 检查当前配置的workflows中是否已有增量更新工作流
            existing_update_workflows = [w for w in self.config.workflows if w.startswith("update_")]
            
            # 只添加当前不存在的增量更新工作流
            if not existing_update_workflows:
                print("为增量更新模式自动追加update工作流到工作流配置中：")
                self.config.workflows.extend(update_workflows)
                for i, workflow in enumerate(self.config.workflows, 1):
                    print(f"{i}. {workflow}")
        
        try:
            # 创建进度回调对象，参考：D:\graphrag_project\graphrag\graphrag\api\index.py
            # WorkflowCallbacks是GraphRAG的回调接口，用于报告索引构建过程中的进度和状态
            # ConsoleWorkflowCallbacks是一个具体实现，专门用于在控制台显示进度信息
            # 回调系统会报告每个工作流的开始、进度百分比、完成状态和错误信息
            progress_logger = ConsoleWorkflowCallbacks()
            callbacks = [progress_logger]  # 创建回调列表，可以添加多个回调实现
            
            # 调用GraphRAG API的build_index方法执行索引构建
            # 该方法会按照配置文件中指定的工作流顺序执行一系列操作
            # GraphRAG API会根据is_update_run参数自动选择要使用的工作流
            results = await api.build_index(
                config=self.config,        # 传递GraphRAG配置对象，包含输入/输出路径、模型设置等
                method=method,             # 传递索引方法(Standard或Fast)
                is_update_run=is_update,   # 指定是否为增量更新模式
                callbacks=callbacks,       # 传递回调列表，用于进度报告
                progress_logger=None       # 设为None表示使用默认的进度记录器(NullProgressLogger)
                                           # 实际进度报告由callbacks参数中的对象处理
            )
            
            # 检查索引过程是否成功完成
            if results:
                success = True
                errors = []
                
                # results是一个PipelineRunResult对象列表，每个对象代表一个工作流的运行结果
                print("\n索引构建完成，工作流结果:")
                for result in results:
                    # 如果工作流有错误，显示错误信息，否则显示成功状态
                    status = "error" if result.errors else "success"
                    print(f"工作流名称: {result.workflow}\t状态: {status}")
                    # 如果当前工作流有错误，添加到错误列表并标记整体构建为失败
                    if result.errors and len(result.errors) > 0:
                        success = False
                        errors.extend(result.errors)
                        print(f"错误详情:")
                        for error in result.errors:
                            print(f"  - {error}")
                
                # 输出最终结果状态
                if success:
                    print("索引构建成功完成")
                else:
                    print("索引构建过程中有错误:")
                    for error in errors:
                        print(f"- {error}")
            else:
                # 如果api.build_index没有返回结果，可能是异常终止
                print("索引构建未返回结果，可能是由于配置问题或其他错误导致")
                import sys
                sys.exit(0)
        
        except Exception as e:
            # 捕获并报告索引构建过程中的任何异常
            print(f"\n索引构建过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_index_data(self):
        """加载索引数据
        
        从输出目录加载GraphRAG索引构建生成的所有Parquet文件，保存原始DataFrame用于API调用。
        """
        try:
            print("\n加载索引数据...")
            
            # 读取核心索引数据，保存为DataFrame用于API调用
            self.community_df = pd.read_parquet(f"{self.output_dir}/communities.parquet")
            self.entity_df = pd.read_parquet(f"{self.output_dir}/entities.parquet")
            self.report_df = pd.read_parquet(f"{self.output_dir}/community_reports.parquet")
            self.text_unit_df = pd.read_parquet(f"{self.output_dir}/text_units.parquet")
            self.relationship_df = pd.read_parquet(f"{self.output_dir}/relationships.parquet")
            self.document_df = pd.read_parquet(f"{self.output_dir}/documents.parquet")
            
            # 尝试读取协变量表(如果存在)
            # 注意：协变量表是可选的，在默认索引工作流中通常不会生成
            # 只有在启用了特定的声明提取工作流时才会创建此表
            try:
                self.covariate_df = pd.read_parquet(f"{self.output_dir}/covariates.parquet")
                print(f"加载了协变量数据，包含 {len(self.covariate_df)} 行记录")
            except FileNotFoundError:
                print("协变量表不存在，跳过加载")
                self.covariate_df = None
            except Exception as e:
                print(f"加载协变量表时出错: {str(e)}")
                self.covariate_df = None
            
            # 打印加载的数据信息
            print(f"已加载索引数据:")
            print(f"- 社区总数: {len(self.community_df)}")
            
            # 计算各个社区级别的报告数量，更有用的统计信息
            levels = self.report_df['level'].unique()
            levels.sort()
            for level in levels:
                level_count = len(self.report_df[self.report_df['level'] == level])
                print(f"- 社区报告数 (级别 {level}): {level_count}")
            
            print(f"- 实体数: {len(self.entity_df)}")
            print(f"- 文本单元数: {len(self.text_unit_df)}")
            print(f"- 关系数: {len(self.relationship_df)}")
            print(f"- 文档数: {len(self.document_df)}")
            
            # 连接向量存储
            # 当前版本不连接向量存储，使用API函数查询会根据传入的配置(self.config)自行创建和管理向量存储连接。
            # self._connect_vector_stores()
            
        except Exception as e:
            print(f"加载索引数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _connect_vector_stores(self):
        """连接向量存储
        
        连接到以下LanceDB向量存储:
        - entity-description: 用于实体描述的向量检索
        - text_unit-text: 用于文本单元内容的向量检索
        - community-full_content: 用于社区报告内容的向量检索
        
        这些向量存储是GraphRAG索引构建过程生成的，用于支持不同类型的搜索操作:
        - 基本搜索使用text_unit-text
        - 本地搜索使用entity-description和text_unit-text
        - DRIFT搜索使用所有三个向量存储
        """
        try:
            # 创建LanceDB向量存储实例
            lance_db_path = os.path.join(self.output_dir, "lancedb")
            # 确认向量存储目录存在
            if not os.path.exists(lance_db_path):
                print(f"向量存储目录不存在: {lance_db_path}")
                return
            
            print("\n连接向量存储...")
            
            # 初始化LanceDB向量存储
            container_name = "default"
            
            # 创建并连接向量存储：
            # - 创建三个LanceDBVectorStore实例，分别用于实体描述、文本单元和社区全文内容的嵌入
            # - 连接到指定的LanceDB数据库
            # 实体描述向量存储 - 用于基于实体的搜索（本地搜索、DRIFT搜索）
            self.description_embedding_store = LanceDBVectorStore(
                collection_name=f"{container_name}-entity-description",
            )
            self.description_embedding_store.connect(db_uri=lance_db_path)
            print("已连接实体描述向量存储")
            
            # 文本单元向量存储 - 用于文本相似度搜索（基本搜索）
            self.text_embedding_store = LanceDBVectorStore(
                collection_name=f"{container_name}-text_unit-text",
            )
            self.text_embedding_store.connect(db_uri=lance_db_path)
            print("已连接文本单元向量存储")
            
            # 社区报告向量存储 - 用于社区报告搜索（DRIFT搜索）
            self.community_content_embedding_store = LanceDBVectorStore(
                collection_name=f"{container_name}-community-full_content",
            )
            self.community_content_embedding_store.connect(db_uri=lance_db_path)
            print("已连接社区报告向量存储")
        except Exception as e:
            print(f"连接向量存储时出错: {str(e)}")
            self.description_embedding_store = None
            self.text_embedding_store = None 
            self.community_content_embedding_store = None
    
    async def _query_loop(self):
        """进入查询循环"""
        if self.entity_df is None or self.report_df is None or self.entity_df.empty or self.report_df.empty:
            print("\n警告: 没有加载索引数据，无法执行查询。请先构建或加载索引。")
            import sys
            sys.exit(0)
        
        print("\n" + "-"*60)
        print("进入查询模式。输入'exit'或'quit'退出程序。")
        print("-"*60)
        
        while True:
            query = input("\n请输入您的查询问题 (输入'exit'或'quit'退出): ")
            
            if query.lower() in ['exit', 'quit']:
                print("感谢使用GraphRAG交互式控制台，再见!")
                break
            
            if not query.strip():
                continue
            
            print("\n选择查询方法:")
            print("1. 全局搜索 (适用于整体概览问题，例如'文档中的主要主题是什么?')")
            print("2. 流式全局搜索 (适用于整体概览问题，例如'文档中的主要主题是什么?')")
            print("3. 本地搜索 (适用于特定实体问题，例如'洋甘菊的治疗特性是什么?')")
            print("4. 流式本地搜索 (适用于特定实体问题，实时显示结果)")
            print("5. DRIFT搜索 (结合全局和本地特点的递归搜索)")
            print("6. 流式DRIFT搜索 (结合全局和本地特点的递归搜索，实时显示结果)")
            print("7. 基本搜索 (简单的向量相似度搜索)")
            print("8. 流式基本搜索 (简单的向量相似度搜索，实时显示结果)")
            print("9. 所有方法 (比较所有搜索方法的结果)")
            
            method_choice = input("请选择方法 [1-9]: ")
            
            if method_choice == '1':
                await self._run_global_search(query)
            elif method_choice == '2':
                await self._run_global_search_streaming(query)
            elif method_choice == '3':
                await self._run_local_search(query)
            elif method_choice == '4':
                await self._run_local_search_streaming(query)
            elif method_choice == '5':
                await self._run_drift_search(query)
            elif method_choice == '6':
                await self._run_drift_search_streaming(query)
            elif method_choice == '7':
                await self._run_basic_search(query)
            elif method_choice == '8':
                await self._run_basic_search_streaming(query)
            elif method_choice == '9':
                await self._run_all_searches(query)
            else:
                print("无效的选择，请输入1-9之间的数字")
    
    async def _run_global_search(self, query: str):
        """运行全局搜索"""
        print("\n正在执行全局搜索...")
        try:
            result = await self.global_search(query)
            print("\n全局搜索结果:")
            print("-" * 80)
            print(result)
            print("-" * 80)
        except Exception as e:
            print(f"全局搜索失败: {str(e)}")
    
    async def _run_global_search_streaming(self, query: str):
        """运行流式全局搜索
        使用流式方式执行全局搜索，实时显示结果片段，提供即时反馈。
        参数:
            query: 用户查询文本
        """
        print("\n正在执行流式全局搜索...")
        try:
            # 定义回调函数，用于实时显示响应片段
            def display_chunk(chunk: str):
                print(chunk, end="", flush=True)
            
            # 执行流式搜索
            result = await self.global_search_streaming(query, callback=display_chunk)
            # 搜索完成后打印分隔线
            print("\n" + "-" * 80)
        except Exception as e:
            print(f"流式全局搜索失败: {str(e)}")

    async def _run_local_search(self, query: str):
        """运行本地搜索"""
        print("\n正在执行本地搜索...")
        try:
            result = await self.local_search(query)
            print("\n本地搜索结果:")
            print("-" * 80)
            print(result)
            print("-" * 80)
        except Exception as e:
            print(f"本地搜索失败: {str(e)}")
    
    async def _run_local_search_streaming(self, query: str):
        """
            运行流式本地搜索
            使用流式方式执行本地搜索，实时显示结果片段，提供即时反馈。
            参数:
                query: 用户查询文本
        """
        print("\n正在执行流式本地搜索...")
        try:
            # 定义回调函数，用于实时显示响应片段
            def display_chunk(chunk: str):
                print(chunk, end="", flush=True)
            
            # 执行流式搜索
            result = await self.local_search_streaming(query, callback=display_chunk)
            # 搜索完成后打印分隔线
            print("\n" + "-" * 80)
        except Exception as e:
            print(f"流式本地搜索失败: {str(e)}")
    
    async def _run_drift_search(self, query: str):
        """运行DRIFT搜索"""
        print("\n正在执行DRIFT搜索...")
        try:
            result = await self.drift_search(query)
            print("\nDRIFT搜索结果:")
            print("-" * 80)
            print(result)
            print("-" * 80)
        except Exception as e:
            print(f"DRIFT搜索失败: {str(e)}")
    
    async def _run_drift_search_streaming(self, query: str):
        """运行流式DRIFT搜索
        使用流式方式执行DRIFT搜索，实时显示结果片段，提供即时反馈。
        参数:
            query: 用户查询文本
        """
        print("\n正在执行流式DRIFT搜索...")
        try:
            # 定义回调函数显示流式响应片段
            def display_chunk(chunk: str):
                print(chunk, end="", flush=True)
            
            # 执行流式搜索
            result = await self.drift_search_streaming(query, callback=display_chunk)
            # 搜索完成后打印分隔线
            print("\n" + "-" * 80)
        except Exception as e:
            print(f"流式DRIFT搜索失败: {str(e)}")
            
    async def _run_basic_search(self, query: str):
        """运行基本搜索"""
        print("\n正在执行基本搜索...")
        try:
            k = input("请输入要检索的文本单元数量 [默认: 5]: ")
            k = int(k) if k.strip().isdigit() else 5
            
            result = await self.basic_search(query, k)
            print("\n基本搜索结果:")
            print("-" * 80)
            print(result)
            print("-" * 80)
        except Exception as e:
            print(f"基本搜索失败: {str(e)}")
    
    async def _run_basic_search_streaming(self, query: str):
        """运行流式基本搜索
        使用流式方式执行基本搜索，实时显示结果片段，提供即时反馈。
        参数:
            query: 用户查询文本
        """
        print("\n正在执行流式基本搜索...")
        try:
            # 询问用户希望检索的文本单元数量
            k_input = input("请输入要检索的文本单元数量 [默认: 5]: ")
            k = int(k_input) if k_input.strip().isdigit() else 5
            
            # 定义回调函数显示流式响应片段
            def display_chunk(chunk: str):
                print(chunk, end="", flush=True)
            
            # 执行流式搜索
            result = await self.basic_search_streaming(query, k=k, callback=display_chunk)
            # 搜索完成后打印分隔线
            print("\n" + "-" * 80)
        except Exception as e:
            print(f"流式基本搜索失败: {str(e)}")
            
    async def _run_all_searches(self, query: str):
        """运行所有搜索方法"""
        print("\n正在执行所有搜索方法...")
        
        try:
            # 全局搜索
            global_result = await self.global_search(query)
            print("\n全局搜索结果:")
            print("-" * 80)
            print(global_result)
            print("-" * 80)
            
            # 本地搜索
            local_result = await self.local_search(query)
            print("\n本地搜索结果:")
            print("-" * 80)
            print(local_result)
            print("-" * 80)
            
            # DRIFT搜索
            drift_result = await self.drift_search(query)
            print("\nDRIFT搜索结果:")
            print("-" * 80)
            print(drift_result)
            print("-" * 80)
            
            # 基本搜索
            basic_result = await self.basic_search(query, 5)
            print("\n基本搜索结果 (top_k=5):")
            print("-" * 80)
            print(basic_result)
            print("-" * 80)
        
        except Exception as e:
            print(f"执行所有搜索方法时出错: {str(e)}")
    
    async def global_search(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                            top_p: float = 0.9, max_tokens: int = 4096) -> str:
        """执行全局搜索
        参考：D:\\graphrag_project\\graphrag\\graphrag\\cli\\query.py
        全局搜索是GraphRAG中最适合获取整体知识的搜索方法。它使用"Map-Reduce"模式在所有社区报告上
        执行搜索，先从多个社区报告中提取相关信息(Map阶段)，再将这些信息组合成一个连贯的总体答案(Reduce阶段)。
        
        适用场景：
        - 当您需要了解整个文档集合的宏观信息时，比如"这些文档的主要主题是什么？"
        - 适合寻找跨多个文档或知识领域的模式和关系
        - 当您希望获得对整个知识库的概述或总结时
        - 当您的问题不针对特定实体，而是关于整体内容时
        
        与其他搜索方法的区别：
        - 比本地搜索更全面，关注整体而非特定实体
        - 比基本搜索更智能，利用了社区结构和摘要报告
        - 比DRIFT搜索更直接，不进行复杂的递归探索
        - 计算效率高于DRIFT搜索，但可能缺少深度细节
        
        参数:
            query: 用户查询文本，要搜索的问题或主题
            community_level: 社区层级(0-N)，0表示顶层社区(最宏观)，更高的值表示更细分的社区
            temperature: 控制回答的创造性(0-1)，值越低越确定性，值越高结果越多样化
            top_p: 控制采样词汇的概率阈值(0-1)，降低该值会使输出更加确定性
            max_tokens: 模型生成回答的最大标记数量，限制输出长度
        
        返回:
            查询的文本回答
        """
        # 验证必要的数据和模型是否存在，全局搜索需要实体、社区和社区报告数据
        if (self.config is None or self.entity_df is None or self.community_df is None or 
            self.report_df is None or self.entity_df.empty or self.community_df.empty or 
            self.report_df.empty):
            # 如果缺少任何必要数据，抛出ValueError异常
            raise ValueError("缺少必要的数据，请先加载索引数据")
        
        # 保存原始配置的状态，以便在执行完搜索后恢复，确保临时更改不会影响后续操作
        original_params = {}
        # 通过修改模型配置对象而不是全局搜索配置来调整温度等参数
        if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
            # 获取默认聊天模型配置对象
            model_config = self.config.models['default_chat_model']
            # 记录原始参数值，便于后续恢复
            for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                if hasattr(model_config, param):
                    # 保存原始值以便后续恢复
                    original_params[param] = getattr(model_config, param)
                    # 设置新值（临时覆盖）
                    setattr(model_config, param, value)
        
        try:
            # 准备收集上下文数据
            context_data = {}
            
            # 定义上下文数据回调函数
            def on_context(context):
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            # 使用官方API进行查询，传入原始DataFrame,全局搜索使用Map-Reduce方法处理社区报告，生成综合回答
            # api.global_search是GraphRAG库提供的官方API，用于执行全局搜索
            # 详细参数含义参见 \graphrag\graphrag\api\query.py的global_search函数文档
            # api官方实现参考：\graphrag\graphrag\cli\query.py
            # user_query = query.strip() + " /no_think"
            user_query = query.strip()
            response, api_context_data = await api.global_search(
                config=self.config,  # 配置对象，包含模型设置、提示模板等配置信息
                entities=self.entity_df,  # 实体数据，用于确定社区权重和排名
                communities=self.community_df,  # 社区数据，提供社区的层次结构信息
                community_reports=self.report_df,  # 社区报告，包含每个社区的摘要报告
                community_level=community_level,  # 使用的社区层级深度，0是最顶层(宏观)
                dynamic_community_selection=False,  # 是否动态选择社区，默认关闭，若启用可以根据查询相关性选择社区
                response_type="详细的中文回答",  # 指定输出格式为中文详细回答
                query=user_query,  # 用户查询文本，是搜索的核心输入
                callbacks=callbacks,  # 回调函数列表，用于收集上下文数据和其他事件处理
            )
            
            # 存储上下文数据到成员变量
            self.context_data = context_data
            
            print("\ngraphrag\\graphrag_demo_01\\main.py global_search context_data完整内容:")
            print(json.dumps(context_data, indent=2, ensure_ascii=False, default=str))
            # context_data是一个字典，只包含'reports'键，其值是包含社区报告的DataFrame
            # 全局搜索返回的context_data比较简洁，只包含用于生成答案的社区报告信息
            # 这些报告包含id、title、content和rank等字段，rank值表示报告的重要性评分
            # 用途：帮助理解全局搜索使用了哪些社区报告来生成答案，以及它们的相对重要性
            if 'reports' in context_data:
                # 检查reports的类型
                reports = context_data['reports']
                if isinstance(reports, pd.DataFrame):
                    # 如果是DataFrame，直接处理
                    print(f"使用了 {len(reports)} 个社区报告生成答案")
                    for i, (_, report) in enumerate(reports.iterrows(), 1):
                        # 获取报告的标题和内容
                        title = report.get('title', '未命名社区')
                        # 尝试获取summary字段，如果不存在则使用content
                        summary = report.get('summary', report.get('content', '无摘要'))
                        # 获取rank值(重要性评分)
                        rank = report.get('rank', '未知')  # 获取rank值
                        # 打印报告信息，包括序号、标题和重要性评分
                        print(f"报告 {i}: {title} (重要性评分: {rank})")
                        print(f"摘要: {summary[:500]}..." if len(str(summary)) > 500 else f"摘要: {summary}")
                        print("-" * 20)

                # 处理动态社区选择数据，这是可选功能，当dynamic_community_selection=True时才会出现
                if 'dynamic_selection' in context_data:
                    selection = context_data['dynamic_selection']
                    if isinstance(selection, list):
                        print(f"动态社区选择: 评估了 {len(selection)} 个社区")
                        print(f"选中了 {sum(1 for s in selection if isinstance(s, dict) and s.get('selected', False))} 个社区用于回答")
                    else:
                        print(f"动态社区选择数据以非列表形式返回: {type(selection)}")
                print("-" * 40)
            
            print(f"\ngraphrag\\graphrag_demo_01\\main.py global_search response完整内容:")
            print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
            # 返回响应文本，这是最终的搜索结果
            return response
            
        except Exception as e:
            print(f"全局搜索执行失败: {str(e)}")
            return f"全局搜索执行失败: {str(e)}"
        finally:
            # 恢复原始模型参数配置，确保不影响后续操作
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 遍历保存的原始参数，恢复它们的值
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def local_search(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                          top_p: float = 0.9, max_tokens: int = 4096) -> str:
        """执行本地搜索
        本地搜索专注于回答关于特定实体或概念的详细问题。它首先识别查询中提到的实体，
        然后收集与这些实体直接相关的信息：它们的描述、关系、相关文本段落等。
        
        适用场景：
        - 当您想了解文档中特定人物、地点、组织或概念的详细信息
        - 适合"谁是XXX？"、"XXX的特性是什么？"、"XXX与YYY的关系如何？"这类问题
        - 当您需要关于特定主题的精确、有针对性的答案时
        
        与其他搜索方法的区别：
        - 比全局搜索更关注特定实体而非整体概述
        - 比基本搜索更精确，因为它利用了知识图谱结构
        - 比DRIFT搜索简单，不会进行多层递归探索
        
        参数:
            query: 用户查询文本
            community_level: 社区层级(0-N)，较高的值使用更细粒度的社区报告，0表示最顶层
            temperature: 控制回答的创造性/随机性(0-1)，越低越确定性
            top_p: 词汇采样的概率阈值(0-1)，控制回答的多样性
            max_tokens: 模型生成回答的最大标记数量
        
        返回:
            查询的文本回答
        """
        # 验证必要的数据是否存在，确保所有索引数据已加载
        # 本地搜索需要图谱数据(实体、关系)和原始文本数据(文本单元)
        if (self.config is None or self.entity_df is None or self.community_df is None or 
            self.report_df is None or self.text_unit_df is None or self.relationship_df is None or
            self.entity_df.empty or self.community_df.empty or self.report_df.empty or
            self.text_unit_df.empty or self.relationship_df.empty):
            # 如果缺少任何必要数据，抛出ValueError异常
            raise ValueError("缺少必要的数据，请先加载索引数据")
        
        # 保存原始配置的状态，以便在执行完搜索后恢复
        original_params = {}
        
        # 通过修改模型配置对象调整温度等参数
        if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
            # 获取默认聊天模型配置对象
            model_config = self.config.models['default_chat_model']
            # 记录原始参数值并设置新值
            for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                if hasattr(model_config, param):
                    # 保存原始值以便后续恢复
                    original_params[param] = getattr(model_config, param)
                    # 设置新值
                    setattr(model_config, param, value)
        
        try:
            # 准备收集上下文数据
            context_data = {}
            
            # 定义上下文数据回调函数
            def on_context(context):
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            # 使用API执行本地搜索
            user_query = query.strip()
            # 使用官方API进行查询，传入原始DataFrame而不是适配后的对象
            # api.local_search是GraphRAG库提供的用于执行本地搜索的API函数
            # 详细参数含义参见 \graphrag\graphrag\api\query.py的local_search函数文档
            # api官方实现参考：\graphrag\graphrag\cli\query.py
            # 本地搜索会从知识图谱中识别与查询相关的实体，并提取与这些实体相关的所有信息
            response = await api.local_search(
                config=self.config,  # 配置对象，包含模型设置、提示模板等
                entities=self.entity_df,  # 实体数据，包含名称、描述、类型等
                communities=self.community_df,  # 社区数据，表示实体如何组织成群组
                community_reports=self.report_df,  # 社区报告，包含每个实体组的摘要
                text_units=self.text_unit_df,  # 原始文本块，从中提取了实体和关系
                relationships=self.relationship_df,  # 实体间的关系数据
                covariates=self.covariate_df,  # 协变量数据框，可能为None，包含额外的声明/事实
                community_level=community_level,  # 要使用的社区层级
                response_type="详细的中文回答",  # 指定输出格式
                query=user_query,  # 用户查询文本
                callbacks=callbacks,
            )
            
            # 将上下文数据保存到成员变量，便于GUI访问
            self.context_data = context_data
            
            print("\ngraphrag\\graphrag_demo_01\\main.py local_search context_data完整内容:")
            print(json.dumps(context_data, indent=2, ensure_ascii=False, default=str))
            # context_data是一个字典，包含以下键值对：
            # - 'entities'：实体数据框(DataFrame)的字符串表示，包含实体ID、名称、描述等
            # - 'relationships'：关系数据框的字符串表示，包含源实体、目标实体、描述、权重等
            # - 'reports'：社区报告数据框的字符串表示，包含报告ID、标题、内容等
            # - 'sources'：原始文本来源数据框的字符串表示，包含文本ID和内容
            # - 'claims'：协变量/声明数据框的字符串表示，通常为空DataFrame
            # 本地搜索返回的context_data提供了与查询相关的结构化知识和原始文本
            # 用途：了解查询与哪些具体实体相关，这些实体之间有什么关系，以及支持这些关联的原始证据文本
            if isinstance(context_data, dict):
                print("\n本地搜索上下文数据分析:")
                print("-" * 40)
                # 确保所有数据类型都能正确处理
                for key in ['entities', 'relationships', 'reports', 'sources', 'claims']:
                    if key in context_data:
                        data = context_data[key]
                        # 添加标题文本，使字段名称更直观
                        title_map = {
                            'entities': '实体数据',
                            'relationships': '关系数据',
                            'reports': '社区报告数据',
                            'sources': '原始文本来源',
                            'claims': '协变量/声明数据'
                        }
                        
                        print(f"\n{title_map.get(key, key)}概览:")
                        # 处理字符串表示的DataFrame或其他数据类型
                        if isinstance(data, str):
                            print(f"{data[:500]}..." if len(data) > 500 else data)
                        elif isinstance(data, list):
                            print(f"包含 {len(data)} 条记录")
                            if len(data) > 0 and isinstance(data[0], dict):
                                for key_field in data[0].keys():
                                    if key_field not in ['id', 'in_context']:
                                        print(f"字段: {key_field}")
                        else:
                            print(f"数据类型: {type(data)}")
                            print(f"{str(data)[:500]}..." if len(str(data)) > 500 else str(data))
                
                print("-" * 40)
            
            print(f"\ngraphrag\\graphrag_demo_01\\main.py local_search response 完整内容:")
            print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
            # 返回响应文本，这是最终的搜索结果
            return response
        except Exception as e:
            # 捕获并处理搜索过程中的任何异常
            print(f"本地搜索执行失败: {str(e)}")
            return f"本地搜索执行失败: {str(e)}"
        finally:
            # 恢复原始模型参数配置，确保不影响后续操作
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 遍历保存的原始参数，恢复它们的值
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
                        
    async def drift_search(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                          top_p: float = 0.9, max_tokens: int = 4096) -> str:
        """执行DRIFT搜索 (Deep Recursive Information Finding Technique)
    
        DRIFT搜索是GraphRAG中最强大的搜索方法，它结合了全局和本地搜索的优点。
        它先从整体角度查看数据，找到相关的社区信息，然后递归深入探索相关实体及其关系，
        最后汇总所有发现以生成全面而深入的答案。
        
        适用场景：
        - 当您需要复杂问题的深入回答，特别是涉及多个相互关联的概念
        - 适合"XXX和YYY之间的关系如何？"、"该文档集合中存在什么隐藏的模式？"这类问题
        - 当您想从宏观视角到微观细节全面了解某个主题
        - 当您不确定问题具体涉及哪些实体，需要系统自动探索时
        
        与其他搜索方法的区别：
        - 比全局搜索更深入，能探索更多相关细节
        - 比本地搜索更全面，不局限于初始识别的实体
        - 比基本搜索更智能，利用了图结构并进行递归探索
        - 计算成本通常高于其他方法，但提供最全面的答案
        
        参数:
            query: 用户查询文本
            community_level: 社区层级(0-N)，较高的值使用更细粒度的社区报告
            temperature: 控制回答的创造性/随机性(0-1)，越低越确定性
            top_p: 词汇采样的概率阈值(0-1)，控制回答的多样性
            max_tokens: 模型生成回答的最大标记数量
        
        返回:
            查询的文本回答
        """
        # 验证必要的数据是否存在，确保所有索引数据已加载
        # DRIFT搜索与本地搜索需要相同的数据，但使用不同的算法处理
        if (self.config is None or self.entity_df is None or self.community_df is None or 
            self.report_df is None or self.text_unit_df is None or self.relationship_df is None or
            self.entity_df.empty or self.community_df.empty or self.report_df.empty or
            self.text_unit_df.empty or self.relationship_df.empty):
            # 如果缺少任何必要数据，抛出ValueError异常
            raise ValueError("缺少必要的数据，请先加载索引数据")
        
        # 保存原始配置的状态，以便在执行完搜索后恢复
        original_params = {}
        
        # 通过修改模型配置对象调整温度等参数
        if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
            # 获取默认聊天模型配置对象
            model_config = self.config.models['default_chat_model']
            # 记录原始参数值并设置新值
            for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                if hasattr(model_config, param):
                    # 保存原始值以便后续恢复
                    original_params[param] = getattr(model_config, param)
                    # 设置新值
                    setattr(model_config, param, value)
        
        try:
            # 准备收集上下文数据
            context_data = {}
            
            # 定义上下文数据回调函数
            def on_context(context):
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            user_query = query.strip()
            # 使用官方API进行查询，传入原始DataFrame而不是适配后的对象
            # api.drift_search是GraphRAG库提供的用于执行DRIFT搜索的API函数
            # 详细参数含义参见 \graphrag\graphrag\api\query.py的drift_search函数文档
            # api官方实现参考：\graphrag\graphrag\cli\query.py
            # DRIFT搜索会进行多轮递归探索，从社区报告开始，深入到实体和它们的关系
            response = await api.drift_search(
                config=self.config,  # 配置对象，包含模型设置、提示模板等
                entities=self.entity_df,  # 实体数据
                communities=self.community_df,  # 社区数据
                community_reports=self.report_df,  # 社区报告
                text_units=self.text_unit_df,  # 原始文本块
                relationships=self.relationship_df,  # 实体间的关系
                community_level=community_level,  # 使用的社区层级深度
                response_type="详细的中文回答",  # 指定输出格式
                query=user_query,  # 用户查询文本
                callbacks=callbacks,
            )
            
            # 将上下文数据保存到成员变量，便于GUI访问
            self.context_data = context_data
            
            print("\ngraphrag\\graphrag_demo_01\\main.py drift_search context_data完整内容:")
            print(json.dumps(context_data, indent=2, ensure_ascii=False, default=str))
            # context_data是一个字典，包含以下键值对：
            # - 'reports'：社区报告数据框的字符串表示，包含报告ID、标题、内容等
            # - 'entities'：实体相关数据，在此示例中为空DataFrame
            # - 'sources'：原始文本来源数据框的字符串表示，包含文本ID和内容
            # DRIFT搜索的context_data展示了完整的递归搜索过程，从初始引导(使用社区报告)到递归探索(追踪相关实体)再到最终综合
            # 用途：深入了解DRIFT搜索如何分解复杂问题，递归探索相关信息，最终综合生成全面而深入的答案
            if isinstance(context_data, dict):
                print("\nDRIFT搜索上下文数据分析:")
                print("-" * 40)
                # 确保所有数据类型都能正确处理
                for key in ['reports', 'entities', 'sources']:
                    if key in context_data:
                        data = context_data[key]
                        # 添加标题文本，使字段名称更直观
                        title_map = {
                            'reports': '社区报告数据',
                            'entities': '实体数据',
                            'sources': '原始文本来源'
                        }
                        print(f"\n{title_map.get(key, key)}概览:")
                        # 处理字符串表示的DataFrame或其他数据类型
                        if isinstance(data, str):
                            print(f"{data[:500]}..." if len(data) > 500 else data)
                        elif isinstance(data, list):
                            print(f"包含 {len(data)} 条记录")
                            if len(data) > 0 and isinstance(data[0], dict):
                                for key_field in data[0].keys():
                                    if key_field not in ['id', 'in_context']:
                                        print(f"字段: {key_field}")
                        else:
                            print(f"数据类型: {type(data)}")
                            print(f"{str(data)[:500]}..." if len(str(data)) > 500 else str(data))
                
                print("-" * 40)
            print(f"\ngraphrag\\graphrag_demo_01\\main.py drift_search response 完整内容:")
            print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
            # 返回响应文本，这是最终的搜索结果
            return response
        except Exception as e:
            # 捕获并处理搜索过程中的任何异常
            print(f"DRIFT搜索执行失败: {str(e)}")
            return f"DRIFT搜索执行失败: {str(e)}"
        finally:
            # 恢复原始模型参数配置，确保不影响后续操作
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 遍历保存的原始参数，恢复它们的值
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def basic_search(self, query: str, k: int = 5, temperature: float = 0.6, 
                          top_p: float = 0.9, max_tokens: int = 4096) -> str:
        """执行基本搜索
        基本搜索是GraphRAG中最简单的搜索方法，类似于传统的RAG系统。
        它仅使用文本相似度查找与查询最相关的文本块，不利用知识图谱结构。
        它提供快速但相对粗略的搜索结果，适合简单查询或作为基准比较。
        
        适用场景：
        - 当您只需要简单的文本匹配，不需要复杂的知识关联
        - 适合查找特定事实或直接在文本中提及的信息
        - 当您想快速获取结果，不需要复杂的上下文处理
        - 作为其他搜索方法的基准比较
        
        与其他搜索方法的区别：
        - 比其他方法简单快速，但不利用图结构
        - 仅依赖于文本相似度，不考虑实体和关系
        - 结果可能不如其他方法精确或全面
        - 计算成本最低，但答案质量也可能较低
        
        参数:
            query: 用户查询文本
            k: 检索的最相似文本块数量，控制考虑的上下文数量
            temperature: 控制回答的创造性/随机性(0-1)，越低越确定性
            top_p: 词汇采样的概率阈值(0-1)，控制回答的多样性
            max_tokens: 模型生成回答的最大标记数量
        
        返回:
            查询的文本回答
        """
        # 验证必要的数据是否存在
        # 基本搜索只需要文本单元(text_units)数据，这是所有方法中需求最少的
        if self.config is None or self.text_unit_df is None or self.text_unit_df.empty:
            # 如果缺少必要数据，抛出ValueError异常
            raise ValueError("缺少必要的数据，请先加载索引数据")
        
        # 保存原始配置的状态
        original_params = {}
        # 保存原始的basic_search.k参数，这是基本搜索特有的
        original_k = None
        # 通过修改模型配置对象调整温度等参数
        if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
            # 获取默认聊天模型配置对象
            model_config = self.config.models['default_chat_model']
            # 记录原始参数值并设置新值
            for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                if hasattr(model_config, param):
                    # 保存原始值以便后续恢复
                    original_params[param] = getattr(model_config, param)
                    # 设置新值
                    setattr(model_config, param, value)
        
        # 如果basic_search配置存在k参数，k控制从向量存储中检索的最相似文本块数量
        # 这个修改是必要的，因为basic_search API内部会使用config.basic_search.k值
        if hasattr(self.config, 'basic_search') and hasattr(self.config.basic_search, 'k'):
            # 保存原始k值以便后续恢复
            original_k = self.config.basic_search.k
            # 临时修改k值为用户指定的值，控制检索的文本单元数量
            self.config.basic_search.k = k
        
        try:
            # 准备收集上下文数据
            context_data = {}
            
            # 定义上下文数据回调函数
            def on_context(context):
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            # 使用API执行基本搜索
            user_query = query.strip()
            # 调用GraphRAG官方API执行基本搜索
            # api.basic_search是GraphRAG库提供的基本搜索功能,基本搜索只使用向量相似度检索文本块，类似于传统RAG系统
            # 详细参数含义参见 \graphrag\graphrag\api\query.py的basic_search函数文档
            # api官方实现参考：\graphrag\graphrag\cli\query.py
            response = await api.basic_search(
                config=self.config,  # 传递包含模型配置的配置对象，包含模型设置、提示模板等
                text_units=self.text_unit_df,  # 文本单元数据框，包含原始文本块
                query=user_query,  # 用户查询
                callbacks=callbacks,  # 回调函数列表
            )
            
            # 将上下文数据保存到成员变量，便于GUI访问
            self.context_data = context_data
            
            print(f"\ngraphrag\\graphrag_demo_01\\main.py basic_search(k={k}) context_data完整内容:")
            print(json.dumps(context_data, indent=2, ensure_ascii=False, default=str))
            # context_data是一个字典，包含'Sources'键（注意首字母大写），其值是检索到的最相关文本块
            # 基本搜索的context_data提供了查询匹配的原始文本片段，每个源文本都有唯一的source_id
            # 用途：基本搜索不使用任何图结构，只是简单地找到与查询最相似的文本
            if isinstance(context_data, dict):
                print("\n基本搜索上下文数据分析:")
                print("-" * 40)
                # 处理Sources数据
                if 'Sources' in context_data:
                    sources_data = context_data['Sources']
                    print("\n原始文本来源数据:")
                    # 处理字符串表示的DataFrame
                    if isinstance(sources_data, pd.DataFrame):
                        # 如果是实际的DataFrame对象
                        source_count = len(sources_data)
                        print(f"\n检索到 {source_count} 个最相关的文本片段")
                        # 显示前k个文本片段
                        for i, (_, source) in enumerate(sources_data.iterrows(), 1):
                            if i <= k:
                                source_id = source.get('source_id', '未知')
                                text = source.get('text', '无内容')
                                print(f"\n文本片段 {i}:")
                                print(f"  来源ID: {source_id}")
                                if len(str(text)) > 100:
                                    print(f"  内容: {str(text)[:100]}...")
                                else:
                                    print(f"  内容: {text}")
                
                print("-" * 40)
            # 返回响应文本，这是最终的搜索结果
            print(f"\ngraphrag\\graphrag_demo_01\\main.py basic_search response 完整内容:")
            print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
            return response
        except Exception as e:
            # 捕获并处理搜索过程中的任何异常
            print(f"基本搜索执行失败: {str(e)}")
            return f"基本搜索执行失败: {str(e)}"
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 遍历保存的原始参数，恢复它们的值
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
            
            # 恢复basic_search配置中的原始k值
            if original_k is not None and hasattr(self.config, 'basic_search') and hasattr(self.config.basic_search, 'k'):
                self.config.basic_search.k = original_k
    
    async def global_search_streaming(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                                     top_p: float = 0.9, max_tokens: int = 4096, 
                                     callback: callable = None) -> tuple[str, dict]:
        """执行流式全局搜索
        
        全局搜索是GraphRAG中的一种搜索方法，使用"Map-Reduce"模式在所有社区报告上执行搜索，提供数据集整体视图。
        与常规全局搜索不同，流式搜索会实时返回响应片段，适合需要即时反馈的场景。
        
        参数:
            query: 用户查询文本
            community_level: 社区层级(0-N)，0表示顶层社区(最宏观)，更高的值表示更细分的社区
            temperature: 控制回答的创造性(0-1)，值越低越确定性，值越高结果越多样化
            top_p: 控制采样词汇的概率阈值(0-1)，降低该值会使输出更加确定性
            max_tokens: 模型生成回答的最大标记数量，限制输出长度
            callback: 回调函数，用于处理每个响应片段，接收一个字符串参数
            
        返回:
            tuple: (完整响应文本, 上下文数据字典)
        """
        # 验证必要的数据是否存在
        if (self.config is None or self.entity_df is None or self.community_df is None or 
            self.report_df is None or self.entity_df.empty or self.community_df.empty or 
            self.report_df.empty):
            raise ValueError("缺少必要的数据，请先加载索引数据")
        
        # 保存原始配置状态
        original_params = {}
        if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
            model_config = self.config.models['default_chat_model']
            for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                if hasattr(model_config, param):
                    original_params[param] = getattr(model_config, param)
                    setattr(model_config, param, value)
        
        try:
            # 准备收集上下文数据
            full_response = ""
            context_data = {}
            
            # 定义上下文回调
            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context
            
            # 创建回调对象
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            # 使用流式API
            # user_query = query.strip() + " /no_think"
            user_query = query.strip()
            async for chunk in api.global_search_streaming(
                config=self.config,
                entities=self.entity_df,
                communities=self.community_df,
                community_reports=self.report_df,
                community_level=community_level,
                dynamic_community_selection=False,
                response_type="详细的中文回答",
                query=user_query,
                callbacks=callbacks,
            ):
                # 累积完整响应
                full_response += chunk
                
                # 如果提供了回调函数，调用它
                if callback and callable(callback):
                    callback(chunk)
                
                # 也可以在控制台打印实时响应，但这里我们将控制权交给调用者
                # print(chunk, end="", flush=True)
            
            # 在所有响应处理完毕后，打印上下文数据分析
            print("\ngraphrag\\graphrag_demo_01\\main.py global_search_streaming context_data 完整内容:")
            print(json.dumps(context_data, indent=2, ensure_ascii=False, default=str))
            
            # 将上下文数据保存到成员变量，便于GUI访问
            self.context_data = context_data
            
            # context_data是一个字典，只包含'reports'键，其值是包含社区报告的DataFrame
            # 全局搜索返回的context_data比较简洁，只包含用于生成答案的社区报告信息
            # 这些报告包含id、title、content和rank等字段，rank值表示报告的重要性评分
            # 用途：帮助理解全局搜索使用了哪些社区报告来生成答案，以及它们的相对重要性
            if 'reports' in context_data:
                # 检查reports的类型
                reports = context_data['reports']
                if isinstance(reports, pd.DataFrame):
                    # 如果是DataFrame，直接处理
                    print(f"使用了 {len(reports)} 个社区报告生成答案")
                    for i, (_, report) in enumerate(reports.iterrows(), 1):
                        # 获取报告的标题和内容
                        title = report.get('title', '未命名社区')
                        # 尝试获取summary字段，如果不存在则使用content
                        summary = report.get('summary', report.get('content', '无摘要'))
                        # 获取rank值(重要性评分)
                        rank = report.get('rank', '未知')  # 获取rank值
                        # 打印报告信息，包括序号、标题和重要性评分
                        print(f"报告 {i}: {title} (重要性评分: {rank})")
                        print(f"摘要: {summary[:500]}..." if len(str(summary)) > 500 else f"摘要: {summary}")
                        print("-" * 20)

                # 处理动态社区选择数据，这是可选功能，当dynamic_community_selection=True时才会出现
                if 'dynamic_selection' in context_data:
                    selection = context_data['dynamic_selection']
                    if isinstance(selection, list):
                        print(f"动态社区选择: 评估了 {len(selection)} 个社区")
                        print(f"选中了 {sum(1 for s in selection if isinstance(s, dict) and s.get('selected', False))} 个社区用于回答")
                    else:
                        print(f"动态社区选择数据以非列表形式返回: {type(selection)}")
                print("-" * 40)
            # 返回完整响应和上下文数据
            return full_response
            
        except Exception as e:
            print(f"流式全局搜索执行失败: {str(e)}")
            return f"流式全局搜索执行失败: {str(e)}", {}
        finally:
            # 恢复原始模型参数
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def local_search_streaming(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                                     top_p: float = 0.9, max_tokens: int = 4096, 
                                     callback: callable = None) -> tuple[str, dict]:
        """执行流式本地搜索
        
        本地搜索专注于回答关于特定实体或概念的详细问题。它首先识别查询中提到的实体，
        然后收集与这些实体直接相关的信息：它们的描述、关系、相关文本段落等。
        
        流式版本会实时返回响应片段，而不是等待整个响应生成完毕，适合需要即时反馈的场景。
        
        适用场景：
        - 当您想了解文档中特定人物、地点、组织或概念的详细信息
        - 适合"谁是XXX？"、"XXX的特性是什么？"、"XXX与YYY的关系如何？"这类问题
        - 当您需要关于特定主题的精确、有针对性的答案时
        - 当用户界面需要逐步显示生成内容，提高交互体验时
        
        与其他搜索方法的区别：
        - 比全局搜索更关注特定实体而非整体概述
        - 比基本搜索更精确，因为它利用了知识图谱结构
        - 比DRIFT搜索简单，不会进行多层递归探索
        - 比非流式版本能提供更好的用户体验，尤其是对长回答
        
        参数:
            query: 用户查询文本
            community_level: 社区层级(0-N)，较高的值使用更细粒度的社区报告，0表示最顶层
            temperature: 控制回答的创造性/随机性(0-1)，越低越确定性
            top_p: 词汇采样的概率阈值(0-1)，控制回答的多样性
            max_tokens: 模型生成回答的最大标记数量
            callback: 回调函数，用于处理每个响应片段，接收一个字符串参数
        
        返回:
            tuple: (完整响应文本, 上下文数据字典)
        """
        # 验证必要的数据是否存在，确保所有索引数据已加载
        # 本地搜索需要图谱数据(实体、关系)和原始文本数据(文本单元)
        if (self.config is None or self.entity_df is None or self.community_df is None or 
            self.report_df is None or self.text_unit_df is None or self.relationship_df is None or
            self.entity_df.empty or self.community_df.empty or self.report_df.empty or
            self.text_unit_df.empty or self.relationship_df.empty):
            # 如果缺少任何必要数据，抛出ValueError异常
            raise ValueError("缺少必要的数据，请先加载索引数据")
        
        # 保存原始配置状态
        original_params = {}
        # 通过修改模型配置对象调整温度等参数
        if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
            model_config = self.config.models['default_chat_model']
            for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                if hasattr(model_config, param):
                    original_params[param] = getattr(model_config, param)
                    setattr(model_config, param, value)
        
        try:
            # 用于存储完整响应
            full_response = ""
            # 用于存储上下文数据
            context_data = {}
            
            # 定义上下文回调函数，用于从API获取上下文数据
            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context
            
            # 创建回调对象并设置上下文回调
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            # 使用流式API执行本地搜索
            user_query = query.strip()
            async for chunk in api.local_search_streaming(
                config=self.config,  # 配置对象，包含模型设置、提示模板等
                entities=self.entity_df,  # 实体数据，包含名称、描述、类型等
                communities=self.community_df,  # 社区数据，表示实体如何组织成群组
                community_reports=self.report_df,  # 社区报告，包含每个实体组的摘要
                text_units=self.text_unit_df,  # 原始文本块，从中提取了实体和关系
                relationships=self.relationship_df,  # 实体间的关系数据
                covariates=self.covariate_df,  # 协变量数据框，可能为None，包含额外的声明/事实
                community_level=community_level,  # 要使用的社区层级
                response_type="详细的中文回答",  # 指定输出格式
                query=user_query,  # 用户查询文本
                callbacks=callbacks,  # 回调函数列表，用于收集上下文数据和其他事件处理
            ):
                # 累积完整响应
                full_response += chunk
                
                # 如果提供了回调函数，调用它处理每个文本片段
                if callback and callable(callback):
                    callback(chunk)
            
            # 流式生成完成后，打印上下文数据分析
            print("\ngraphrag\\graphrag_demo_01\\main.py local_search_streaming context_data 完整内容:")
            print(json.dumps(context_data, indent=2, ensure_ascii=False, default=str))
            
            # 将上下文数据保存到成员变量，便于GUI访问
            self.context_data = context_data
            
            # context_data是一个字典，包含以下键值对：
            # 'entities': 实体数据
            # 'relationships': 关系数据
            # 'reports': 社区报告数据
            # 'sources': 原始文本来源数据
            # 'claims': 协变量/声明数据
            if isinstance(context_data, dict):
                print("\n本地搜索上下文数据分析:")
                print("-" * 40)
                # 确保所有数据类型都能正确处理
                for key in ['entities', 'relationships', 'reports', 'sources', 'claims']:
                    if key in context_data:
                        data = context_data[key]
                        # 添加标题文本，使字段名称更直观
                        title_map = {
                            'entities': '实体数据',
                            'relationships': '关系数据',
                            'reports': '社区报告数据',
                            'sources': '原始文本来源',
                            'claims': '协变量/声明数据'
                        }
                        
                        print(f"\n{title_map.get(key, key)}概览:")
                        # 处理字符串表示的DataFrame或其他数据类型
                        if isinstance(data, str):
                            print(f"{data[:500]}..." if len(data) > 500 else data)
                        elif isinstance(data, list):
                            print(f"包含 {len(data)} 条记录")
                            if len(data) > 0 and isinstance(data[0], dict):
                                for key_field in data[0].keys():
                                    if key_field not in ['id', 'in_context']:
                                        print(f"字段: {key_field}")
                        else:
                            print(f"数据类型: {type(data)}")
                            print(f"{str(data)[:500]}..." if len(str(data)) > 500 else str(data))
                
                print("-" * 40)
            
            # 返回完整响应文本
            return full_response
            
        except Exception as e:
            # 捕获并处理搜索过程中的任何异常
            print(f"流式本地搜索执行失败: {str(e)}")
            return f"流式本地搜索执行失败: {str(e)}"
        finally:
            # 恢复原始模型参数配置，确保不影响后续操作
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 遍历保存的原始参数，恢复它们的值
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    async def basic_search_streaming(self, query: str, k: int = 5, temperature: float = 0.6, 
                               top_p: float = 0.9, max_tokens: int = 4096,
                               callback: callable = None) -> tuple[str, dict]:
        """执行流式基本搜索
        
        基本搜索是GraphRAG中最简单的搜索方法，类似于传统的RAG系统。
        它仅使用文本相似度查找与查询最相关的文本块，不利用知识图谱结构。
        
        流式版本会实时返回响应片段，而不是等待整个响应生成完毕，适合需要即时反馈的场景。
        
        适用场景：
        - 当您只需要简单的文本匹配，不需要复杂的知识关联
        - 适合查找特定事实或直接在文本中提及的信息
        - 当您想快速获取结果，不需要复杂的上下文处理
        - 当用户界面需要逐步显示生成内容，提高交互体验时
        
        与其他搜索方法的区别：
        - 比其他方法简单快速，但不利用图结构
        - 仅依赖于文本相似度，不考虑实体和关系
        - 结果可能不如其他方法精确或全面
        - 计算成本最低，但答案质量也可能较低
        
        参数:
            query: 用户查询文本
            k: 检索的最相似文本块数量，控制考虑的上下文数量
            temperature: 控制回答的创造性/随机性(0-1)，越低越确定性
            top_p: 词汇采样的概率阈值(0-1)，控制回答的多样性
            max_tokens: 模型生成回答的最大标记数量
            callback: 回调函数，用于处理每个响应片段，接收一个字符串参数
        
        返回:
            tuple: (完整响应文本, 上下文数据字典)
        """
        # 验证必要的数据是否存在
        if self.config is None or self.text_unit_df is None or self.text_unit_df.empty:
            raise ValueError("缺少必要的数据，请先加载索引数据")
        
        # 保存原始配置的状态
        original_params = {}
        # 保存原始的basic_search.k参数
        original_k = None
        
        # 通过修改模型配置对象调整温度等参数
        if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
            model_config = self.config.models['default_chat_model']
            for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                if hasattr(model_config, param):
                    original_params[param] = getattr(model_config, param)
                    setattr(model_config, param, value)
        
        # 如果basic_search配置存在k参数，暂时修改为用户指定的值
        if hasattr(self.config, 'basic_search') and hasattr(self.config.basic_search, 'k'):
            original_k = self.config.basic_search.k
            self.config.basic_search.k = k
        
        try:
            # 用于存储完整响应
            full_response = ""
            # 用于存储上下文数据
            context_data = {}
            
            # 定义上下文回调函数，用于从API获取上下文数据
            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context
            
            # 创建回调对象并设置上下文回调
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            # 使用流式API执行基本搜索
            user_query = query.strip()
            async for chunk in api.basic_search_streaming(
                config=self.config,      # 配置对象，包含模型设置、提示模板等
                text_units=self.text_unit_df,  # 文本单元数据框，包含原始文本块
                query=user_query,        # 用户查询
                callbacks=callbacks,     # 回调函数列表，用于收集上下文数据和其他事件处理
            ):
                # 累积完整响应
                full_response += chunk
                
                # 如果提供了回调函数，调用它处理每个文本片段
                if callback and callable(callback):
                    callback(chunk)
            
            # 流式生成完成后，打印上下文数据分析
            print("\ngraphrag\\graphrag_demo_01\\main.py basic_search_streaming context_data 完整内容:")
            print(json.dumps(context_data, indent=2, ensure_ascii=False, default=str))
            
            # 将上下文数据保存到成员变量，便于GUI访问
            self.context_data = context_data
            
            # 分析上下文数据，提供更直观的信息
            if isinstance(context_data, dict):
                print("\n基本搜索上下文数据分析:")
                print("-" * 40)
                # 处理Sources数据
                if 'Sources' in context_data:
                    sources_data = context_data['Sources']
                    print("\n原始文本来源数据:")
                    # 处理DataFrame
                    if isinstance(sources_data, pd.DataFrame):
                        source_count = len(sources_data)
                        print(f"\n检索到 {source_count} 个最相关的文本片段")
                        # 显示前k个文本片段
                        for i, (_, source) in enumerate(sources_data.iterrows(), 1):
                            if i <= k:
                                source_id = source.get('source_id', '未知')
                                text = source.get('text', '无内容')
                                print(f"\n文本片段 {i}:")
                                print(f"  来源ID: {source_id}")
                                if len(str(text)) > 100:
                                    print(f"  内容: {str(text)[:100]}...")
                                else:
                                    print(f"  内容: {text}")
                
                print("-" * 40)
            
            # 返回完整响应文本
            return full_response
            
        except Exception as e:
            print(f"流式基本搜索执行失败: {str(e)}")
            return f"流式基本搜索执行失败: {str(e)}"
        finally:
            # 恢复原始模型参数配置
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
            
            # 恢复basic_search配置中的原始k值
            if original_k is not None and hasattr(self.config, 'basic_search') and hasattr(self.config.basic_search, 'k'):
                self.config.basic_search.k = original_k
    
    async def drift_search_streaming(self, query: str, community_level: int = 0, temperature: float = 0.6, 
                                 top_p: float = 0.9, max_tokens: int = 4096,
                                 callback: callable = None) -> tuple[str, dict]:
        """执行流式DRIFT搜索 (Deep Recursive Information Finding Technique)
    
        DRIFT搜索是GraphRAG中最强大的搜索方法，它结合了全局和本地搜索的优点。
        它先从整体角度查看数据，找到相关的社区信息，然后递归深入探索相关实体及其关系，
        最后汇总所有发现以生成全面而深入的答案。
        
        流式版本会实时返回响应片段，而不是等待整个响应生成完毕，适合需要即时反馈的场景。
        
        适用场景：
        - 当您需要复杂问题的深入回答，特别是涉及多个相互关联的概念
        - 适合"XXX和YYY之间的关系如何？"、"该文档集合中存在什么隐藏的模式？"这类问题
        - 当您想从宏观视角到微观细节全面了解某个主题
        - 当您不确定问题具体涉及哪些实体，需要系统自动探索时
        - 当用户界面需要逐步显示生成内容，提高交互体验时
        
        与其他搜索方法的区别：
        - 比全局搜索更深入，能探索更多相关细节
        - 比本地搜索更全面，不局限于初始识别的实体
        - 比基本搜索更智能，利用了图结构并进行递归探索
        - 计算成本通常高于其他方法，但提供最全面的答案
        
        参数:
            query: 用户查询文本
            community_level: 社区层级(0-N)，较高的值使用更细粒度的社区报告
            temperature: 控制回答的创造性/随机性(0-1)，越低越确定性
            top_p: 词汇采样的概率阈值(0-1)，控制回答的多样性
            max_tokens: 模型生成回答的最大标记数量
            callback: 回调函数，用于处理每个响应片段，接收一个字符串参数
        
        返回:
            tuple: (完整响应文本, 上下文数据字典)
        """
        # 验证必要的数据是否存在，确保所有索引数据已加载
        # DRIFT搜索与本地搜索需要相同的数据，但使用不同的算法处理
        if (self.config is None or self.entity_df is None or self.community_df is None or 
            self.report_df is None or self.text_unit_df is None or self.relationship_df is None or
            self.entity_df.empty or self.community_df.empty or self.report_df.empty or
            self.text_unit_df.empty or self.relationship_df.empty):
            # 如果缺少任何必要数据，抛出ValueError异常
            raise ValueError("缺少必要的数据，请先加载索引数据")
        
        # 保存原始配置状态
        original_params = {}
        # 通过修改模型配置对象调整温度等参数
        if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
            model_config = self.config.models['default_chat_model']
            for param, value in [('temperature', temperature), ('top_p', top_p), ('max_tokens', max_tokens)]:
                if hasattr(model_config, param):
                    original_params[param] = getattr(model_config, param)
                    setattr(model_config, param, value)
        
        try:
            # 用于存储完整响应
            full_response = ""
            # 用于存储上下文数据
            context_data = {}
            
            # 定义上下文回调函数，用于从API获取上下文数据
            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context
            
            # 创建回调对象并设置上下文回调
            local_callbacks = NoopQueryCallbacks()
            local_callbacks.on_context = on_context
            callbacks = [local_callbacks]
            
            # 使用流式API执行DRIFT搜索
            user_query = query.strip()
            async for chunk in api.drift_search_streaming(
                config=self.config,  # 配置对象，包含模型设置、提示模板等
                entities=self.entity_df,  # 实体数据
                communities=self.community_df,  # 社区数据
                community_reports=self.report_df,  # 社区报告
                text_units=self.text_unit_df,  # 原始文本块
                relationships=self.relationship_df,  # 实体间的关系
                community_level=community_level,  # 使用的社区层级深度
                response_type="详细的中文回答",  # 指定输出格式
                query=user_query,  # 用户查询文本
                callbacks=callbacks,  # 回调函数列表，用于收集上下文数据和其他事件处理
            ):
                # 累积完整响应
                full_response += chunk
                
                # 如果提供了回调函数，调用它处理每个文本片段
                if callback and callable(callback):
                    callback(chunk)
            
            # 流式生成完成后，打印上下文数据分析
            print("\ngraphrag\\graphrag_demo_01\\main.py drift_search_streaming context_data 完整内容:")
            print(json.dumps(context_data, indent=2, ensure_ascii=False, default=str))
            
            # 将上下文数据保存到成员变量，便于GUI访问
            self.context_data = context_data
            
            # 分析上下文数据，提供更直观的信息
            if isinstance(context_data, dict):
                print("\nDRIFT搜索上下文数据分析:")
                print("-" * 40)
                # 确保所有数据类型都能正确处理
                for key in ['reports', 'entities', 'sources']:
                    if key in context_data:
                        data = context_data[key]
                        # 添加标题文本，使字段名称更直观
                        title_map = {
                            'reports': '社区报告数据',
                            'entities': '实体数据',
                            'sources': '原始文本来源'
                        }
                        print(f"\n{title_map.get(key, key)}概览:")
                        # 处理字符串表示的DataFrame或其他数据类型
                        if isinstance(data, str):
                            print(f"{data[:500]}..." if len(data) > 500 else data)
                        elif isinstance(data, list):
                            print(f"包含 {len(data)} 条记录")
                            if len(data) > 0 and isinstance(data[0], dict):
                                for key_field in data[0].keys():
                                    if key_field not in ['id', 'in_context']:
                                        print(f"字段: {key_field}")
                        else:
                            print(f"数据类型: {type(data)}")
                            print(f"{str(data)[:500]}..." if len(str(data)) > 500 else str(data))
                
                print("-" * 40)
            
            # 返回完整响应文本
            return full_response
            
        except Exception as e:
            # 捕获并处理搜索过程中的任何异常
            print(f"流式DRIFT搜索执行失败: {str(e)}")
            return f"流式DRIFT搜索执行失败: {str(e)}"
        finally:
            # 恢复原始模型参数配置，确保不影响后续操作
            if hasattr(self.config, 'models') and 'default_chat_model' in self.config.models:
                model_config = self.config.models['default_chat_model']
                # 遍历保存的原始参数，恢复它们的值
                for param, value in original_params.items():
                    if hasattr(model_config, param):
                        setattr(model_config, param, value)
    
    
    
async def main():
    """主函数"""
    interactive = GraphRAGInteractive()
    await interactive.start()


if __name__ == "__main__":
    asyncio.run(main()) 