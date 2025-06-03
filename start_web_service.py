#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG Web服务启动脚本
简化启动过程并提供配置选项
"""

import os
import sys
import uvicorn
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    print(f"✅ Python版本: {sys.version}")
    
    # 检查必要的文件
    required_files = ["app.py", "graphrag_service.py"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ 缺少必要文件: {file}")
            return False
    print("✅ 必要文件检查通过")
    
    # 检查依赖包
    try:
        import fastapi
        import uvicorn
        import httpx
        import graphrag
        print("✅ 核心依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("💡 请运行: pip install -r requirements_web.txt")
        return False
    
    return True

def setup_directories():
    """设置必要的目录"""
    print("📁 设置项目目录...")
    
    directories = ["input", "output", "cache", "logs", "prompts"]
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {dir_name}")
        else:
            print(f"📁 目录已存在: {dir_name}")

def check_input_files():
    """检查输入文件"""
    input_dir = Path("input")
    if not input_dir.exists():
        print("⚠️ input目录不存在")
        return False
    
    files = list(input_dir.glob("*"))
    if not files:
        print("⚠️ input目录为空")
        print("💡 请将要处理的文档文件放入input目录")
        return False
    
    print(f"✅ 发现 {len(files)} 个输入文件:")
    for file in files[:5]:  # 只显示前5个
        print(f"  - {file.name}")
    if len(files) > 5:
        print(f"  ... 还有 {len(files) - 5} 个文件")
    
    return True

def check_env_file():
    """检查环境配置文件"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env文件不存在")
        print("💡 如果使用API服务（如OpenAI），请创建.env文件配置API密钥")
        return False
    
    print("✅ 发现.env配置文件")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GraphRAG Web服务启动脚本")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="服务端口 (默认: 8000)")
    parser.add_argument("--reload", action="store_true", help="开启自动重载 (开发模式)")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数 (默认: 1)")
    parser.add_argument("--log-level", default="info", 
                       choices=["critical", "error", "warning", "info", "debug"],
                       help="日志级别 (默认: info)")
    parser.add_argument("--skip-checks", action="store_true", help="跳过环境检查")
    
    args = parser.parse_args()
    
    print("🚀 GraphRAG Web服务启动器")
    print("=" * 50)
    
    if not args.skip_checks:
        # 环境检查
        if not check_environment():
            print("❌ 环境检查失败，退出启动")
            sys.exit(1)
        
        # 设置目录
        setup_directories()
        
        # 检查输入文件（警告但不阻止启动）
        check_input_files()
        
        # 检查环境配置（警告但不阻止启动）
        check_env_file()
        
        print("✅ 环境检查完成")
    else:
        print("⚠️ 跳过环境检查")
    
    print("\n🌟 启动Web服务...")
    print(f"📍 地址: http://{args.host}:{args.port}")
    print(f"📚 API文档: http://{args.host}:{args.port}/docs")
    print(f"🔍 健康检查: http://{args.host}:{args.port}/health")
    print("💡 按 Ctrl+C 停止服务")
    print("-" * 50)
    
    try:
        # 启动服务
        uvicorn.run(
            "app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()