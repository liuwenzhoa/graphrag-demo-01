#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG GUI
用于GraphRAG的图形化界面
"""

import os
import sys
import asyncio
from pathlib import Path
import markdown
import re
import tiktoken

# 导入PyQt5库
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, 
    QSplitter, QFileDialog, QMessageBox, QTabWidget, QStatusBar, QInputDialog,
    QApplication, QListView, QDialog
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QTextCursor

# 导入qasync以支持Qt事件循环中的异步操作
import qasync

# 导入GraphRAG相关模块
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig

# 导入当前目录中的main.py模块
from graphrag_demo_01.main import GraphRAGInteractive

# 添加教育主题样式类
class EducationTheme:
    """教育主题样式定义，用于统一GUI的外观 - 专为教育类知识搜索应用设计"""
    
    # 主色调 - 温和专业的教育配色
    PRIMARY_COLOR = "#2563eb"  # 温和的教育蓝，专业可信
    PRIMARY_LIGHT = "#3b82f6"  # 浅一点的主色调
    PRIMARY_DARK = "#1d4ed8"   # 深一点的主色调
    SECONDARY_COLOR = "#10b981"  # 柔和的成功绿，表示成长和进步
    SECONDARY_LIGHT = "#34d399"  # 浅绿色
    ACCENT_COLOR = "#f59e0b"  # 温和的橙色，表示活力但不刺眼
    
    # 背景色系 - 清洁现代
    BACKGROUND_COLOR = "#fafbfc"  # 极浅的背景色，更纯净
    CONTENT_BG_COLOR = "#ffffff"  # 纯白内容区背景
    CARD_BG_COLOR = "#f8fafc"  # 卡片背景色，微微偏蓝
    HOVER_BG_COLOR = "#f1f5f9"  # 悬停背景色
    
    # 文本颜色 - 高对比度
    TEXT_COLOR = "#1f2937"  # 主文本颜色，深灰色确保可读性
    TEXT_SECONDARY_COLOR = "#4b5563"  # 次要文本颜色，中等灰色
    TEXT_LIGHT_COLOR = "#6b7280"  # 浅色文本，用于提示等
    TEXT_MUTED_COLOR = "#9ca3af"  # 静音文本颜色
    
    # 状态颜色
    SUCCESS_COLOR = "#10b981"  # 成功/正确
    SUCCESS_LIGHT = "#d1fae5"  # 浅成功色背景
    WARNING_COLOR = "#f59e0b"  # 警告
    WARNING_LIGHT = "#fef3c7"  # 浅警告色背景
    ERROR_COLOR = "#ef4444"  # 错误
    ERROR_LIGHT = "#fee2e2"  # 浅错误色背景
    INFO_COLOR = "#3b82f6"  # 信息
    INFO_LIGHT = "#dbeafe"  # 浅信息色背景
    
    # 边框和分割线
    BORDER_COLOR = "#e5e7eb"  # 主边框色
    BORDER_LIGHT_COLOR = "#f3f4f6"  # 浅边框色
    BORDER_FOCUS_COLOR = "#2563eb"  # 聚焦边框色
    DIVIDER_COLOR = "#e5e7eb"  # 分割线颜色
    
    # 字体系统
    FONT_FAMILY = "'Segoe UI', 'Microsoft YaHei UI', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', sans-serif"
    FONT_SIZE_SMALL = 12
    FONT_SIZE_NORMAL = 14
    FONT_SIZE_MEDIUM = 15
    FONT_SIZE_LARGE = 16
    FONT_SIZE_TITLE = 18
    
    # 圆角系统 - 统一的圆角设计
    BORDER_RADIUS_SMALL = "4px"
    BORDER_RADIUS_MEDIUM = "6px"
    BORDER_RADIUS_LARGE = "8px"
    BORDER_RADIUS_XL = "12px"
    
    # 间距系统 - 8px基础网格
    SPACING_XS = "4px"
    SPACING_SM = "8px"
    SPACING_MD = "12px"
    SPACING_LG = "16px"
    SPACING_XL = "20px"
    SPACING_XXL = "24px"
    
    # 内边距系统
    PADDING_XS = "4px"
    PADDING_SM = "8px"
    PADDING_MD = "12px"
    PADDING_LG = "16px"
    PADDING_XL = "20px"
    
    # 窗口和布局尺寸常量
    WINDOW_WIDTH = 1300
    WINDOW_HEIGHT = 1000
    
    # 布局间距和边距
    MAIN_LAYOUT_MARGIN = (24, 20, 24, 20)  # 左、上、右、下
    MAIN_LAYOUT_SPACING = 20
    TOP_LAYOUT_SPACING = 16
    PROJECT_CONTROL_SPACING = 12
    HELP_LAYOUT_SPACING = 12
    QUERY_LAYOUT_MARGIN = (24, 20, 24, 20)
    QUERY_LAYOUT_SPACING = 16
    TOKEN_LAYOUT_SPACING = 12
    SETTINGS_LAYOUT_SPACING = 16
    RESULT_LAYOUT_MARGIN = (24, 20, 24, 20)
    RESULT_LAYOUT_SPACING = 16
    
    # 组件尺寸
    TOKEN_WIDGET_HEIGHT = 40
    SETTINGS_WIDGET_HEIGHT = 44
    CONTROL_COMPONENT_HEIGHT = 36
    QUERY_BUTTON_HEIGHT = 36
    QUERY_INPUT_MIN_HEIGHT = 100
    QUERY_INPUT_MAX_HEIGHT = 140
    
    # 图标尺寸
    ICON_SIZE_SMALL = (16, 16)
    ICON_SIZE_NORMAL = (18, 18)
    ICON_SIZE_LARGE = (20, 20)
    
    # 分割器尺寸
    SPLITTER_SIZES = [250, 750]  # 查询区域250px，结果区域750px
    
    @classmethod
    def get_window_size(cls):
        """返回窗口尺寸"""
        return cls.WINDOW_WIDTH, cls.WINDOW_HEIGHT
    
    @classmethod
    def apply_main_layout(cls, layout):
        """应用主布局设置"""
        layout.setContentsMargins(*cls.MAIN_LAYOUT_MARGIN)
        layout.setSpacing(cls.MAIN_LAYOUT_SPACING)
    
    @classmethod
    def apply_top_layout(cls, layout):
        """应用顶部布局设置"""
        layout.setSpacing(cls.TOP_LAYOUT_SPACING)
    
    @classmethod
    def apply_project_control_layout(cls, layout):
        """应用项目控制布局设置"""
        layout.setSpacing(cls.PROJECT_CONTROL_SPACING)
    
    @classmethod
    def apply_help_layout(cls, layout):
        """应用帮助布局设置"""
        layout.setSpacing(cls.HELP_LAYOUT_SPACING)
    
    @classmethod
    def apply_query_layout(cls, layout):
        """应用查询布局设置"""
        layout.setContentsMargins(*cls.QUERY_LAYOUT_MARGIN)
        layout.setSpacing(cls.QUERY_LAYOUT_SPACING)
    
    @classmethod
    def apply_token_layout(cls, layout):
        """应用Token布局设置"""
        layout.setSpacing(cls.TOKEN_LAYOUT_SPACING)
    
    @classmethod
    def apply_settings_layout(cls, layout):
        """应用设置布局设置"""
        layout.setSpacing(cls.SETTINGS_LAYOUT_SPACING)
    
    @classmethod
    def apply_result_layout(cls, layout):
        """应用结果布局设置"""
        layout.setContentsMargins(*cls.RESULT_LAYOUT_MARGIN)
        layout.setSpacing(cls.RESULT_LAYOUT_SPACING)
    
    @classmethod
    def setup_widget_sizes(cls, widget_dict):
        """统一设置组件尺寸"""
        if 'token_widget' in widget_dict:
            widget_dict['token_widget'].setFixedHeight(cls.TOKEN_WIDGET_HEIGHT)
        
        if 'settings_widget' in widget_dict:
            widget_dict['settings_widget'].setFixedHeight(cls.SETTINGS_WIDGET_HEIGHT)
        
        if 'method_combo' in widget_dict:
            widget_dict['method_combo'].setFixedHeight(cls.CONTROL_COMPONENT_HEIGHT)
        
        if 'community_level' in widget_dict:
            widget_dict['community_level'].setFixedHeight(cls.CONTROL_COMPONENT_HEIGHT)
        
        if 'temperature' in widget_dict:
            widget_dict['temperature'].setFixedHeight(cls.CONTROL_COMPONENT_HEIGHT)
        
        if 'query_button' in widget_dict:
            widget_dict['query_button'].setFixedHeight(cls.QUERY_BUTTON_HEIGHT)
        
        if 'query_input' in widget_dict:
            widget_dict['query_input'].setMinimumHeight(cls.QUERY_INPUT_MIN_HEIGHT)
            # 移除最大高度限制，让输入框能够拉伸填充剩余空间
            # widget_dict['query_input'].setMaximumHeight(cls.QUERY_INPUT_MAX_HEIGHT)
    
    @classmethod
    def setup_icon_sizes(cls, widget_dict):
        """统一设置图标尺寸"""
        normal_size = QSize(*cls.ICON_SIZE_NORMAL)
        small_size = QSize(*cls.ICON_SIZE_SMALL)
        large_size = QSize(*cls.ICON_SIZE_LARGE)
        
        # 普通图标尺寸
        for widget_name in ['select_dir_button', 'load_config_button', 'help_button', 'method_info_button']:
            if widget_name in widget_dict:
                widget_dict[widget_name].setIconSize(normal_size)
        
        # 小图标尺寸
        for widget_name in ['query_button']:
            if widget_name in widget_dict:
                widget_dict[widget_name].setIconSize(small_size)
    
    @classmethod
    def get_application_style(cls):
        """返回应用程序全局样式表 - 教育类产品专用设计"""
        return f"""
        /* ===== 全局样式 ===== */
        QMainWindow, QDialog {{
            background-color: {cls.BACKGROUND_COLOR};
            font-family: {cls.FONT_FAMILY};
            font-size: {cls.FONT_SIZE_NORMAL}px;
            color: {cls.TEXT_COLOR};
        }}
        
        /* ===== 标签样式系统 ===== */
        QLabel {{
            color: {cls.TEXT_COLOR};
            font-family: {cls.FONT_FAMILY};
            font-size: {cls.FONT_SIZE_NORMAL}px;
            font-weight: 500;
            line-height: 1.5;
        }}
        
        /* 主标题样式 */
        QLabel[objectName="title_label"] {{
            font-family: {cls.FONT_FAMILY};
            font-size: 20px;
            font-weight: 700;
            color: {cls.PRIMARY_COLOR};
            margin-bottom: 0px;
            padding: {cls.PADDING_MD};
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 rgba(37, 99, 235, 0.05), 
                                       stop: 1 rgba(37, 99, 235, 0.02));
            border-radius: {cls.BORDER_RADIUS_LARGE};
        }}
        
        /* 描述标签样式 */
        QLabel[objectName="description_label"] {{
            font-family: {cls.FONT_FAMILY};
            font-size: {cls.FONT_SIZE_SMALL}px;
            color: {cls.TEXT_SECONDARY_COLOR};
            margin-bottom: 0px;
            padding: 0 {cls.PADDING_XS};
            font-weight: 400;
        }}
        
        /* 区域标题样式 */
        QLabel[objectName="query_title"], QLabel[objectName="result_title"] {{
            height: 30px;
            min-height: 30px;
            font-family: {cls.FONT_FAMILY};
            font-size: {cls.FONT_SIZE_NORMAL}px;
            font-weight: 600;
            color: {cls.PRIMARY_COLOR};
            margin: 0 0 0 0;
            padding: 0 0 0 0;
            border-bottom: 2px solid {cls.BORDER_LIGHT_COLOR};
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 rgba(37, 99, 235, 0.03), 
                                       stop: 1 transparent);
        }}
        
        /* Token统计标签样式 */
        QLabel[objectName="token_input_label"], QLabel[objectName="token_output_label"] {{
            color: {cls.TEXT_SECONDARY_COLOR}; 
            font-size: {cls.FONT_SIZE_SMALL}px;
            font-weight: 500;
            padding: 2px 4px;
            background-color: {cls.CARD_BG_COLOR};
            border: 1px solid {cls.BORDER_LIGHT_COLOR};
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
            min-height: 24px;
        }}
        
        /* Token数值标签样式 */
        QLabel[objectName="input_token_value"] {{
            color: {cls.PRIMARY_COLOR}; 
            font-size: {cls.FONT_SIZE_SMALL}px;
            font-weight: 600;
            padding: 2px 4px;
            background-color: {cls.INFO_LIGHT};
            border: 1px solid rgba(37, 99, 235, 0.2);
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
            min-height: 24px;
        }}
        
        QLabel[objectName="output_token_value"] {{
            color: {cls.SECONDARY_COLOR}; 
            font-size: {cls.FONT_SIZE_SMALL}px;
            font-weight: 600;
            padding: 2px 4px;
            background-color: {cls.SUCCESS_LIGHT};
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
            min-height: 24px;
        }}
        
        /* 设置标签样式 */
        QLabel[objectName="method_label"], QLabel[objectName="community_label"], QLabel[objectName="temp_label"] {{
            height: 30px;
            font-size: {cls.FONT_SIZE_SMALL}px; 
            font-weight: 500;
            color: {cls.TEXT_COLOR};
            min-height: 30px;
            padding: 8px 0;
        }}
        
        /* ===== 按钮样式系统 ===== */
        QPushButton {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.PRIMARY_COLOR}, 
                                       stop: 1 {cls.PRIMARY_DARK});
            color: white;
            border: none;
            border-radius: {cls.BORDER_RADIUS_LARGE};
            padding: {cls.PADDING_XS} {cls.PADDING_SM};
            font-family: {cls.FONT_FAMILY};
            font-size: {cls.FONT_SIZE_SMALL}px;
            font-weight: 600;
            min-height: 20px;
        }}
        
        QPushButton:hover {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.PRIMARY_LIGHT}, 
                                       stop: 1 {cls.PRIMARY_COLOR});
        }}
        
        QPushButton:pressed {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.PRIMARY_DARK}, 
                                       stop: 1 #1e40af);
        }}
        
        QPushButton:focus {{
            outline: none;
        }}
        
        QPushButton:disabled {{
            background: {cls.BORDER_COLOR};
            color: {cls.TEXT_MUTED_COLOR};
        }}
        
        /* 次要按钮样式 */
        QPushButton[secondary="true"] {{
            background: {cls.CONTENT_BG_COLOR};
            color: {cls.TEXT_COLOR};
            border: 2px solid {cls.BORDER_COLOR};
            font-weight: 500;
        }}
        
        QPushButton[secondary="true"]:hover {{
            background: {cls.HOVER_BG_COLOR};
            border-color: {cls.PRIMARY_COLOR};
            color: {cls.PRIMARY_COLOR};
        }}
        
        QPushButton[secondary="true"]:pressed {{
            background: {cls.CARD_BG_COLOR};
            border-color: {cls.PRIMARY_DARK};
            color: {cls.PRIMARY_DARK};
        }}
        
        /* 成功按钮样式 */
        QPushButton[success="true"] {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.SECONDARY_COLOR}, 
                                       stop: 1 #059669);
            color: white;
        }}
        
        QPushButton[success="true"]:hover {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.SECONDARY_LIGHT}, 
                                       stop: 1 {cls.SECONDARY_COLOR});
        }}
        
        /* 查询按钮特殊样式 */
        QPushButton[objectName="query_button"] {{
            min-width: 120px;
            font-size: {cls.FONT_SIZE_SMALL}px;
            font-weight: 600;
            height: 36px;
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
        }}
        
        /* ===== 输入框样式系统 ===== */
        QTextEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {cls.CONTENT_BG_COLOR};
            border: 2px solid {cls.BORDER_COLOR};
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
            padding: {cls.PADDING_XS};
            color: {cls.TEXT_COLOR};
            font-family: {cls.FONT_FAMILY};
            font-size: {cls.FONT_SIZE_NORMAL}px;
            selection-background-color: {cls.INFO_LIGHT};
            min-height: 16px;
            line-height: 1.5;
        }}
        
         QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {cls.BORDER_FOCUS_COLOR};
            background-color: {cls.CONTENT_BG_COLOR};
        }}
        
        QTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {cls.PRIMARY_LIGHT};
            background-color: {cls.CONTENT_BG_COLOR};
        }}
        
        /* 流式文本编辑器样式 */
        QTextEdit[objectName="stream_text_edit"] {{
            background-color: {cls.CONTENT_BG_COLOR};
            border: 2px solid {cls.BORDER_COLOR};
            border-radius: 0 0 {cls.BORDER_RADIUS_MEDIUM} {cls.BORDER_RADIUS_MEDIUM};
            padding: {cls.PADDING_XS};
            selection-background-color: {cls.INFO_LIGHT};
            font-size: {cls.FONT_SIZE_NORMAL}px;
            line-height: 1.6;
            color: {cls.TEXT_COLOR};
        }}
        
        /* 查询输入框特殊样式 */
        QTextEdit[objectName="query_input"] {{
            background-color: {cls.CONTENT_BG_COLOR};
            border: 2px solid {cls.BORDER_COLOR};
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
            padding: {cls.PADDING_XS};
            selection-background-color: {cls.INFO_LIGHT};
            font-size: {cls.FONT_SIZE_NORMAL}px;
            line-height: 1.5;
            color: {cls.TEXT_COLOR};
            min-height: 80px;
            /* 移除max-height限制，让输入框能够拉伸 */
        }}
        
        QTextEdit[objectName="query_input"]:focus {{
            border: 2px solid {cls.BORDER_FOCUS_COLOR};
            background-color: {cls.CONTENT_BG_COLOR};
        }}
        
        QTextEdit[objectName="query_input"]:hover {{
            border-color: {cls.PRIMARY_LIGHT};
            background-color: {cls.CONTENT_BG_COLOR};
        }}
        
        /* ===== 下拉框样式系统 ===== */
        QComboBox {{
            background-color: {cls.CONTENT_BG_COLOR};
            border: 2px solid {cls.BORDER_COLOR};
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
            padding: {cls.PADDING_MD} {cls.PADDING_LG};
            min-width: 120px;
            min-height: 16px;
            color: {cls.TEXT_COLOR};
            font-family: {cls.FONT_FAMILY};
            font-size: {cls.FONT_SIZE_NORMAL}px;
            font-weight: 500;
        }}
        
        QComboBox:hover {{
            border-color: {cls.PRIMARY_LIGHT};
            background-color: {cls.HOVER_BG_COLOR};
        }}
        
        QComboBox:focus {{
            border: 2px solid {cls.BORDER_FOCUS_COLOR};
        }}
        
        /* 方法选择下拉框特殊样式 */
        QComboBox[objectName="method_combo"] {{
            min-width: 160px;
            max-width: 180px;
            font-size: {cls.FONT_SIZE_SMALL}px;
            height: 36px;
            padding: 0 12px;
            border: 2px solid {cls.BORDER_COLOR};
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
        }}
        
        /* 数值输入框特殊样式 */
        QSpinBox[objectName="community_level"], QDoubleSpinBox[objectName="temperature"] {{
            min-width: 70px;
            max-width: 90px;
            font-size: {cls.FONT_SIZE_SMALL}px;
            height: 36px;
            padding: 0 8px;
            border: 2px solid {cls.BORDER_COLOR};
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
        }}
        
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: right;
            width: 30px;
            border-left: 1px solid {cls.BORDER_COLOR};
            border-top-right-radius: {cls.BORDER_RADIUS_MEDIUM};
            border-bottom-right-radius: {cls.BORDER_RADIUS_MEDIUM};
            background: {cls.CARD_BG_COLOR};
        }}
        
        QComboBox::down-arrow {{
            width: 12px;
            height: 12px;
            border: 2px solid {cls.TEXT_SECONDARY_COLOR};
            border-top: none;
            border-right: none;
            margin-top: -2px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {cls.CONTENT_BG_COLOR};
            border: 1px solid {cls.BORDER_COLOR};
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
            selection-background-color: {cls.PRIMARY_COLOR};
            selection-color: white;
            padding: 4px;
            font-size: {cls.FONT_SIZE_NORMAL}px;
        }}
        
        QComboBox QAbstractItemView::item {{
            min-height: 36px;
            padding: 8px {cls.PADDING_MD};
            border-radius: {cls.BORDER_RADIUS_SMALL};
            margin: 2px;
        }}
        
        QComboBox QAbstractItemView::item:hover {{
            background-color: {cls.HOVER_BG_COLOR};
        }}
        
        /* ===== 容器样式系统 ===== */
        QWidget[objectName="query_widget"], QWidget[objectName="result_widget"] {{
            background-color: {cls.CONTENT_BG_COLOR};
            border: 1px solid {cls.BORDER_COLOR};
            border-radius: {cls.BORDER_RADIUS_XL};
        }}
        
        /* ===== 选项卡样式系统 ===== */
        QTabWidget::pane {{
            border: none;
            background-color: {cls.CONTENT_BG_COLOR};
            border-radius: {cls.BORDER_RADIUS_MEDIUM};
            margin-top: -2px;
        }}
        
        QTabBar::tab {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.CARD_BG_COLOR}, 
                                       stop: 1 {cls.BORDER_LIGHT_COLOR});
            color: {cls.TEXT_SECONDARY_COLOR};
            border: 1px solid {cls.BORDER_COLOR};
            border-bottom: none;
            border-top-left-radius: {cls.BORDER_RADIUS_MEDIUM};
            border-top-right-radius: {cls.BORDER_RADIUS_MEDIUM};
            padding: {cls.PADDING_MD} {cls.PADDING_LG};
            margin-right: 2px;
            font-family: {cls.FONT_FAMILY};
            font-weight: 500;
            min-width: 80px;
        }}
        
        QTabBar::tab:selected {{
            background: {cls.CONTENT_BG_COLOR};
            color: {cls.PRIMARY_COLOR};
            border-bottom: 2px solid {cls.PRIMARY_COLOR};
            font-weight: 600;
        }}
        
        QTabBar::tab:hover:!selected {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.HOVER_BG_COLOR}, 
                                       stop: 1 {cls.CARD_BG_COLOR});
            color: {cls.TEXT_COLOR};
        }}
        
        /* ===== 状态栏样式 ===== */
        QStatusBar {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.CONTENT_BG_COLOR}, 
                                       stop: 1 {cls.CARD_BG_COLOR});
            color: {cls.TEXT_SECONDARY_COLOR};
            font-family: {cls.FONT_FAMILY};
            border-top: 1px solid {cls.BORDER_COLOR};
            padding: 8px {cls.PADDING_LG};
            font-size: {cls.FONT_SIZE_SMALL}px;
            font-weight: 500;
        }}
        
        /* ===== 滚动条样式系统 ===== */
        QScrollBar:vertical {{
            border: none;
            background: {cls.CARD_BG_COLOR};
            width: 12px;
            border-radius: 6px;
            margin: 0;
        }}
        
        QScrollBar::handle:vertical {{
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 {cls.BORDER_COLOR}, 
                                       stop: 1 {cls.TEXT_LIGHT_COLOR});
            min-height: 30px;
            border-radius: 6px;
            margin: 2px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 {cls.TEXT_LIGHT_COLOR}, 
                                       stop: 1 {cls.TEXT_SECONDARY_COLOR});
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            border: none;
            background: {cls.CARD_BG_COLOR};
            height: 12px;
            border-radius: 6px;
            margin: 0;
        }}
        
        QScrollBar::handle:horizontal {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.BORDER_COLOR}, 
                                       stop: 1 {cls.TEXT_LIGHT_COLOR});
            min-width: 30px;
            border-radius: 6px;
            margin: 2px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 {cls.TEXT_LIGHT_COLOR}, 
                                       stop: 1 {cls.TEXT_SECONDARY_COLOR});
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        
        /* ===== 分割器样式 ===== */
        QSplitter::handle {{
            background-color: {cls.BORDER_COLOR};
            border-radius: 2px;
        }}
        
        QSplitter::handle:horizontal {{
            width: 4px;
            margin: 0 2px;
        }}
        
        QSplitter::handle:vertical {{
            height: 4px;
            margin: 2px 0;
        }}
        
        QSplitter::handle:hover {{
            background-color: {cls.PRIMARY_COLOR};
        }}
        """

class StreamTextEdit(QTextEdit):
    """
    支持流式输出和富文本格式的文本编辑器组件
    增强markdown支持，改进文本显示效果
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.document().setMaximumBlockCount(10000)  # 防止内存占用过大
        
        # 设置更适合阅读的字体和大小
        font = QFont(EducationTheme.FONT_FAMILY.replace("'", ""), EducationTheme.FONT_SIZE_NORMAL)
        self.setFont(font)
        
        # 设置objectName以应用统一样式
        self.setObjectName("stream_text_edit")
    
    def append_text(self, text):
        """追加文本到编辑器，并自动滚动到底部"""
        self.moveCursor(QTextCursor.End)
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.End)
        
    def clear_and_set_text(self, text):
        """清空编辑器并设置新文本"""
        self.clear()
        self.setPlainText(text)
        self.moveCursor(QTextCursor.End)

    def set_html_content(self, html_content):
        """设置HTML格式的内容"""
        self.clear()
        self.setHtml(html_content)
        self.moveCursor(QTextCursor.End)
    
    def set_markdown_content(self, markdown_text):
        """
        将Markdown文本转换为HTML并显示
        增强实现，支持更完善的Markdown语法和更好的视觉效果
        """
        if not markdown_text:
            self.clear()
            return
            
        # 确保输入是字符串
        if not isinstance(markdown_text, str):
            markdown_text = str(markdown_text)
        
        # 预处理Markdown文本，处理特殊格式
        markdown_text = self._preprocess_markdown(markdown_text)
        
        # 转换为HTML，支持更多扩展
        html = markdown.markdown(
            markdown_text, 
            extensions=[
                'tables', 
                'fenced_code', 
                'codehilite',
                'toc',
                'nl2br'
            ],
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'use_pygments': False
                }
            }
        )
        
        # 包装在样式中
        styled_html = f"""
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            body {{ 
                font-family: {EducationTheme.FONT_FAMILY}; 
                line-height: 1.7;
                color: {EducationTheme.TEXT_COLOR};
                padding: 20px;
                margin: 0;
                background-color: {EducationTheme.CONTENT_BG_COLOR};
                max-width: 100%;
                word-wrap: break-word;
            }}
            h1 {{ 
                color: {EducationTheme.PRIMARY_COLOR}; 
                font-size: 26px; 
                margin-bottom: 24px;
                margin-top: 28px;
                border-bottom: 3px solid {EducationTheme.BORDER_LIGHT_COLOR};
                padding-bottom: 10px;
                font-weight: 700;
                line-height: 1.3;
            }}
            h2 {{ 
                color: {EducationTheme.PRIMARY_COLOR}; 
                font-size: 22px; 
                margin-top: 28px;
                margin-bottom: 18px;
                font-weight: 600;
                line-height: 1.3;
            }}
            h3 {{ 
                color: {EducationTheme.TEXT_COLOR}; 
                font-size: 19px; 
                margin-top: 24px;
                margin-bottom: 14px;
                font-weight: 600;
                line-height: 1.3;
            }}
            h4 {{ 
                color: {EducationTheme.TEXT_COLOR}; 
                font-size: 17px; 
                margin-top: 20px;
                margin-bottom: 12px;
                font-weight: 600;
                line-height: 1.3;
            }}
            p {{ 
                margin-top: 14px;
                margin-bottom: 14px; 
                text-align: justify;
                line-height: 1.7;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }}
            ul, ol {{ 
                margin-top: 10px;
                margin-bottom: 14px;
                padding-left: 32px;
            }}
            li {{ 
                margin-bottom: 8px;
                margin-top: 8px;
                line-height: 1.6;
                word-wrap: break-word;
            }}
            li p {{
                margin: 4px 0;
            }}
            blockquote {{
                margin: 20px 0;
                padding: 16px 20px;
                background-color: {EducationTheme.CARD_BG_COLOR};
                border-left: 5px solid {EducationTheme.PRIMARY_COLOR};
                color: {EducationTheme.TEXT_SECONDARY_COLOR};
                border-radius: {EducationTheme.BORDER_RADIUS_MEDIUM};
                font-style: italic;
                position: relative;
            }}
            blockquote::before {{
                content: '"';
                font-size: 48px;
                color: {EducationTheme.PRIMARY_COLOR};
                position: absolute;
                left: 10px;
                top: -5px;
                opacity: 0.3;
            }}
            code {{ 
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                background-color: {EducationTheme.CARD_BG_COLOR};
                padding: 4px 8px;
                border-radius: {EducationTheme.BORDER_RADIUS_SMALL};
                font-size: 90%;
                color: {EducationTheme.ACCENT_COLOR};
                font-weight: 500;
                border: 1px solid {EducationTheme.BORDER_COLOR};
            }}
            pre {{ 
                background-color: {EducationTheme.CARD_BG_COLOR};
                border-radius: {EducationTheme.BORDER_RADIUS_MEDIUM};
                padding: 20px;
                overflow-x: auto;
                margin: 20px 0;
                border: 1px solid {EducationTheme.BORDER_COLOR};
                position: relative;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
                color: {EducationTheme.TEXT_COLOR};
                font-size: 90%;
                white-space: pre;
                border: none;
            }}
            .highlight {{
                background-color: {EducationTheme.CARD_BG_COLOR};
                border-radius: {EducationTheme.BORDER_RADIUS_MEDIUM};
                padding: 16px;
                margin: 16px 0;
                border: 1px solid {EducationTheme.BORDER_COLOR};
            }}
            b, strong {{ 
                font-weight: 600; 
                color: {EducationTheme.TEXT_COLOR};
            }}
            i, em {{ 
                font-style: italic; 
                color: {EducationTheme.TEXT_SECONDARY_COLOR};
            }}
            hr {{
                margin: 32px 0;
                border: 0;
                height: 2px;
                background: linear-gradient(to right, {EducationTheme.PRIMARY_COLOR}, transparent);
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                background-color: {EducationTheme.CONTENT_BG_COLOR};
                border-radius: {EducationTheme.BORDER_RADIUS_MEDIUM};
                overflow: hidden;
            }}
            th, td {{
                padding: 14px 16px;
                text-align: left;
                border: 1px solid {EducationTheme.BORDER_COLOR};
                word-wrap: break-word;
                max-width: 300px;
            }}
            th {{
                background-color: {EducationTheme.PRIMARY_COLOR};
                color: white;
                font-weight: 600;
                font-size: 95%;
            }}
            tr:nth-child(even) {{
                background-color: {EducationTheme.CARD_BG_COLOR};
            }}
            tr:hover {{
                background-color: {EducationTheme.BORDER_LIGHT_COLOR};
                transition: background-color 0.2s ease;
            }}
            /* 链接样式 */
            a {{
                color: {EducationTheme.PRIMARY_COLOR};
                text-decoration: none;
                font-weight: 500;
                border-bottom: 1px solid transparent;
                transition: all 0.2s ease;
            }}
            a:hover {{
                border-bottom: 1px solid {EducationTheme.PRIMARY_COLOR};
                color: {EducationTheme.SECONDARY_COLOR};
            }}
            /* 特殊内容样式 */
            .answer-section {{
                background-color: {EducationTheme.CARD_BG_COLOR};
                border-left: 4px solid {EducationTheme.SECONDARY_COLOR};
                padding: 16px 20px;
                margin: 20px 0;
                border-radius: 0 {EducationTheme.BORDER_RADIUS_MEDIUM} {EducationTheme.BORDER_RADIUS_MEDIUM} 0;
            }}
            
            .source-reference {{
                background-color: {EducationTheme.BORDER_LIGHT_COLOR};
                padding: 8px 12px;
                border-radius: {EducationTheme.BORDER_RADIUS_SMALL};
                font-size: 90%;
                color: {EducationTheme.TEXT_SECONDARY_COLOR};
                margin: 8px 0;
                display: inline-block;
                border-left: 3px solid {EducationTheme.PRIMARY_COLOR};
                font-weight: 500;
            }}
            
            /* 修复行间距和段落间距问题 */
            p + p {{ margin-top: 18px; }}
            ul p, ol p {{ margin: 0; }}
            p + ul, p + ol {{ margin-top: 14px; }}
            ul + p, ol + p {{ margin-top: 14px; }}
            
            /* 响应式设计 */
            @media (max-width: 768px) {{
                body {{ padding: 12px; }}
                h1 {{ font-size: 22px; }}
                h2 {{ font-size: 19px; }}
                h3 {{ font-size: 17px; }}
                table {{ font-size: 90%; }}
                th, td {{ padding: 10px 12px; }}
            }}
        </style>
        </head>
        <body>
        {html}
        </body>
        </html>
        """
        
        self.clear()
        self.setHtml(styled_html)
        self.moveCursor(QTextCursor.End)

    def _preprocess_markdown(self, markdown_text):
        """预处理Markdown文本，处理特殊格式"""
        # 处理GraphRAG特有的格式
        lines = markdown_text.split('\n')
        processed_lines = []
        in_answer_section = False
        
        for line in lines:
            # 处理引用格式 [Data: Sources (n)] 或 [Data: Sources (n); Sources (m)]
            if '[Data:' in line and ']' in line:
                # 使用正则表达式匹配所有的引用格式
                # 匹配 [Data: Sources (n)] 或 [Data: Sources (n); Sources (m)] 等格式
                citation_pattern = r'\[Data:\s*([^\]]+)\]'
                
                def replace_citation(match):
                    citation_content = match.group(1)
                    # 将引用转换为特殊样式的span
                    return f'<span class="source-reference">📚 数据来源: {citation_content}</span>'
                
                # 替换所有匹配的引用
                processed_line = re.sub(citation_pattern, replace_citation, line)
                processed_lines.append(processed_line)
            # 处理答案部分标记
            elif line.strip().startswith('## ') and ('答案' in line or 'Answer' in line or '回答' in line):
                if in_answer_section:
                    processed_lines.append('</div>')  # 关闭之前的答案部分
                processed_lines.append('<div class="answer-section">')
                processed_lines.append(line)
                in_answer_section = True
            # 处理长文本换行
            elif len(line) > 120 and not line.startswith('#') and not line.startswith('```') and not line.startswith('|'):
                # 对于很长的行，在适当位置添加换行
                words = line.split(' ')
                current_line = ''
                for word in words:
                    if len(current_line + ' ' + word) > 120:
                        if current_line:  # 确保不是空行
                            processed_lines.append(current_line)
                        current_line = word
                    else:
                        current_line += ' ' + word if current_line else word
                if current_line:
                    processed_lines.append(current_line)
            # 处理重复内容（DRIFT搜索结果中可能出现的问题）
            elif line.strip() and len(processed_lines) > 0:
                # 检查是否与前面的行重复
                if line.strip() not in [prev_line.strip() for prev_line in processed_lines[-3:]]:
                    processed_lines.append(line)
                # 如果重复，跳过这一行
            else:
                processed_lines.append(line)
        
        # 关闭答案部分标记（如果还在答案部分中）
        if in_answer_section:
            processed_lines.append('</div>')
        
        return '\n'.join(processed_lines)

class TokenCounter:
    """
    Token计数器类，用于计算输入和输出的token数量
    """
    
    def __init__(self):
        # 使用cl100k_base编码器，适用于最新的OpenAI模型
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            # 如果获取失败，使用默认编码器
            self.encoder = None
            print("无法初始化tiktoken编码器，token计数将不可用")
        
        self.input_tokens = 0
        self.output_tokens = 0
    
    def count_tokens(self, text):
        """计算文本的token数量"""
        if self.encoder is None:
            return 0
        
        try:
            tokens = self.encoder.encode(text)
            return len(tokens)
        except:
            return 0
    
    def update_input_tokens(self, text):
        """更新输入token计数"""
        self.input_tokens = self.count_tokens(text)
        return self.input_tokens
    
    def update_output_tokens(self, text):
        """更新输出token计数"""
        self.output_tokens = self.count_tokens(text)
        return self.output_tokens
    
    def reset(self):
        """重置计数器"""
        self.input_tokens = 0
        self.output_tokens = 0

class GraphRAGGUI(QMainWindow):
    """
    GraphRAG图形用户界面主窗口类
    提供访问GraphRAG所有功能的图形界面
    """
    
    def __init__(self):
        """初始化GraphRAGGUI对象"""
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("Qwen3 + GraphRAG 知识搜索助手")
        width, height = EducationTheme.get_window_size()
        self.resize(width, height)
        
        # 应用教育主题样式
        app = QApplication.instance()
        if app:
            app.setStyleSheet(EducationTheme.get_application_style())
        
        # 初始化成员变量
        self.project_dir = None
        self.graphrag = GraphRAGInteractive()
        self.token_counter = TokenCounter()
        self.query_running = False
        self.context_data = {}
        
        # 初始化用户界面
        self.setup_ui()
        
        # 连接信号和槽
        self.connect_signals()
        
        # 显示欢迎消息
        self.log_text.append_text("欢迎使用GraphRAG知识搜索助手！\n")
        self.log_text.append_text("请选择项目目录并加载配置。\n")
    
    def setup_ui(self):
        """设置UI组件和布局，采用教育主题风格"""
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        EducationTheme.apply_main_layout(main_layout)
        
        # 创建标题标签
        title_label = QLabel("Qwen3GraphRAG 学情检索助手")
        title_label.setObjectName("title_label")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建描述标签
        description_label = QLabel("基于Qwen3的知识图谱智能问答系统")
        description_label.setObjectName("description_label")
        description_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(description_label)
        
        # 创建顶部工具栏
        top_layout = QHBoxLayout()
        EducationTheme.apply_top_layout(top_layout)
        
        # 项目控制区域
        project_control_layout = QHBoxLayout()
        EducationTheme.apply_project_control_layout(project_control_layout)
        
        # 创建项目目录选择按钮
        self.select_dir_button = QPushButton("选择目录")
        self.select_dir_button.setIcon(self.style().standardIcon(self.style().SP_DialogOpenButton))
        self.select_dir_button.setToolTip("选择学情检索项目的根目录")
        project_control_layout.addWidget(self.select_dir_button)
        
        # 创建加载配置按钮
        self.load_config_button = QPushButton("加载配置")
        self.load_config_button.setEnabled(False)
        self.load_config_button.setIcon(self.style().standardIcon(self.style().SP_DialogApplyButton))
        self.load_config_button.setToolTip("加载所选目录中的学情检索配置")
        project_control_layout.addWidget(self.load_config_button)
        
        top_layout.addLayout(project_control_layout)
        
        # 添加弹性空间
        top_layout.addStretch(1)
        
        # 帮助按钮区域
        help_layout = QHBoxLayout()
        EducationTheme.apply_help_layout(help_layout)
        
        # 创建帮助按钮
        self.help_button = QPushButton("帮助")
        self.help_button.setProperty("secondary", "true")
        self.help_button.setIcon(self.style().standardIcon(self.style().SP_MessageBoxQuestion))
        self.help_button.setToolTip("显示帮助信息")
        help_layout.addWidget(self.help_button)
        
        # 创建查询方法说明按钮
        self.method_info_button = QPushButton("方法说明")
        self.method_info_button.setProperty("secondary", "true")
        self.method_info_button.setIcon(self.style().standardIcon(self.style().SP_MessageBoxInformation))
        self.method_info_button.setToolTip("显示不同方法的说明")
        help_layout.addWidget(self.method_info_button)
        
        top_layout.addLayout(help_layout)
        
        # 添加顶部布局
        main_layout.addLayout(top_layout)
        
        # 创建分割器，分隔上下两部分
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter, 1)
        
        # 创建查询区域
        query_widget = QWidget()
        query_widget.setObjectName("query_widget")
        query_layout = QVBoxLayout(query_widget)
        EducationTheme.apply_query_layout(query_layout)
        query_layout.setSpacing(8)  # 固定间距
        
        # 查询区域标题
        query_title = QLabel("提问区域")
        query_title.setObjectName("query_title")
        query_layout.addWidget(query_title, 0)  # 不拉伸
        
        # 输入框 - 给予充足的高度
        self.query_input = QTextEdit()
        self.query_input.setObjectName("query_input")  # 设置objectName以应用特殊样式
        self.query_input.setPlaceholderText("请输入您的问题，例如：'文档中最重要的主题是什么？'")
        # 输入框占用所有可用的垂直空间
        query_layout.addWidget(self.query_input, 1)  # 拉伸因子为1，垂直拉伸占用剩余空间
        
        # Token统计行 - 紧凑的水平布局，固定高度
        token_widget = QWidget()
        token_widget_layout = QHBoxLayout(token_widget)
        token_widget_layout.setContentsMargins(0, 0, 0, 0)
        EducationTheme.apply_token_layout(token_widget_layout)
        token_widget_layout.setAlignment(Qt.AlignVCenter)  # 垂直居中对齐
        
        token_input_label = QLabel("输入Token:")
        token_input_label.setObjectName("token_input_label")
        token_widget_layout.addWidget(token_input_label)
        
        self.input_token_label = QLabel("0")
        self.input_token_label.setObjectName("input_token_value")
        token_widget_layout.addWidget(self.input_token_label)
        
        token_widget_layout.addSpacing(8)
        
        token_output_label = QLabel("输出Token:")
        token_output_label.setObjectName("token_output_label")
        token_widget_layout.addWidget(token_output_label)
        
        self.output_token_label = QLabel("0")
        self.output_token_label.setObjectName("output_token_value")
        token_widget_layout.addWidget(self.output_token_label)
        
        token_widget_layout.addStretch()  # 推到左侧
        query_layout.addWidget(token_widget, 0)  # 不垂直拉伸
        
        # 设置控制行 - 所有设置项水平排列，固定高度
        settings_widget = QWidget()
        settings_layout = QHBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        EducationTheme.apply_settings_layout(settings_layout)
        settings_layout.setAlignment(Qt.AlignVCenter)  # 垂直居中对齐
        
        # 查询方法
        method_label = QLabel("查询方法:")
        method_label.setObjectName("method_label")
        settings_layout.addWidget(method_label)
        
        self.method_combo = QComboBox()
        self.method_combo.setObjectName("method_combo")
        self.method_combo.setView(QListView())
        self.method_combo.addItems([
            "全局搜索", "流式全局搜索", 
            "本地搜索", "流式本地搜索", 
            "DRIFT搜索", "流式DRIFT搜索", 
            "基本搜索", "流式基本搜索"
        ])
        self.method_combo.setMaxVisibleItems(8)
        self.method_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        settings_layout.addWidget(self.method_combo)
        
        settings_layout.addSpacing(8)
        
        # 社区级别
        community_label = QLabel("社区级别:")
        community_label.setObjectName("community_label")
        settings_layout.addWidget(community_label)
        
        self.community_level = QSpinBox()
        self.community_level.setObjectName("community_level")
        self.community_level.setMinimum(0)
        self.community_level.setMaximum(10)
        self.community_level.setValue(0)
        self.community_level.setToolTip("控制使用的社区层级深度，0表示顶层社区")
        settings_layout.addWidget(self.community_level)
        
        settings_layout.addSpacing(8)
        
        # 温度
        temp_label = QLabel("温度:")
        temp_label.setObjectName("temp_label")
        settings_layout.addWidget(temp_label)
        
        self.temperature = QDoubleSpinBox()
        self.temperature.setObjectName("temperature")
        self.temperature.setMinimum(0.0)
        self.temperature.setMaximum(2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.6)
        self.temperature.setToolTip("控制回答的创造性/确定性，越低越确定")
        settings_layout.addWidget(self.temperature)
        
        # 弹性空间，将执行按钮推到右侧
        settings_layout.addStretch()
        
        # 执行查询按钮
        self.query_button = QPushButton("执行查询")
        self.query_button.setObjectName("query_button")
        self.query_button.setEnabled(False)
        self.query_button.setProperty("success", "true")
        self.query_button.setIcon(self.style().standardIcon(self.style().SP_CommandLink))
        self.query_button.setToolTip("开始执行查询")
        settings_layout.addWidget(self.query_button)
        
        query_layout.addWidget(settings_widget, 0)  # 不垂直拉伸
        
        # 添加查询区域到分割器
        splitter.addWidget(query_widget)
        
        # 创建结果显示区域
        result_widget = QWidget()
        result_widget.setObjectName("result_widget")
        result_layout = QVBoxLayout(result_widget)
        EducationTheme.apply_result_layout(result_layout)
        
        # 结果区域标题
        result_title = QLabel("结果区域")
        result_title.setObjectName("result_title")
        result_layout.addWidget(result_title)
        
        # 创建选项卡区域
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        
        # 结果选项卡
        self.result_text = StreamTextEdit()
        self.tabs.addTab(self.result_text, "查询结果")
        self.tabs.setTabIcon(0, self.style().standardIcon(self.style().SP_FileDialogInfoView))
        
        # 上下文信息选项卡
        self.context_text = StreamTextEdit()
        self.tabs.addTab(self.context_text, "上下文信息")
        self.tabs.setTabIcon(1, self.style().standardIcon(self.style().SP_FileDialogContentsView))
        
        # 日志选项卡
        self.log_text = StreamTextEdit()
        self.tabs.addTab(self.log_text, "日志")
        self.tabs.setTabIcon(2, self.style().standardIcon(self.style().SP_FileDialogListView))
        
        # 设置图标大小
        icon_size = QSize(*EducationTheme.ICON_SIZE_NORMAL)
        self.tabs.setIconSize(icon_size)
        
        # 添加选项卡到结果布局
        result_layout.addWidget(self.tabs)
        
        # 添加结果区域到分割器
        splitter.addWidget(result_widget)
        
        # 设置分割器的初始大小比例，给结果区域更多空间
        splitter.setSizes(EducationTheme.SPLITTER_SIZES)
        
        # 设置分割器的拉伸因子：两个区域都可以拉伸
        splitter.setStretchFactor(0, 0)  # 查询区域可以调整但不主动拉伸
        splitter.setStretchFactor(1, 1)  # 结果区域主要拉伸区域
        
        # 设置分割器手柄的样式，使其不可拖拽
        # splitter.handle(1).setEnabled(False)  # 禁用分割器手柄
        
        # 创建状态栏
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("准备就绪")
        
        # 统一设置组件尺寸和图标
        widget_dict = {
            'token_widget': token_widget,
            'settings_widget': settings_widget,
            'method_combo': self.method_combo,
            'community_level': self.community_level,
            'temperature': self.temperature,
            'query_button': self.query_button,
            'query_input': self.query_input,
            'select_dir_button': self.select_dir_button,
            'load_config_button': self.load_config_button,
            'help_button': self.help_button,
            'method_info_button': self.method_info_button
        }
        
        # 应用组件尺寸设置
        # EducationTheme.setup_widget_sizes(widget_dict)
        
        # 应用图标尺寸设置
        EducationTheme.setup_icon_sizes(widget_dict)
    
    def connect_signals(self):
        """连接信号和槽"""
        # 按钮点击信号
        self.select_dir_button.clicked.connect(self.select_project_directory)
        self.load_config_button.clicked.connect(self.load_config)
        self.query_button.clicked.connect(self.execute_query)
        self.help_button.clicked.connect(self.show_help)
        self.method_info_button.clicked.connect(self.show_method_info)
        
        # 查询文本变化时更新token计数
        self.query_input.textChanged.connect(self.update_input_tokens)
    
    def select_project_directory(self):
        """选择项目目录"""
        # 打开文件对话框选择目录
        directory = QFileDialog.getExistingDirectory(self, "选择GraphRAG项目目录")
        if directory:
            self.project_dir = directory
            self.statusBar().showMessage(f"已选择项目目录: {directory}")
            self.log_text.append_text(f"已选择项目目录: {directory}\n")
            
            # 更新UI状态
            self.load_config_button.setEnabled(True)
    
    def load_config(self):
        """加载GraphRAG配置"""
        # 检查是否选择了项目目录
        if not self.project_dir:
            QMessageBox.warning(self, "错误", "请先选择项目目录")
            return
        
        # 更新状态
        self.statusBar().showMessage("正在加载配置...")
        self.log_text.append_text(f"开始加载配置...\n")
        
        # 使用asyncio创建任务执行异步操作
        asyncio.create_task(self._async_load_config(self.project_dir))
    
    async def _async_load_config(self, project_dir):
        """异步加载配置"""
        try:
            # 设置项目目录
            self.graphrag.project_dir = project_dir
            self.graphrag.output_dir = os.path.join(project_dir, "output")
            
            # 加载环境变量（如果存在）
            env_path = os.path.join(project_dir, ".env")
            if os.path.exists(env_path):
                from dotenv import load_dotenv
                load_dotenv(env_path)
                self.log_text.append_text(f"已加载环境变量: {env_path}\n")
            
            # 直接创建默认配置，不尝试读取settings.yaml
            self.log_text.append_text("创建GraphRAG配置...\n")
            self._create_config(project_dir)
            
            self.log_text.append_text("配置加载成功\n")
            self.statusBar().showMessage("配置加载成功")
            
            # 检查是否有索引数据
            entities_path = os.path.join(self.graphrag.output_dir, "entities.parquet")
            if os.path.exists(entities_path):
                self.log_text.append_text("正在加载索引数据...\n")
                self.graphrag._load_index_data()
                self.log_text.append_text("索引数据加载成功\n")
                self.query_button.setEnabled(True)
            else:
                self.log_text.append_text("警告: 未找到索引数据，请确保已经构建了索引\n")
                self.statusBar().showMessage("未找到索引数据")
                
        except Exception as e:
            self.log_text.append_text(f"加载配置时出错: {str(e)}\n")
            import traceback
            traceback_str = traceback.format_exc()
            self.log_text.append_text(f"错误详情:\n{traceback_str}\n")
            self.statusBar().showMessage("加载配置失败")
            QMessageBox.critical(self, "错误", f"加载配置时出错: {str(e)}")
    
    def _create_config(self, project_dir):
        """创建GraphRAG配置对象
        
        参考main.py中的_create_config方法，自动创建完整的配置对象，无需用户交互
        """
        self.log_text.append_text("创建GraphRAG配置...\n")
        
        # 获取模型配置
        chat_model_config, embedding_model_config = self._create_model_configs()
        
        # 创建配置字典
        config_data = {
            # 模型配置
            "models": {
                "default_chat_model": chat_model_config,
                "default_embedding_model": embedding_model_config
            },
            
            # 输入配置
            "input": {
                "type": "file",
                "file_type": "text",
                "base_dir": os.path.join(project_dir, "input"),
            },
            
            # 分块配置
            "chunks": {
                "size": 1200,
                "overlap": 100,
                "strategy": "tokens",
            },
            
            # 输出配置
            "output": {
                "type": "file",
                "base_dir": os.path.join(project_dir, "output"),
            },
            
            # 更新索引输出配置
            "update_index_output": {
                "type": "file",
                "base_dir": os.path.join(project_dir, "output"),
            },
            
            # 缓存配置
            "cache": {
                "type": "file",
                "base_dir": os.path.join(project_dir, "cache"),
            },
            
            # 报告配置
            "reporting": {
                "type": "file",
                "base_dir": os.path.join(project_dir, "logs"),
            },
            
            # 向量存储配置
            "vector_store": {
                "default_vector_store": {
                    "type": "lancedb",
                    "db_uri": os.path.join(project_dir, "output", "lancedb"),
                    "container_name": "default",
                    "overwrite": True,
                }
            },
            
            # 工作流配置
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
            
            # 文本嵌入配置
            "embed_text": {
                "enabled": True,
                "model_id": "default_embedding_model",
                "vector_store_id": "default_vector_store",
                "batch_size": 10,
                "batch_max_tokens": 8000,
            },
            
            # 实体图谱提取配置
            "extract_graph": {
                "model_id": "default_chat_model",
                "prompt": "prompts/extract_graph.txt",
                "entity_types": ["organization", "person", "geo", "event"],
                "max_gleanings": 2,
            },
            
            # 描述摘要配置
            "summarize_descriptions": {
                "model_id": "default_chat_model",
                "prompt": "prompts/summarize_descriptions.txt",
                "max_length": 1000,
                "max_input_length": 12000,
            },
            
            # NLP图谱提取配置
            "extract_graph_nlp": {
                "normalize_edge_weights": True,
                "text_analyzer": {
                    "extractor_type": "regex_english",
                }
            },
            
            # 社区聚类配置
            "cluster_graph": {
                "max_cluster_size": 10,
                "use_lcc": True,
                "seed": 42,
            },
            
            # 协变量提取配置
            "extract_claims": {
                "enabled": False,
                "model_id": "default_chat_model",
                "prompt": "prompts/extract_claims.txt",
                "description": "提取与时间相关的事实声明、决策和行动",
                "max_gleanings": 3,
            },
            
            # 社区报告生成配置
            "community_reports": {
                "model_id": "default_chat_model",
                "graph_prompt": "prompts/community_report_graph.txt",
                "text_prompt": "prompts/community_report_text.txt",
                "max_length": 4000,
                "max_input_length": 8000,
            },
            
            # 图嵌入配置
            "embed_graph": {
                "enabled": False,
            },
            
            # UMAP配置
            "umap": {
                "enabled": False,
            },
            
            # 图谱快照配置
            "snapshots": {
                "graphml": False,
                "embeddings": False,
            },
            
            # 本地搜索配置
            "local_search": {
                "chat_model_id": "default_chat_model",
                "embedding_model_id": "default_embedding_model",
                "prompt": "prompts/local_search_system_prompt.txt",
            },
            
            # 全局搜索配置
            "global_search": {
                "chat_model_id": "default_chat_model",
                "map_prompt": "prompts/global_search_map_system_prompt.txt",
                "reduce_prompt": "prompts/global_search_reduce_system_prompt.txt",
                "knowledge_prompt": "prompts/global_search_knowledge_system_prompt.txt",
            },
            
            # DRIFT搜索配置
            "drift_search": {
                "chat_model_id": "default_chat_model",
                "embedding_model_id": "default_embedding_model",
                "prompt": "prompts/drift_search_system_prompt.txt",
                "reduce_prompt": "prompts/drift_reduce_prompt.txt",
                "drift_k_followups": 3, # 检索的顶级全局结果数，控制搜索广度(默认 20)
                "primer_folds": 2, # 搜索启动的折数(默认 3)
                "n_depth": 1, # 要采取的漂移搜索步骤数，控制搜索深度(默认 3)
                "local_search_max_data_tokens": 8000, # 本地搜索的最大上下文大小，以标记为单位 (text-embedding-v3 模型限制8192)
            },
            
            # 基本搜索配置
            "basic_search": {
                "chat_model_id": "default_chat_model",
                "embedding_model_id": "default_embedding_model",
                "prompt": "prompts/basic_search_system_prompt.txt",
            },
        }
        
        # 创建GraphRAG配置对象
        try:
            self.graphrag.config = create_graphrag_config(config_data, root_dir=project_dir)
            self.log_text.append_text("已创建完整的GraphRAG配置对象\n")
            
            # 更新提示词路径
            self._update_prompt_paths(project_dir)
            
        except Exception as e:
            self.log_text.append_text(f"创建配置对象时出错: {str(e)}\n")
            import traceback
            traceback_str = traceback.format_exc()
            self.log_text.append_text(f"错误详情:\n{traceback_str}\n")
            raise
    
    def _create_model_configs(self):
        """创建默认的模型配置对象，自动从环境变量获取API密钥"""
        # 从环境变量获取API密钥
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "sk-********")
        deepseek_api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        aliyun_api_key = os.environ.get("ALIYUN_API_KEY", "sk-********")
        aliyun_api_base = os.environ.get("ALIYUN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 配置聊天模型
        chat_config = LanguageModelConfig(
            type=ModelType.OpenAIChat,
            api_base=deepseek_api_base,
            auth_type="api_key",
            api_key=deepseek_api_key,
            model="qwen3-235b-a22b",
            encoding_model="cl100k_base",
            model_supports_json=True,
            async_mode="threaded",
            max_retries=2,
            parallelization_num_threads=50,
            parallelization_stagger=0.3,
        )
        
        # 配置嵌入模型
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
        
        self.log_text.append_text("已创建模型配置\n")
        return chat_config, embedding_config
    
    def _update_prompt_paths(self, project_dir):
        """更新配置中的提示词路径，直接使用已有的提示词文件
        
        参考main.py中的_update_prompt_paths_in_config方法
        """
        self.log_text.append_text("检查提示词文件路径...\n")
        
        # 获取prompts目录路径
        prompts_dir = Path(project_dir) / "prompts"
        
        # 检查prompts目录是否存在
        if not prompts_dir.exists():
            self.log_text.append_text(f"警告: 提示词目录不存在: {prompts_dir}\n")
            return
        
        # 检查常用的提示词文件是否存在
        expected_prompts = [
            "extract_graph.txt",
            "summarize_descriptions.txt",
            "extract_claims.txt",
            "community_report_graph.txt",
            "community_report_text.txt",
            "drift_search_system_prompt.txt",
            "drift_reduce_prompt.txt",
            "global_search_map_system_prompt.txt",
            "global_search_reduce_system_prompt.txt",
            "global_search_knowledge_system_prompt.txt",
            "local_search_system_prompt.txt",
            "basic_search_system_prompt.txt",
            "question_gen_system_prompt.txt",
        ]
        
        missing_prompts = []
        for prompt_file in expected_prompts:
            if not (prompts_dir / prompt_file).exists():
                missing_prompts.append(prompt_file)
        
        if missing_prompts:
            self.log_text.append_text(f"警告: 以下提示词文件不存在:\n")
            for prompt in missing_prompts:
                self.log_text.append_text(f"- {prompt}\n")
        else:
            self.log_text.append_text("所有提示词文件都存在\n")
        
        # 更新GraphRAG配置中的提示词路径
        if not hasattr(self.graphrag, "config") or self.graphrag.config is None:
            self.log_text.append_text("警告: 配置对象未初始化，无法更新提示词路径\n")
            return
        
        # 更新索引相关提示词路径
        if hasattr(self.graphrag.config, "extract_graph"):
            self.graphrag.config.extract_graph.prompt = "prompts/extract_graph.txt"
        
        if hasattr(self.graphrag.config, "summarize_descriptions"):
            self.graphrag.config.summarize_descriptions.prompt = "prompts/summarize_descriptions.txt"
        
        if hasattr(self.graphrag.config, "extract_claims"):
            self.graphrag.config.extract_claims.prompt = "prompts/extract_claims.txt"
        
        if hasattr(self.graphrag.config, "community_reports"):
            self.graphrag.config.community_reports.graph_prompt = "prompts/community_report_graph.txt"
            self.graphrag.config.community_reports.text_prompt = "prompts/community_report_text.txt"
        
        # 更新查询相关提示词路径
        if hasattr(self.graphrag.config, "local_search"):
            self.graphrag.config.local_search.prompt = "prompts/local_search_system_prompt.txt"
        
        if hasattr(self.graphrag.config, "global_search"):
            self.graphrag.config.global_search.map_prompt = "prompts/global_search_map_system_prompt.txt"
            self.graphrag.config.global_search.reduce_prompt = "prompts/global_search_reduce_system_prompt.txt"
            self.graphrag.config.global_search.knowledge_prompt = "prompts/global_search_knowledge_system_prompt.txt"
        
        if hasattr(self.graphrag.config, "drift_search"):
            self.graphrag.config.drift_search.prompt = "prompts/drift_search_system_prompt.txt"
            self.graphrag.config.drift_search.reduce_prompt = "prompts/drift_reduce_prompt.txt"
        
        if hasattr(self.graphrag.config, "basic_search"):
            self.graphrag.config.basic_search.prompt = "prompts/basic_search_system_prompt.txt"
        
        self.log_text.append_text("已更新配置中的提示词路径\n")
    
    def execute_query(self):
        """执行查询"""
        # 检查是否已经有查询正在运行
        if self.query_running:
            QMessageBox.warning(self, "警告", "查询正在进行中，请等待完成")
            return
            
        # 获取查询文本
        query_text = self.query_input.toPlainText().strip()
        if not query_text:
            QMessageBox.warning(self, "错误", "请输入查询问题")
            return
        
        # 获取查询参数
        method_index = self.method_combo.currentIndex()
        community_level = self.community_level.value()
        temperature = self.temperature.value()
        
        # 清空结果区域
        self.result_text.clear()
        self.context_text.clear()
        
        # 更新状态
        self.statusBar().showMessage("正在执行查询...")
        self.query_button.setEnabled(False)
        self.query_running = True
        
        # 根据方法选择执行相应的查询
        asyncio.create_task(
            self._async_execute_query(
                query_text, method_index, community_level, temperature
            )
        )
    
    async def _async_execute_query(self, query, method_index, community_level, temperature):
        """异步执行查询"""
        try:
            # 记录开始时间
            start_time = asyncio.get_event_loop().time()
            self.log_text.append_text(f"开始执行查询: {query}\n")
            self.log_text.append_text(f"使用查询方法: {self.method_combo.currentText()}\n")
            self.log_text.append_text(f"社区级别: {community_level}, 温度: {temperature}\n")
            
            # 更新状态栏
            self.statusBar().showMessage("查询正在执行中...")
            
            # 定义流式输出回调函数
            def handle_streaming_output(chunk):
                """处理流式输出回调"""
                self.result_text.append_text(chunk)
                # 更新token统计
                self.update_output_tokens(self.result_text.toPlainText())
            
            # 根据方法索引选择查询函数
            if method_index == 0:  # 全局搜索
                self.log_text.append_text("执行全局搜索...\n")
                self.statusBar().showMessage("正在执行全局搜索...")
                result = await self.graphrag.global_search(
                    query, community_level, temperature
                )
                # 使用markdown渲染结果
                self.result_text.set_markdown_content(result)

                # 更新token统计
                self.update_output_tokens(result)
                
            elif method_index == 1:  # 流式全局搜索
                self.log_text.append_text("执行流式全局搜索...\n")
                self.statusBar().showMessage("正在执行流式全局搜索...")
                # 清空结果文本区域
                self.result_text.clear()
                # 执行流式搜索
                await self.graphrag.global_search_streaming(
                    query, community_level, temperature, 
                    callback=handle_streaming_output
                )
                
            elif method_index == 2:  # 本地搜索
                self.log_text.append_text("执行本地搜索...\n")
                self.statusBar().showMessage("正在执行本地搜索...")
                result_data = await self.graphrag.local_search(
                    query, community_level, temperature
                )

                # 处理本地搜索返回结果
                if isinstance(result_data, tuple) and len(result_data) > 0:
                    # 如果是元组，第一个元素是Markdown文本
                    result = result_data[0]
                else:
                    # 如果不是元组，则整个结果就是Markdown文本
                    result = result_data
                    
                # 使用markdown渲染结果
                self.result_text.set_markdown_content(result)
                # 更新token统计
                self.update_output_tokens(result)
                
            elif method_index == 3:  # 流式本地搜索
                self.log_text.append_text("执行流式本地搜索...\n")
                self.statusBar().showMessage("正在执行流式本地搜索...")
                # 清空结果文本区域
                self.result_text.clear()
                # 执行流式搜索
                await self.graphrag.local_search_streaming(
                    query, community_level, temperature,
                    callback=handle_streaming_output
                )
                
            elif method_index == 4:  # DRIFT搜索
                self.log_text.append_text("执行DRIFT搜索...\n")
                self.statusBar().showMessage("正在执行DRIFT搜索...")
                result_data = await self.graphrag.drift_search(
                    query, community_level, temperature
                )

                # 处理本地搜索返回结果
                if isinstance(result_data, tuple) and len(result_data) > 0:
                    # 如果是元组，第一个元素是Markdown文本
                    result = result_data[0]
                else:
                    # 如果不是元组，则整个结果就是Markdown文本
                    result = result_data
                    
                # 使用markdown渲染结果
                self.result_text.set_markdown_content(result)
                # 更新token统计
                self.update_output_tokens(result)
                
            elif method_index == 5:  # 流式DRIFT搜索
                self.log_text.append_text("执行流式DRIFT搜索...\n")
                self.statusBar().showMessage("正在执行流式DRIFT搜索...")
                # 清空结果文本区域
                self.result_text.clear()
                # 执行流式搜索
                await self.graphrag.drift_search_streaming(
                    query, community_level, temperature,
                    callback=handle_streaming_output
                )
                
            elif method_index == 6:  # 基本搜索
                self.log_text.append_text("执行基本搜索...\n")
                # 获取k值
                k, ok = QInputDialog.getInt(
                    self, "参数设置", "请输入要检索的文本单元数量:", 5, 1, 50, 1
                )
                if not ok:
                    k = 5
                
                self.log_text.append_text(f"使用k值: {k}\n")
                self.statusBar().showMessage(f"正在执行基本搜索 (k={k})...")
                result_data = await self.graphrag.basic_search(
                    query, k, temperature
                )

                # 处理本地搜索返回结果
                if isinstance(result_data, tuple) and len(result_data) > 0:
                    # 如果是元组，第一个元素是Markdown文本
                    result = result_data[0]
                else:
                    # 如果不是元组，则整个结果就是Markdown文本
                    result = result_data
                    
                # 使用markdown渲染结果
                self.result_text.set_markdown_content(result)
                # 更新token统计
                self.update_output_tokens(result)
                
            elif method_index == 7:  # 流式基本搜索
                self.log_text.append_text("执行流式基本搜索...\n")
                # 获取k值
                k, ok = QInputDialog.getInt(
                    self, "参数设置", "请输入要检索的文本单元数量:", 5, 1, 50, 1
                )
                if not ok:
                    k = 5
                
                self.log_text.append_text(f"使用k值: {k}\n")
                self.statusBar().showMessage(f"正在执行流式基本搜索 (k={k})...")
                # 清空结果文本区域
                self.result_text.clear()
                # 执行流式搜索
                await self.graphrag.basic_search_streaming(
                    query, k, temperature,
                    callback=handle_streaming_output
                )
            
            # 从GraphRAGInteractive对象获取上下文数据
            # 在所有查询结束后统一获取，确保每种查询方法都能正确获取上下文数据
            self.context_data = self.graphrag.context_data
            
            # 计算执行时间
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            # 解析上下文数据
            self.parse_context_data()
            
            # 记录查询完成
            self.log_text.append_text(f"查询执行完成，耗时: {execution_time:.2f}秒\n")
            # 更新状态栏
            self.statusBar().showMessage(f"查询完成，耗时: {execution_time:.2f}秒")
            
        except Exception as e:
            error_message = f"查询执行时出错: {str(e)}"
            self.log_text.append_text(f"{error_message}\n")
            
            self.statusBar().showMessage("查询执行失败")
            
            # 记录详细错误信息到日志
            import traceback
            traceback_info = traceback.format_exc()
            self.log_text.append_text(f"错误详情:\n{traceback_info}\n")
            
        finally:
            self.query_button.setEnabled(True)
            self.query_running = False
    
    def parse_context_data(self):
        """解析并显示上下文数据"""
        try:
            # 获取上下文数据
            context_data = self.context_data
            if not context_data:
                self.context_text.clear_and_set_text("无可用的上下文数据")
                return
            
            # 获取当前查询方法名称
            method_name = self.method_combo.currentText()
            self.log_text.append_text(f"\n开始解析{method_name}的上下文数据...\n")
            
            # 根据不同的查询方法，使用不同的格式化函数
            html_content = ""
            if "全局搜索" in method_name:
                html_content = self._format_global_search_context(context_data)
                # 在日志中显示上下文数据摘要信息
                self._log_context_summary("全局搜索", context_data)
            elif "本地搜索" in method_name:
                html_content = self._format_local_search_context(context_data)
                # 在日志中显示上下文数据摘要信息
                self._log_context_summary("本地搜索", context_data)
            elif "DRIFT搜索" in method_name:
                html_content = self._format_drift_search_context(context_data)
                # 在日志中显示上下文数据摘要信息
                self._log_context_summary("DRIFT搜索", context_data)
            elif "基本搜索" in method_name:
                html_content = self._format_basic_search_context(context_data)
                # 在日志中显示上下文数据摘要信息
                self._log_context_summary("基本搜索", context_data)
            
            # 设置HTML内容到上下文文本框
            self.context_text.set_html_content(html_content)
                
            # 切换到上下文选项卡
            self.tabs.setCurrentIndex(1)
            
        except Exception as e:
            self.log_text.append_text(f"解析上下文数据时出错: {str(e)}\n")
            import traceback
            traceback_info = traceback.format_exc()
            self.log_text.append_text(f"错误详情:\n{traceback_info}\n")
    
    def _format_global_search_context(self, context_data):
        """格式化全局搜索的上下文数据"""
        import pandas as pd
        
        html = """
        <html>
        <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; font-size: 26px; margin-bottom: 20px; }
            h2 { color: #3498db; font-size: 22px; margin-top: 30px; margin-bottom: 15px; }
            h3 { color: #7f8c8d; font-size: 20px; margin-top: 20px; margin-bottom: 10px; }
            .section { margin-bottom: 25px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            .info { color: #555; margin-bottom: 10px; font-size: 15px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 14px; }
            th { background-color: #3498db; color: white; text-align: left; padding: 8px; font-size: 13px; line-height: 1.3; }
            td { border: 1px solid #ddd; padding: 8px; font-size: 13px; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .highlight { background-color: #ffffcc; }
            .important { color: #e74c3c; font-weight: bold; }
            .text-content { max-width: 300px; word-wrap: break-word; }
            .empty-data { color: #7f8c8d; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            pre { background-color: #f8f9fa; padding: 10px; overflow-x: auto; border-radius: 5px; }
        </style>
        </head>
        <body>
        <h1>全局搜索上下文数据</h1>
        <div class="info">全局搜索使用Map-Reduce方法对所有社区报告进行搜索，适合回答需要整体理解的问题。</div>
        """
        
        # 添加查询参数信息
        html += f"""
        <div class="section">
        <h2>查询参数</h2>
        <p><strong>社区级别:</strong> {self.community_level.value()}  <strong>温度参数:</strong> {self.temperature.value()}</p>
        </div>
        """
        
        # 检查上下文数据类型
        if not isinstance(context_data, dict):
            html += """
            <div class="section">
            <h2>上下文数据格式错误</h2>
            <p class="important">上下文数据不是预期的字典格式。</p>
            </div>
            """
            return html + "</body></html>"
        
        html += """
        <div class="section">
        <h2>全局搜索上下文摘要</h2>
        <p>全局搜索通过Map-Reduce方法处理社区报告，以下是参与搜索的社区报告数据。</p>
        """
        
        # 处理reports字段
        if 'reports' in context_data:
            reports = context_data['reports']
            html += """
            <h3>社区报告</h3>
            """
            
            if isinstance(reports, pd.DataFrame):
                if not reports.empty:
                    # 创建社区报告表格展示
                    html += f"<p>全局搜索使用了{len(reports)}个社区报告进行Map-Reduce处理：</p>"
                    html += "<table>"
                    # 动态生成表头，使用中英文对照
                    html += "<tr>"
                    for col in reports.columns:
                        display_name = self._get_column_display_name(col)
                        html += f"<th>{display_name}</th>"
                    html += "</tr>"
                    
                    for _, report in reports.iterrows():
                        html += "<tr>"
                        for col in reports.columns:
                            value = report.get(col, '')
                            # 对于内容列进行特殊处理
                            if col in ['content', 'full_content'] and isinstance(value, str):
                                html += f'<td class="text-content">{value}</td>'
                            else:
                                html += f"<td>{value}</td>"
                        html += "</tr>"
                    
                    html += "</table>"
                else:
                    html += "<div class='empty-data'>社区报告数据为空。</div>"
            else:
                html += "<div class='empty-data'>社区报告数据格式无法解析</div>"
        else:
            html += "<div class='empty-data'>上下文数据中未找到reports字段。</div>"
        
        html += "</div>"
        
        html += """
        </body>
        </html>
        """
        return html
    
    def _format_local_search_context(self, context_data):
        """格式化本地搜索的上下文数据"""
        import pandas as pd
        
        html = """
        <html>
        <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; font-size: 26px; margin-bottom: 20px; }
            h2 { color: #3498db; font-size: 22px; margin-top: 30px; margin-bottom: 15px; }
            h3 { color: #7f8c8d; font-size: 20px; margin-top: 20px; margin-bottom: 10px; }
            .section { margin-bottom: 25px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            .info { color: #555; margin-bottom: 10px; font-size: 15px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 14px; }
            th { background-color: #3498db; color: white; text-align: left; padding: 8px; font-size: 13px; line-height: 1.3; }
            td { border: 1px solid #ddd; padding: 8px; font-size: 13px; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .highlight { background-color: #ffffcc; }
            .important { color: #e74c3c; font-weight: bold; }
            .text-content { max-width: 300px; word-wrap: break-word; }
            .empty-data { color: #7f8c8d; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            pre { background-color: #f8f9fa; padding: 10px; overflow-x: auto; border-radius: 5px; }
        </style>
        </head>
        <body>
        <h1>本地搜索上下文数据</h1>
        <div class="info">本地搜索结合知识图谱和原始文档数据，回答关于特定实体的详细问题。</div>
        """
        
        # 添加查询参数信息
        html += f"""
        <div class="section">
        <h2>查询参数</h2>
        <p><strong>社区级别:</strong> {self.community_level.value()}  <strong>温度参数:</strong> {self.temperature.value()}</p>
        </div>
        """
        
        # 检查上下文数据类型
        if not isinstance(context_data, dict):
            html += """
            <div class="section">
            <h2>上下文数据格式错误</h2>
            <p class="important">上下文数据不是预期的字典格式。</p>
            </div>
            """
            return html + "</body></html>"
        
        html += """
        <div class="section">
        <h2>本地搜索上下文摘要</h2>
        <p>本地搜索通过混合上下文构建，结合多种数据源生成详细回答。</p>
        """
        
        # 定义字段处理顺序和显示名称
        field_info = {
            'entities': {'title': '相关实体', 'description': '与查询相关的实体信息'},
            'relationships': {'title': '实体关系', 'description': '实体之间的关系连接'},
            'reports': {'title': '社区报告', 'description': '相关社区的摘要报告'},
            'sources': {'title': '文本来源', 'description': '原始文档中的相关文本片段'},
            'claims': {'title': '声明信息', 'description': '提取的事实声明和协变量'}
        }
        
        # 按顺序处理每个字段
        for field_name, field_config in field_info.items():
            if field_name in context_data:
                data = context_data[field_name]
                html += f"""
                <h3>{field_config['title']}</h3>
                <p class="info">{field_config['description']}</p>
                """
                
                if isinstance(data, pd.DataFrame):
                    if not data.empty:
                        # 显示非空DataFrame
                        html += f"<p>找到 {len(data)} 条{field_config['title']}记录：</p>"
                        html += "<table>"
                        
                        # 动态生成表头，使用中英文对照
                        html += "<tr>"
                        for col in data.columns:
                            display_name = self._get_column_display_name(col)
                            html += f"<th>{display_name}</th>"
                        html += "</tr>"
                        
                        # 显示所有数据行，不限制行数
                        for _, row in data.iterrows():
                            html += "<tr>"
                            for col in data.columns:
                                value = row.get(col, '')
                                # 根据字段类型进行特殊处理
                                if col in ['text', 'content', 'description', 'full_content'] and isinstance(value, str):
                                    html += f'<td class="text-content">{value}</td>'
                                else:
                                    html += f"<td>{value}</td>"
                            html += "</tr>"
                        
                        html += "</table>"
                    else:
                        # 处理空DataFrame
                        if field_name == 'claims':
                            html += """
                            <div class="empty-data">
                            未提取到声明信息。这可能是因为声明提取功能未启用或当前查询不涉及具体声明。
                            """
                            if len(data.columns) > 0:
                                html += f"<br>可用列结构: {', '.join(data.columns)}"
                            html += "</div>"
                        else:
                            html += f"<div class='empty-data'>未找到相关的{field_config['title']}。</div>"
                else:
                    html += f"<div class='empty-data'>{field_config['title']}数据格式无法解析</div>"
        
        html += "</div>"
        
        html += """
        </body>
        </html>
        """
        return html

    def _format_drift_search_context(self, context_data):
        """格式化DRIFT搜索的上下文数据"""
        import pandas as pd
        
        html = """
        <html>
        <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; font-size: 26px; margin-bottom: 20px; }
            h2 { color: #3498db; font-size: 22px; margin-top: 30px; margin-bottom: 15px; }
            h3 { color: #7f8c8d; font-size: 20px; margin-top: 20px; margin-bottom: 10px; }
            .section { margin-bottom: 25px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            .info { color: #555; margin-bottom: 10px; font-size: 15px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 14px; }
            th { background-color: #3498db; color: white; text-align: left; padding: 8px; font-size: 13px; line-height: 1.3; }
            td { border: 1px solid #ddd; padding: 8px; font-size: 13px; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .highlight { background-color: #ffffcc; }
            .important { color: #e74c3c; font-weight: bold; }
            .text-content { max-width: 300px; word-wrap: break-word; }
            .empty-data { color: #7f8c8d; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            pre { background-color: #f8f9fa; padding: 10px; overflow-x: auto; border-radius: 5px; }
        </style>
        </head>
        <body>
        <h1>DRIFT搜索上下文数据</h1>
        <div class="info">DRIFT (Deep Recursive Information Finding Technique) 搜索是一种递归搜索策略，通过深入挖掘相关信息生成综合回答。</div>
        """
        
        # 添加查询参数信息
        html += f"""
        <div class="section">
        <h2>查询参数</h2>
        <p><strong>社区级别:</strong> {self.community_level.value()}  <strong>温度参数:</strong> {self.temperature.value()}</p>
        </div>
        """
        
        # 检查上下文数据类型
        if not isinstance(context_data, dict):
            html += """
            <div class="section">
            <h2>上下文数据格式错误</h2>
            <p class="important">上下文数据不是预期的字典格式。</p>
            </div>
            """
            return html + "</body></html>"
        
        html += """
        <div class="section">
        <h2>DRIFT搜索上下文摘要</h2>
        <p>DRIFT搜索通过递归方式收集相关信息，以下是生成回答时使用的关键数据。</p>
        """
        
        # 1. 处理实体部分
        if 'entities' in context_data:
            entities = context_data['entities']
            html += """
            <h3>相关实体</h3>
            """
            
            if isinstance(entities, pd.DataFrame):
                if not entities.empty:
                    # 展示非空DataFrame
                    html += f"<p>找到 {len(entities)} 个相关实体：</p>"
                    html += "<table>"
                    # 动态生成表头，使用中英文对照
                    html += "<tr>"
                    for col in entities.columns:
                        display_name = self._get_column_display_name(col)
                        html += f"<th>{display_name}</th>"
                    html += "</tr>"
                    
                    for _, entity in entities.iterrows():
                        html += "<tr>"
                        for col in entities.columns:
                            value = entity.get(col, '')
                            html += f"<td>{value}</td>"
                        html += "</tr>"
                    
                    html += "</table>"
                else:
                    html += """
                    <div class="empty-data">
                    查询未使用具体实体或未找到相关实体。
                    """
                    if len(entities.columns) > 0:
                        html += f"<br>实体数据框列：{', '.join(entities.columns)}"
                    html += "</div>"
            else:
                html += "<div class='empty-data'>实体数据格式无法解析</div>"
        
        # 2. 处理文本来源部分
        if 'sources' in context_data:
            sources = context_data['sources']
            html += """
            <h3>文本来源</h3>
            """
            
            if isinstance(sources, pd.DataFrame):
                if not sources.empty:
                    # 创建文本片段表格展示
                    html += f"<p>以下是与查询相关的{len(sources)}个文本片段：</p>"
                    html += "<table>"
                    # 动态生成表头，使用中英文对照
                    html += "<tr>"
                    for col in sources.columns:
                        display_name = self._get_column_display_name(col)
                        html += f"<th>{display_name}</th>"
                    html += "</tr>"
                    
                    for _, source in sources.iterrows():
                        html += "<tr>"
                        for col in sources.columns:
                            value = source.get(col, '')
                            # 对于文本内容列进行特殊处理
                            if col == 'text' and isinstance(value, str):
                                html += f'<td class="text-content">{value}</td>'
                            else:
                                html += f"<td>{value}</td>"
                        html += "</tr>"
                    
                    html += "</table>"
                else:
                    html += "<div class='empty-data'>未找到相关文本片段。</div>"
            else:
                html += "<div class='empty-data'>文本片段数据格式无法解析</div>"
        
        html += "</div>"
        
        html += """
        </body>
        </html>
        """
        return html
    
    def _format_basic_search_context(self, context_data):
        """格式化基本搜索的上下文数据"""
        import pandas as pd
        
        html = """
        <html>
        <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; font-size: 26px; margin-bottom: 20px; }
            h2 { color: #3498db; font-size: 22px; margin-top: 30px; margin-bottom: 15px; }
            h3 { color: #7f8c8d; font-size: 20px; margin-top: 20px; margin-bottom: 10px; }
            .section { margin-bottom: 25px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            .info { color: #555; margin-bottom: 10px; font-size: 15px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 14px; }
            th { background-color: #3498db; color: white; text-align: left; padding: 8px; font-size: 13px; line-height: 1.3; }
            td { border: 1px solid #ddd; padding: 8px; font-size: 13px; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .highlight { background-color: #ffffcc; }
            .important { color: #e74c3c; font-weight: bold; }
            .text-content { max-width: 300px; word-wrap: break-word; }
            .empty-data { color: #7f8c8d; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            pre { background-color: #f8f9fa; padding: 10px; overflow-x: auto; border-radius: 5px; }
        </style>
        </head>
        <body>
        <h1>基本搜索上下文数据</h1>
        <div class="info">基本搜索是一种简单的向量RAG实现，通过向量相似度检索最相关的文本片段。</div>
        """
        
        # 添加查询参数信息
        html += f"""
        <div class="section">
        <h2>查询参数</h2>
        <p><strong>社区级别:</strong> {self.community_level.value()}  <strong>温度参数:</strong> {self.temperature.value()}</p>
        </div>
        <div class="section">
        <h2>基本搜索上下文摘要</h2>
        <p>基本搜索通过向量相似度匹配，以下是检索到的最相关文本片段。</p>
        """
        
        # 处理Sources字段（注意大写S）
        if 'Sources' in context_data:
            sources = context_data['Sources']
            html += """
            <h3>检索到的文本片段</h3>
            """
            
            if isinstance(sources, pd.DataFrame):
                if not sources.empty:
                    # 创建文本片段表格展示
                    html += f"<p>基于向量相似度检索到{len(sources)}个最相关的文本片段：</p>"
                    html += "<table>"
                    # 动态生成表头，使用中英文对照
                    html += "<tr>"
                    for col in sources.columns:
                        display_name = self._get_column_display_name(col)
                        html += f"<th>{display_name}</th>"
                    html += "</tr>"
                    
                    for _, source in sources.iterrows():
                        html += "<tr>"
                        for col in sources.columns:
                            value = source.get(col, '')
                            # 对于文本内容列进行特殊处理
                            if col == 'text' and isinstance(value, str):
                                html += f'<td class="text-content">{value}</td>'
                            else:
                                html += f"<td>{value}</td>"
                        html += "</tr>"
                    
                    html += "</table>"
                else:
                    html += "<div class='empty-data'>未找到相关文本片段。</div>"
            else:
                html += "<div class='empty-data'>文本片段数据格式无法解析</div>"
        else:
            html += "<div class='empty-data'>上下文数据中未找到Sources字段。</div>"
        
        html += "</div>"
        
        html += """
        </body>
        </html>
        """
        return html

    def _get_column_display_name(self, col_name):
        """获取列名的中英文对照显示名称"""
        # 核心列映射 - 根据实际控制台日志数据完善
        column_mapping = {
            # 通用字段
            'id': 'ID (标识符)',
            'title': 'Title (标题)',
            'content': 'Content (内容)',
            'text': 'Text (文本内容)',
            'in_context': 'In Context (上下文中)',  # 新增DRIFT搜索中的in_context列
            
            # 实体表字段 (entities)
            'entity': 'Entity (实体名称)',
            'number of relationships': 'Number of Relationships (关系数量)',
            'human_readable_id': 'Human Readable ID (可读标识)',
            'description': 'Description (描述)',
            'type': 'Type (类型)',
            'community': 'Community (社区)',
            'degree': 'Degree (度数)',
            'x': 'X (X坐标)',
            'y': 'Y (Y坐标)',
            
            # 关系表字段 (relationships)
            'source': 'Source (源实体)',
            'target': 'Target (目标实体)',
            'weight': 'Weight (权重)',
            'links': 'Links (链接数)',
            'combined_degree': 'Combined Degree (组合度数)',
            'text_unit_ids': 'Text Unit IDs (文本单元ID)',
            
            # 文本片段字段 (sources/Sources)
            'source_id': 'Source ID (来源标识)',
            'text_unit_id': 'Text Unit ID (文本单元ID)',
            'chunk_id': 'Chunk ID (文本块ID)',
            'document_ids': 'Document IDs (文档ID)',
            'n_tokens': 'N Tokens (标记数)',
            
            # 社区报告字段 (reports)
            'community_id': 'Community ID (社区ID)',
            'level': 'Level (级别)',
            'rank': 'Rank (排名)',
            'occurrence weight': 'Occurrence Weight (出现权重)',
            'summary': 'Summary (摘要)',
            'full_content': 'Full Content (完整内容)',
            'findings': 'Findings (发现)',
            'rating': 'Rating (评分)',
            'rating_explanation': 'Rating Explanation (评分说明)',
            
            # 声明/协变量字段 (claims)
            'subject_id': 'Subject ID (主体ID)',
            'object_id': 'Object ID (客体ID)',
            'claim_type': 'Claim Type (声明类型)',
            'status': 'Status (状态)',
            'start_date': 'Start Date (开始日期)',
            'end_date': 'End Date (结束日期)',
            
            # 其他常见字段
            'name': 'Name (名称)',
            'score': 'Score (分数)',
            'similarity': 'Similarity (相似度)',
            'distance': 'Distance (距离)',
            'frequency': 'Frequency (频率)',
            'size': 'Size (大小)',
            'parent': 'Parent (父级)',
            'children': 'Children (子级)',
            'metadata': 'Metadata (元数据)',
            'attributes': 'Attributes (属性)',
        }
        
        # 如果找到映射，返回中英文对照
        if col_name in column_mapping:
            return column_mapping[col_name]
        
        # 默认情况：显示原始列名，添加小字体提示
        return f'{col_name}<br><span style="font-size:11px;color:#888;">{col_name}</span>'
    
    def _log_context_summary(self, method_name, context_data):
        """在日志中显示上下文数据摘要"""
        self.log_text.append_text("\n=== 上下文数据摘要 ===\n")
        
        # 检查上下文数据类型
        if not isinstance(context_data, dict):
            self.log_text.append_text(f"⚠️ 上下文数据格式异常: {type(context_data)}\n")
            return
        
        # 根据不同查询方法处理上下文数据
        if "全局搜索" in method_name:
            self._log_global_search_summary(context_data)
        elif "本地搜索" in method_name:
            self._log_local_search_summary(context_data)
        elif "DRIFT搜索" in method_name:
            self._log_drift_search_summary(context_data)
        elif "基本搜索" in method_name:
            self._log_basic_search_summary(context_data)
        
        self.log_text.append_text("========================\n")
    
    def _log_global_search_summary(self, context_data):
        """记录全局搜索的上下文摘要到日志"""
        import pandas as pd
        
        try:
            if not isinstance(context_data, dict):
                self.log_text.append_text("⚠️ 全局搜索上下文数据格式错误：不是字典格式\n")
                return
            
            self.log_text.append_text("=== 全局搜索上下文数据摘要 ===\n")
            
            # 处理reports字段
            if 'reports' in context_data:
                reports = context_data['reports']
                if isinstance(reports, pd.DataFrame):
                    if not reports.empty:
                        self.log_text.append_text(f"📊 社区报告: {len(reports)} 个报告\n")
                        self.log_text.append_text(f"   - 列名: {', '.join(reports.columns)}\n")
                        
                        # 计算内容长度统计
                        content_lengths = reports['content'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
                        avg_length = content_lengths.mean()
                        max_length = content_lengths.max()
                        min_length = content_lengths.min()
                        self.log_text.append_text(f"   - 内容长度统计: 平均{avg_length:.0f}字符, 最长{max_length}字符, 最短{min_length}字符\n")
                    else:
                        self.log_text.append_text("📊 社区报告: 空DataFrame\n")
                        if len(reports.columns) > 0:
                            self.log_text.append_text(f"   - 可用列: {', '.join(reports.columns)}\n")
                else:
                    self.log_text.append_text(f"📊 社区报告: 无法解析的格式 ({type(reports).__name__})\n")
            else:
                self.log_text.append_text("📊 社区报告: 未找到reports字段\n")
            
            self.log_text.append_text("=== 全局搜索上下文摘要结束 ===\n")
            
        except Exception as e:
            self.log_text.append_text(f"❌ 记录全局搜索摘要时出错: {str(e)}\n")
            import traceback
            self.log_text.append_text(f"错误详情: {traceback.format_exc()}\n")
    
    def _log_local_search_summary(self, context_data):
        """记录本地搜索的上下文摘要到日志"""
        import pandas as pd
        
        try:
            self.log_text.append_text("=== 本地搜索上下文数据摘要 ===\n")
            
            # 添加调试信息
            self.log_text.append_text(f"🔍 上下文数据包含字段: {list(context_data.keys())}\n")
            
            # 定义字段处理顺序和描述
            field_info = {
                'entities': '相关实体',
                'relationships': '实体关系',
                'reports': '社区报告',
                'sources': '文本来源',
                'claims': '声明信息'
            }
            
            # 按顺序处理每个字段
            for field_name, field_desc in field_info.items():
                if field_name in context_data:
                    data = context_data[field_name]
                    
                    # 添加调试信息
                    self.log_text.append_text(f"🔍 处理字段 {field_name}: 数据类型 {type(data).__name__}\n")
                    
                    if isinstance(data, pd.DataFrame):
                        self.log_text.append_text(f"   - DataFrame形状: {data.shape}\n")
                        if not data.empty:
                            self.log_text.append_text(f"📊 {field_desc}: {len(data)} 条记录\n")
                            self.log_text.append_text(f"   - 列名: {', '.join(data.columns)}\n")
                            
                            # 根据字段类型提供特定统计信息
                            if field_name == 'entities':
                                # 实体类型统计
                                if 'type' in data.columns:
                                    type_counts = data['type'].value_counts()
                                    top_types = type_counts.head(3)
                                    type_info = [f"{t}({c}个)" for t, c in top_types.items()]
                                    self.log_text.append_text(f"   - 主要类型: {', '.join(type_info)}\n")
                            
                            elif field_name == 'relationships':
                                # 关系权重统计
                                if 'weight' in data.columns:
                                    # 添加详细调试信息
                                    self.log_text.append_text(f"   - weight列数据类型: {data['weight'].dtype}\n")
                                    self.log_text.append_text(f"   - weight列前5个值: {list(data['weight'].head())}\n")
                                    
                                    try:
                                        # 处理权重数据
                                        weights_series = data['weight']
                                        processed_weights = []
                                        
                                        for weight_value in weights_series:
                                            if pd.isna(weight_value):
                                                continue
                                            
                                            weight_str = str(weight_value)
                                            
                                            # 检查是否是连接的字符串（包含多个数值）
                                            if '.' in weight_str and len(weight_str) > 10:
                                                # 尝试分割连接的数值字符串
                                                # 假设数值之间没有分隔符，需要智能分割
                                                # 查找所有可能的数值模式
                                                numbers = re.findall(r'\d+\.?\d*', weight_str)
                                                for num_str in numbers:
                                                    try:
                                                        num = float(num_str)
                                                        if 0 <= num <= 100:  # 合理的权重范围
                                                            processed_weights.append(num)
                                                    except ValueError:
                                                        continue
                                            else:
                                                # 尝试直接转换
                                                try:
                                                    num = float(weight_str)
                                                    processed_weights.append(num)
                                                except ValueError:
                                                    continue
                                        
                                        if processed_weights:
                                            avg_weight = sum(processed_weights) / len(processed_weights)
                                            max_weight = max(processed_weights)
                                            min_weight = min(processed_weights)
                                            self.log_text.append_text(f"   - 权重统计: 平均{avg_weight:.2f}, 范围{min_weight:.2f}-{max_weight:.2f} (处理了{len(processed_weights)}个有效值)\n")
                                        else:
                                            self.log_text.append_text("   - 权重统计: 无有效数值数据\n")
                                    except Exception as weight_error:
                                        self.log_text.append_text(f"   - 权重统计: 数据格式异常 ({str(weight_error)})\n")
                            
                            elif field_name == 'sources':
                                # 文本长度统计
                                if 'text' in data.columns:
                                    try:
                                        text_lengths = data['text'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
                                        if len(text_lengths) > 0:
                                            avg_length = text_lengths.mean()
                                            max_length = text_lengths.max()
                                            min_length = text_lengths.min()
                                            self.log_text.append_text(f"   - 文本长度: 平均{avg_length:.0f}字符, 范围{min_length}-{max_length}字符\n")
                                        else:
                                            self.log_text.append_text("   - 文本长度: 无有效文本数据\n")
                                    except Exception as text_error:
                                        self.log_text.append_text(f"   - 文本长度: 数据格式异常 ({str(text_error)})\n")
                            
                            elif field_name == 'reports':
                                # 报告内容长度统计
                                try:
                                    content_lengths = data['content'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
                                    if len(content_lengths) > 0:
                                        avg_length = content_lengths.mean()
                                        self.log_text.append_text(f"   - 报告内容平均长度: {avg_length:.0f}字符\n")
                                    else:
                                        self.log_text.append_text("   - 报告内容: 无有效内容数据\n")
                                except Exception as content_error:
                                    self.log_text.append_text(f"   - 报告内容: 数据格式异常 ({str(content_error)})\n")
                        else:
                            # 处理空DataFrame
                            if field_name == 'claims':
                                self.log_text.append_text(f"📊 {field_desc}: 空DataFrame (声明提取可能未启用)\n")
                                if len(data.columns) > 0:
                                    self.log_text.append_text(f"   - 可用列: {', '.join(data.columns)}\n")
                    else:
                        self.log_text.append_text(f"📊 {field_desc}: 无法解析的格式 ({type(data).__name__})\n")
                else:
                    self.log_text.append_text(f"📊 {field_desc}: 未找到{field_name}字段\n")
            
            # 计算总体统计
            total_records = 0
            for field_name in field_info.keys():
                if field_name in context_data:
                    data = context_data[field_name]
                    if isinstance(data, pd.DataFrame):
                        total_records += len(data)
            
            self.log_text.append_text(f"📈 总计: {total_records} 条记录用于构建本地搜索上下文\n")
            self.log_text.append_text("=== 本地搜索上下文摘要结束 ===\n")
            
        except Exception as e:
            self.log_text.append_text(f"❌ 记录本地搜索摘要时出错: {str(e)}\n")
            import traceback
            self.log_text.append_text(f"错误详情: {traceback.format_exc()}\n")
    
    def _log_drift_search_summary(self, context_data):
        """记录DRIFT搜索的上下文摘要到日志"""
        import pandas as pd
        
        try:
            if not isinstance(context_data, dict):
                self.log_text.append_text("⚠️ DRIFT搜索上下文数据格式错误：不是字典格式\n")
                return
            
            self.log_text.append_text("=== DRIFT搜索上下文数据摘要 ===\n")
            
            # 处理entities字段
            if 'entities' in context_data:
                entities = context_data['entities']
                if isinstance(entities, pd.DataFrame):
                    if not entities.empty:
                        self.log_text.append_text(f"实体数据: {len(entities)} 个实体\n")
                        self.log_text.append_text(f"   - 列名: {', '.join(entities.columns)}\n")
                    else:
                        self.log_text.append_text("实体数据: 空DataFrame\n")
                        if len(entities.columns) > 0:
                            self.log_text.append_text(f"   - 可用列: {', '.join(entities.columns)}\n")
                else:
                    self.log_text.append_text(f"实体数据: 无法解析的格式 ({type(entities).__name__})\n")
            else:
                self.log_text.append_text("实体数据: 未找到entities字段\n")
            
            # 处理sources字段
            if 'sources' in context_data:
                sources = context_data['sources']
                if isinstance(sources, pd.DataFrame):
                    if not sources.empty:
                        self.log_text.append_text(f"文本来源: {len(sources)} 个文本片段\n")
                        self.log_text.append_text(f"   - 列名: {', '.join(sources.columns)}\n")
                        
                        # 计算文本长度统计
                        if 'text' in sources.columns:
                            text_lengths = sources['text'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
                            avg_length = text_lengths.mean()
                            max_length = text_lengths.max()
                            min_length = text_lengths.min()
                            self.log_text.append_text(f"   - 文本长度统计: 平均{avg_length:.0f}字符, 最长{max_length}字符, 最短{min_length}字符\n")
                    else:
                        self.log_text.append_text("文本来源: 空DataFrame\n")
                        if len(sources.columns) > 0:
                            self.log_text.append_text(f"   - 可用列: {', '.join(sources.columns)}\n")
                else:
                    self.log_text.append_text(f"文本来源: 无法解析的格式 ({type(sources).__name__})\n")
            else:
                self.log_text.append_text("文本来源: 未找到sources字段\n")
            
            self.log_text.append_text("=== DRIFT搜索上下文摘要结束 ===\n")
            
        except Exception as e:
            self.log_text.append_text(f"❌ 记录DRIFT搜索摘要时出错: {str(e)}\n")
            import traceback
            self.log_text.append_text(f"错误详情: {traceback.format_exc()}\n")
    
    def _log_basic_search_summary(self, context_data):
        """记录基本搜索的上下文摘要到日志"""
        import pandas as pd
        
        try:
            if not isinstance(context_data, dict):
                self.log_text.append_text("⚠️ 基本搜索上下文数据格式错误：不是字典格式\n")
                return
            
            self.log_text.append_text("=== 基本搜索上下文数据摘要 ===\n")
            
            # 添加调试信息
            self.log_text.append_text(f"🔍 上下文数据包含字段: {list(context_data.keys())}\n")
            
            # 处理Sources字段（注意大写S）
            if 'Sources' in context_data:
                sources = context_data['Sources']
                
                # 添加调试信息
                self.log_text.append_text(f"🔍 处理字段 Sources: 数据类型 {type(sources).__name__}\n")
                
                if isinstance(sources, pd.DataFrame):
                    self.log_text.append_text(f"   - DataFrame形状: {sources.shape}\n")
                    if not sources.empty:
                        self.log_text.append_text(f"📊 文本来源: {len(sources)} 个文本片段\n")
                        self.log_text.append_text(f"   - 列名: {', '.join(sources.columns)}\n")
                        
                        # 计算文本长度统计
                        if 'text' in sources.columns:
                            try:
                                text_lengths = sources['text'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
                                if len(text_lengths) > 0:
                                    avg_length = text_lengths.mean()
                                    max_length = text_lengths.max()
                                    min_length = text_lengths.min()
                                    self.log_text.append_text(f"   - 📏 文本长度统计: 平均{avg_length:.0f}字符, 范围{min_length}-{max_length}字符\n")
                                else:
                                    self.log_text.append_text("   - 📏 文本长度: 无有效文本数据\n")
                            except Exception as text_error:
                                self.log_text.append_text(f"   - 📏 文本长度: 数据格式异常 ({str(text_error)})\n")
                    else:
                        self.log_text.append_text("📊 文本来源: 空DataFrame\n")
                        if len(sources.columns) > 0:
                            self.log_text.append_text(f"   - 可用列: {', '.join(sources.columns)}\n")
                else:
                    self.log_text.append_text(f"📊 文本来源: 无法解析的格式 ({type(sources).__name__})\n")
            else:
                self.log_text.append_text("📊 文本来源: 未找到Sources字段\n")
            
            # 计算总体统计
            total_records = 0
            if 'Sources' in context_data:
                sources = context_data['Sources']
                total_records = len(sources)
            
            self.log_text.append_text(f"📈 总计: {total_records} 个文本片段用于构建基本搜索上下文\n")
            self.log_text.append_text("=== 基本搜索上下文摘要结束 ===\n")
            
        except Exception as e:
            self.log_text.append_text(f"❌ 记录基本搜索摘要时出错: {str(e)}\n")
            import traceback
            self.log_text.append_text(f"错误详情: {traceback.format_exc()}\n")
    
    def update_input_tokens(self):
        """更新输入token计数"""
        text = self.query_input.toPlainText()
        token_count = self.token_counter.update_input_tokens(text)
        self.input_token_label.setText(str(token_count))
    
    def update_output_tokens(self, text):
        """更新输出token计数"""
        token_count = self.token_counter.update_output_tokens(text)
        self.output_token_label.setText(str(token_count))
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
        <html>
        <head>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                line-height: 1.5;
                color: #333;
                padding: 10px;
            }
            h2 {
                color: #2980b9;
                font-size: 16px;
                margin-bottom: 10px;
                border-bottom: 1px solid #eee;
                padding-bottom: 8px;
            }
            h3 {
                color: #27ae60;
                font-size: 16px;
                margin-top: 10px;
                margin-bottom: 5px;
            }
            .intro {
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                border-left: 4px solid #3498db;
            }
            ol, ul {
                margin-top: 6px;
                margin-bottom: 12px;
                padding-left: 25px;
            }
            li {
                margin-bottom: 5px;
            }
            .highlight {
                background-color: #e8f4f8;
                padding: 10px;
                border-radius: 5px;
                margin-top: 12px;
                margin-bottom: 12px;
            }
            .tip {
                background-color: #e8f8f5;
                padding: 10px;
                border-radius: 5px;
                margin-top: 12px;
                border-left: 4px solid #2ecc71;
            }
            b {
                color: #2c3e50;
            }
        </style>
        </head>
        <body>
        <h2>GraphRAG 知识搜索助手使用指南</h2>
        
        <div class="intro">
        GraphRAG是一种先进的知识检索增强生成系统，能够基于知识图谱智能回答您的问题。本助手帮助您轻松使用GraphRAG的强大功能。
        </div>
        
        <h3>基本操作步骤</h3>
        <ol>
            <li>点击<b>"选择项目目录"</b>按钮，选择GraphRAG项目目录</li>
            <li>点击<b>"加载配置"</b>按钮，加载项目配置和索引数据</li>
            <li>在提问区域输入您的问题</li>
            <li>选择合适的查询方法（不同方法适合不同类型的问题）</li>
            <li>设置社区级别和温度参数</li>
            <li>点击<b>"执行查询"</b>按钮获取回答</li>
        </ol>
        
        <h3>界面区域说明</h3>
        <ul>
            <li><b>提问区域</b>：输入您的问题并设置查询参数</li>
            <li><b>查询结果</b>：显示系统对您问题的回答</li>
            <li><b>上下文信息</b>：查看系统如何找到答案的详细信息</li>
            <li><b>日志</b>：显示系统操作日志和错误信息</li>
        </ul>
        
        <div class="highlight">
        <h3>参数说明</h3>
        <ul>
            <li><b>社区级别</b>：控制知识图谱中使用的社区层级深度
                <ul>
                    <li>0表示顶层社区（最宏观的视角）</li>
                    <li>更高的值表示更细分的社区（更具体的视角）</li>
                </ul>
            </li>
            <li><b>温度</b>：控制回答的创造性/确定性（0-2）
                <ul>
                    <li>较低值（如0.3）：回答更确定、更一致</li>
                    <li>较高值（如0.8）：回答更多样化、更创造性</li>
                </ul>
            </li>
        </ul>
        </div>
        
        <div class="tip">
        <p>要了解各种查询方法的特点和适用场景，请点击"查询方法说明"按钮。</p>
        </div>
        
        </body>
        </html>
        """
        
        # 创建自定义对话框而不是消息框
        dialog = QDialog(self)
        dialog.setWindowTitle("GraphRAG使用帮助")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setModal(True)
        
        # 创建对话框布局
        layout = QVBoxLayout(dialog)
        
        # 创建文本浏览器以显示HTML内容
        text_browser = QTextEdit(dialog)
        text_browser.setReadOnly(True)
        text_browser.setHtml(help_text)
        layout.addWidget(text_browser)
        
        # 添加确定按钮
        button_box = QHBoxLayout()
        ok_button = QPushButton("确定", dialog)
        ok_button.clicked.connect(dialog.accept)
        button_box.addStretch()
        button_box.addWidget(ok_button)
        layout.addLayout(button_box)
        
        # 设置对话框大小
        dialog.resize(1000, 900)
        
        # 显示对话框
        dialog.exec_()
    
    def show_method_info(self):
        """显示查询方法说明"""
        method_info = """
        <html>
        <head>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                line-height: 1.5;
                color: #333;
                padding: 10px;
            }
            h2 {
                color: #2980b9;
                font-size: 16px;
                margin-bottom: 10px;
                border-bottom: 1px solid #eee;
                padding-bottom: 8px;
            }
            h3 {
                color: #2980b9;
                font-size: 16px;
                margin-top: 15px;
                margin-bottom: 5px;
            }
            p {
                margin-bottom: 10px;
            }
            .method {
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 5px;
                margin-bottom: 15px;
                border-left: 4px solid #3498db;
            }
            .method-title {
                font-weight: bold;
                color: #2c3e50;
                font-size: 16px;
                margin-bottom: 8px;
            }
            .suitable {
                background-color: #e8f8f5;
                padding: 8px;
                border-radius: 5px;
                margin-top: 8px;
                border-left: 3px solid #2ecc71;
            }
            .suitable-title {
                font-weight: bold;
                color: #27ae60;
                margin-bottom: 5px;
            }
            .method-comparison {
                margin-top: 20px;
                background-color: #eef4f7;
                padding: 12px;
                border-radius: 5px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 12px;
            }
            th {
                background-color: #3498db;
                color: white;
                text-align: left;
                padding: 8px;
            }
            td {
                border-bottom: 1px solid #ddd;
                padding: 8px;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        </style>
        </head>
        <body>
        <h2>GraphRAG 查询方法详解</h2>
        
        <div class="method">
            <div class="method-title">全局搜索</div>
            <p>全局搜索是GraphRAG中最适合获取整体知识的搜索方法。它使用"Map-Reduce"模式在所有社区报告上执行搜索，先从多个社区报告中提取相关信息(Map阶段)，再将这些信息组合成一个连贯的总体答案(Reduce阶段)。</p>
            
            <div class="suitable">
                <div class="suitable-title">适用场景</div>
                <p>整体概览类问题，如"文档中的主要主题是什么？"、"这个数据集包含哪些关键信息？"或"不同章节之间有什么关联？"</p>
            </div>
        </div>
        
        <div class="method">
            <div class="method-title">本地搜索</div>
            <p>本地搜索专注于回答关于特定实体或概念的详细问题。它首先识别查询中提到的实体，然后收集与这些实体直接相关的信息：它们的描述、关系、相关文本段落等。本地搜索能够提供更深入、更聚焦的答案。</p>
            
            <div class="suitable">
                <div class="suitable-title">适用场景</div>
                <p>特定实体问题，如"甲醛的化学性质是什么？"、"唐代诗人李白的主要作品有哪些？"或"维生素C的功效与作用是什么？"</p>
            </div>
        </div>
        
        <div class="method">
            <div class="method-title">DRIFT搜索</div>
            <p>DRIFT (Deep Recursive Information Finding Technique) 搜索是GraphRAG中最强大的搜索方法，它结合了全局和本地搜索的优点。它先从整体角度查看数据，找到相关的社区信息，然后递归深入探索相关实体及其关系，最后汇总所有发现以生成全面而深入的答案。</p>
            
            <div class="suitable">
                <div class="suitable-title">适用场景</div>
                <p>复杂关联问题，如"量子力学和相对论之间的关系如何？"、"数字化转型对教育行业的影响是什么？"或"气候变化与全球粮食安全之间有什么联系？"</p>
            </div>
        </div>
        
        <div class="method">
            <div class="method-title">基本搜索</div>
            <p>基本搜索是GraphRAG中最简单的搜索方法，类似于传统的RAG系统。它仅使用文本相似度查找与查询最相关的文本块，不利用知识图谱结构。它提供快速但相对粗略的搜索结果，适合简单查询或作为基准比较。</p>
            
            <div class="suitable">
                <div class="suitable-title">适用场景</div>
                <p>简单文本匹配，如查找特定事实或直接在文本中提及的信息，如"2019年中国GDP增长率是多少？"或"谁发明了电灯？"</p>
            </div>
        </div>
        
        <div class="method-comparison">
            <h3>查询方法比较</h3>
            <table>
                <tr>
                    <th>搜索方法</th>
                    <th>优势</th>
                    <th>资源消耗</th>
                    <th>结果深度</th>
                </tr>
                <tr>
                    <td>全局搜索</td>
                    <td>全面的整体视角</td>
                    <td>高</td>
                    <td>广泛但不深入</td>
                </tr>
                <tr>
                    <td>本地搜索</td>
                    <td>特定实体的深入信息</td>
                    <td>中</td>
                    <td>针对特定实体深入</td>
                </tr>
                <tr>
                    <td>DRIFT搜索</td>
                    <td>结合全局视角和深度探索</td>
                    <td>很高</td>
                    <td>既有广度又有深度</td>
                </tr>
                <tr>
                    <td>基本搜索</td>
                    <td>简单直接，响应快速</td>
                    <td>低</td>
                    <td>有限，仅基于文本相似度</td>
                </tr>
            </table>
        </div>
        
        <h3>流式与非流式查询</h3>
        <p>每种查询方法都有流式和非流式两种版本：</p>
        <ul>
            <li><b>流式版本</b>：实时返回响应片段，适合长回答场景，提供更好的用户体验</li>
            <li><b>非流式版本</b>：等待完整响应后一次性显示，适合需要完整格式化的回答</li>
        </ul>
        
        </body>
        </html>
        """
        
        # 创建自定义对话框而不是消息框
        dialog = QDialog(self)
        dialog.setWindowTitle("GraphRAG查询方法说明")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setModal(True)
        
        # 创建对话框布局
        layout = QVBoxLayout(dialog)
        
        # 创建文本浏览器以显示HTML内容
        text_browser = QTextEdit(dialog)
        text_browser.setReadOnly(True)
        text_browser.setHtml(method_info)
        layout.addWidget(text_browser)
        
        # 添加确定按钮
        button_box = QHBoxLayout()
        ok_button = QPushButton("确定", dialog)
        ok_button.clicked.connect(dialog.accept)
        button_box.addStretch()
        button_box.addWidget(ok_button)
        layout.addLayout(button_box)
        
        # 设置对话框大小
        dialog.resize(1000, 500)
        
        # 显示对话框
        dialog.exec_()

def main():
    """主函数"""
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 设置应用程序图标（使用内置图标）
    app_icon = app.style().standardIcon(app.style().SP_DialogHelpButton)
    app.setWindowIcon(app_icon)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建事件循环
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # 创建并显示主窗口
    window = GraphRAGGUI()
    window.show()
    
    # 运行事件循环
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
