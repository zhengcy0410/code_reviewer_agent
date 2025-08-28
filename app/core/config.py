# -*- coding: utf-8 -*-
"""
配置文件
"""
import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    # 数据库配置
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "19900410"
    MYSQL_DATABASE: str = "code-reviewer"
    
    # AI 模型配置
    DASHSCOPE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    
    
    # 模型选择配置
    LLM_PROVIDER: str = "openai"  # "qwen" 或 "openai"
    
    # 服务器配置
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    DEBUG: bool = True
    
    # 文件上传配置
    MAX_FILE_SIZE: int = 50000000  # 50MB
    UPLOAD_DIR: str = "./uploads"
    
    # 向量搜索配置
    ENABLE_VECTOR_SEARCH: bool = True
    
    # 向量模型配置 - 使用通义千问嵌入模型（中文+代码检索最佳）
    VECTOR_MODEL_PROVIDER: str = "qwen"
    VECTOR_MODEL_NAME: str = "text-embedding-v3"  # 通义千问嵌入模型
    VECTOR_MODEL_DIMENSION: int = 1024  # qwen-text-embedding-v3 输出维度
    
    VECTOR_SIMILARITY_THRESHOLD: float = 0.3  # 降低阈值，提高召回率
    VECTOR_TOP_K: int = 10
    VECTOR_INDEX_DIR: str = "./data/vector_index"
    
    @property
    def database_url(self) -> str:
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 全局设置实例
settings = Settings()
