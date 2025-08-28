# 使用Python 3.11官方镜像作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止生成.pyc文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2

# 安装系统依赖和MySQL服务器
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    mysql-server \
    mysql-client \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 确保编译后的文件有执行权限
RUN chmod +x *.so app/*.so app/*/*.so scripts/*.so

# 创建必要的目录
RUN mkdir -p uploads data/vector_index /var/run/mysqld

# 设置MySQL数据目录权限
RUN chown -R mysql:mysql /var/lib/mysql /var/run/mysqld

# 设置文件权限
RUN chmod +x start.sh

# 暴露端口
EXPOSE 8000 3306

# 创建MySQL初始化脚本
RUN echo '#!/bin/bash\n\
echo "启动MySQL服务..."\n\
service mysql start\n\
\n\
echo "等待MySQL启动..."\n\
sleep 10\n\
\n\
echo "配置MySQL用户和数据库..."\n\
mysql -e "CREATE DATABASE IF NOT EXISTS \`code-reviewer\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"\n\
mysql -e "UPDATE mysql.user SET authentication_string=PASSWORD('\''19900410'\'') WHERE User='\''root'\'';"\n\
mysql -e "GRANT ALL PRIVILEGES ON *.* TO '\''root'\''@'\''%'\'' IDENTIFIED BY '\''19900410'\'' WITH GRANT OPTION;"\n\
mysql -e "FLUSH PRIVILEGES;"\n\
\n\
echo "MySQL配置完成"\n\
echo "启动代码分析 AI Agent..."\n\
echo "初始化数据库表结构..."\n\
python -c "import scripts.init_db; scripts.init_db.init_database()"\n\
echo "数据库初始化完成"\n\
echo "启动Web服务..."\n\
python -c "import run; run.main()"\n\
' > /app/docker_start.sh && chmod +x /app/docker_start.sh


# 启动命令
CMD ["/app/docker_start.sh"]
