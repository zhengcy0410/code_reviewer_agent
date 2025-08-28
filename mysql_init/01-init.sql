-- MySQL初始化脚本
-- 创建数据库和用户

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS `code-reviewer` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE `code-reviewer`;

-- 设置root用户密码（如果需要）
-- ALTER USER 'root'@'%' IDENTIFIED BY '19900410';

-- 创建专用用户（可选）
-- CREATE USER IF NOT EXISTS 'code_reviewer'@'%' IDENTIFIED BY '20250828';
-- GRANT ALL PRIVILEGES ON `code-reviewer`.* TO 'code_reviewer'@'%';

-- 刷新权限
FLUSH PRIVILEGES;

-- 显示数据库信息
SHOW DATABASES;
SELECT USER(), DATABASE();
