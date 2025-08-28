# -*- coding: utf-8 -*-
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn

from app.core.config import settings
from app.core.database import DatabaseManager, get_db_session
from app.models.response_models import (
    AnalyzeResponse, 
    ExtendedAnalyzeResponse, 
    FeatureAnalysis,
    FunctionInfo,
    FunctionalVerification,
    TestResult
)
from app.utils.file_processor import FileProcessor
from app.services.ckg_service import CKGSystem
from app.core.llm_service import LLMService
from app.services.vector_search_service import vector_search_service

# 创建 FastAPI 应用
app = FastAPI(
    title="代码分析 AI Agent",
    description="代码分析 AI Agent",
    version="1.0.0",
    docs_url="/docs"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务组件
file_processor = FileProcessor()
ckg_system = CKGSystem()
llm_service = LLMService()


@app.post("/analyze-v1", response_model=ExtendedAnalyzeResponse)
async def analyze_code_v1(
    problem_description: str = Form(..., description="功能需求描述"),
    code_zip: UploadFile = File(..., description="包含项目代码的ZIP文件"),
    generate_tests: bool = Form(False, description="是否生成测试代码"),
    execute_tests: bool = Form(False, description="是否执行测试代码")
):
    """方案1：对解压后每个文件和问题都调用一次LLM，然后返回结果"""
    start_time = time.time()
    temp_dir = None
    
    try:
        # 1. 验证输入
        if not problem_description:
            raise HTTPException(
                status_code=400,
                detail="问题描述不能为空"
            )
        
        if not code_zip.filename or not code_zip.filename.endswith('.zip'):
            raise HTTPException(
                status_code=400,
                detail="请上传ZIP格式的代码文件"
            )
        
        # 检查文件大小
        if code_zip.size and code_zip.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # 2. 读取并处理ZIP文件
        zip_content = await code_zip.read()
        
        # 3. 解压和扫描代码文件
        code_files, project_stats, file_hash = await file_processor.process_uploaded_zip(
            zip_content, code_zip.filename
        )
        
        if not code_files:
            raise HTTPException(
                status_code=400,
                detail="ZIP文件中未找到支持的代码文件"
            )
        
        logger.info(f"成功处理 {len(code_files)} 个代码文件")
        
        # 4. 直接对每个文件调用LLM分析，获取完整的功能分析结果
        
        feature_analyses = llm_service.analyze_files_with_llm(code_files, problem_description)        
        
        if not feature_analyses:
            # 如果没有找到相关代码，返回基本信息
            feature_analyses = [FeatureAnalysis(
                feature_description=f"未找到与 '{problem_description}' 直接相关的代码实现",
                implementation_location=[]
            )]
        
        # 5. 生成执行计划
        execution_plan = llm_service.generate_execution_plan(code_files, project_stats)
        
        # 6. 构建基础响应
        response_data = {
            "feature_analysis": feature_analyses,
            "execution_plan_suggestion": execution_plan
        }
        
        # 7. 功能测试
        if generate_tests and feature_analyses and feature_analyses[0].implementation_location:
            try:
                if execute_tests:
                    logger.info("开始生成并执行测试代码...")
                    verification_result = llm_service.generate_and_execute_test(
                        problem_description, 
                        feature_analyses,
                        code_files
                    )
                else:
                    # 仅生成测试代码
                    test_code = llm_service.generate_test_code(problem_description, feature_analyses)
                    verification_result = {
                        "generated_test_code": test_code,
                        "execution_result": {
                            "tests_passed": None,
                            "log": "测试代码已生成，但未执行（execute_tests=False）",
                            "note": "可以设置 execute_tests=true 来执行测试代码"
                        }
                    }
                
                if verification_result["generated_test_code"]:
                    verification = FunctionalVerification(
                        generated_test_code=verification_result["generated_test_code"],
                        execution_result=verification_result["execution_result"]
                    )
                    
                    # 返回扩展响应
                    return ExtendedAnalyzeResponse(
                        **response_data,
                        functional_verification=verification
                    )
                    
            except Exception as e:
                try:
                    test_code = llm_service.generate_test_code(problem_description, feature_analyses)
                    verification = FunctionalVerification(
                        generated_test_code=test_code,
                        execution_result={
                            "tests_passed": False,
                            "log": f"执行测试时出错: {str(e)}",
                            "error": str(e)
                        }
                    )
                    
                    return ExtendedAnalyzeResponse(
                        **response_data,
                        functional_verification=verification
                    )
                except:
                    logger.warning("生成测试代码也失败了")
        
        # 8. 处理时间
        processing_time = time.time() - start_time
        logger.info(f"代码分析完成，耗时: {processing_time:.2f}秒")
        
        return ExtendedAnalyzeResponse(**response_data, functional_verification=None)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/analyze-v2", response_model=ExtendedAnalyzeResponse)
async def analyze_code_v2(
    problem_description: str = Form(..., description="功能需求描述"),
    code_zip: UploadFile = File(..., description="项目代码ZIP文件"),
    generate_tests: bool = Form(False, description="是否生成测试代码"),
    execute_tests: bool = Form(False, description="是否执行测试代码"),
    use_vector_search: bool = Form(False, description="是否启用向量搜索增强")
):
    """方案2：调用LLM获取问题相关的英文关键词，然后模糊查询数据库中能匹配到的函数"""
    start_time = time.time()
    temp_dir = None
    
    try:
        # 1. 验证输入
        if not problem_description:
            raise HTTPException(
                status_code=400,
                detail="问题描述不能为空"
            )
        
        if not code_zip.filename or not code_zip.filename.endswith('.zip'):
            raise HTTPException(
                status_code=400,
                detail="请上传ZIP格式的代码文件"
            )
        
        # 检查文件大小
        if code_zip.size and code_zip.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # 2. 读取并处理ZIP文件
        zip_content = await code_zip.read()
        
        # 3. 解压和扫描代码文件
        code_files, project_stats, file_hash = await file_processor.process_uploaded_zip(
            zip_content, code_zip.filename
        )
        
        if not code_files:
            raise HTTPException(
                status_code=400,
                detail="ZIP文件中未找到支持的代码文件"
            )
        
        
        # 4. 构建代码知识图谱
        project_id = ckg_system.build_ckg(
            code_files, 
            file_hash, 
            f"分析项目_{int(time.time())}", 
            problem_description
        )
        
        # 5. 智能搜索：关键词搜索 + 向量搜索（可选）
        keywords = llm_service.extract_english_keywords(problem_description)
        logger.info(f"LLM提取的关键词: {keywords}")
        
        if use_vector_search:
            # 使用混合搜索
            logger.info("启用混合搜索（关键词 + 向量）")
            functions, classes = ckg_system.hybrid_search_functions(
                project_id, problem_description, keywords
            )
        else:
            # 使用传统关键词搜索
            logger.info("使用传统关键词搜索")
            functions = ckg_system.search_functions(project_id, keywords)
            classes = ckg_system.search_classes(project_id, keywords)
        
        # 6. LLM 分析代码相关性
        feature_analyses = llm_service.analyze_code_relevance(
            problem_description, functions, classes
        )
        
        if not feature_analyses:
            # 如果没有找到相关代码，返回基本信息
            feature_analyses = [FeatureAnalysis(
                feature_description=f"未找到与 '{problem_description}' 直接相关的代码实现",
                implementation_location=[]
            )]
        
        # 7. 生成执行计划
        execution_plan = llm_service.generate_execution_plan(code_files, project_stats)
        
        # # 8. 构建基础响应
        response_data = {
            "feature_analysis": feature_analyses,
            "execution_plan_suggestion": execution_plan
        }
        
        # 9. 功能验证
        if generate_tests and feature_analyses and feature_analyses[0].implementation_location:
            try:
                if execute_tests:
                    # 执行完整的测试生成和执行
                    logger.info("执行完整的测试生成和执行")
                    verification_result = llm_service.generate_and_execute_test(
                        problem_description, 
                        feature_analyses,
                        code_files
                    )
                else:
                    # 仅生成测试代码
                    logger.info("仅生成测试代码")
                    test_code = llm_service.generate_test_code(problem_description, feature_analyses)
                    verification_result = {
                        "generated_test_code": test_code,
                        "execution_result": {
                            "tests_passed": None,
                            "log": "测试代码已生成，但未执行（execute_tests=False）",
                            "note": "可以设置 execute_tests=true 来执行测试代码"
                        }
                    }
                                
                if verification_result["generated_test_code"]:
                    # 确保 execution_result 符合 TestResult 模型
                    exec_result = verification_result["execution_result"]
                    test_result = TestResult(
                        tests_passed=exec_result.get("tests_passed"),
                        log=exec_result.get("log", ""),
                        note=exec_result.get("note")
                    )
                    
                    verification = FunctionalVerification(
                        generated_test_code=verification_result["generated_test_code"],
                        execution_result=test_result
                    )
                    
                    # 返回扩展响应
                    return ExtendedAnalyzeResponse(
                        **response_data,
                        functional_verification=verification
                    )
                else:
                    logger.warning("测试代码为空，不返回functional_verification")
                    
            except Exception as e:
                # 无法执行则仅生成测试代码
                try:
                    test_code = llm_service.generate_test_code(problem_description, feature_analyses)
                    test_result = TestResult(
                        tests_passed=False,
                        log=f"执行测试时出错: {str(e)}",
                        note=f"错误详情: {str(e)}"
                    )
                    verification = FunctionalVerification(
                        generated_test_code=test_code,
                        execution_result=test_result
                    )
                    
                    return ExtendedAnalyzeResponse(
                        **response_data,
                        functional_verification=verification
                    )
                except Exception as inner_e:
                    logger.error(f"生成测试代码也失败了: {str(inner_e)}")
        else:
            pass
        # 10. 处理时间
        processing_time = time.time() - start_time
        logger.info(f"代码分析完成: {processing_time:.2f}秒")
        
        return ExtendedAnalyzeResponse(**response_data, functional_verification=None)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # 清理向量索引（如果使用了向量搜索）
        if use_vector_search and 'project_id' in locals():
            try:
                vector_search_service.clear_project_index(project_id)
            except Exception as e:
                logger.error(f"清理向量索引失败: {e}")


@app.post("/analyze-v3", response_model=ExtendedAnalyzeResponse)
async def analyze_code_v3(
    problem_description: str = Form(..., description="功能需求描述"),
    code_zip: UploadFile = File(..., description="包含项目代码的ZIP文件"),
    generate_tests: bool = Form(False, description="是否生成测试代码"),
    execute_tests: bool = Form(False, description="是否执行测试代码"),
    use_vector_search: bool = Form(False, description="是否启用向量搜索增强")
):
    """方案3：从数据库中查询所有项目的函数名，然后调用LLM来找到跟问题相关的函数"""
    start_time = time.time()
    temp_dir = None
    
    try:
        # 1. 验证输入
        if not problem_description:
            raise HTTPException(
                status_code=400,
                detail="问题描述不能为空"
            )
        
        if not code_zip.filename or not code_zip.filename.endswith('.zip'):
            raise HTTPException(
                status_code=400,
                detail="请上传ZIP格式的代码文件"
            )
        
        # 检查文件大小
        if code_zip.size and code_zip.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # 2. 读取并处理ZIP文件
        zip_content = await code_zip.read()
        
        # 3. 解压和扫描代码文件
        code_files, project_stats, file_hash = await file_processor.process_uploaded_zip(
            zip_content, code_zip.filename
        )
        
        if not code_files:
            raise HTTPException(
                status_code=400,
                detail="ZIP文件中未找到支持的代码文件"
            )
        
        logger.info(f"成功处理 {len(code_files)} 个代码文件")
        
        # 4. 构建代码知识图谱
        project_id = ckg_system.build_ckg(
            code_files, 
            file_hash, 
            f"分析项目_{int(time.time())}", 
            problem_description
        )
        
        # 5. 方案3：LLM筛选 + 向量搜索增强（可选）
        logger.info("开始方案3：LLM筛选 + 向量搜索增强...")
        
        # 获取数据库中所有函数和类的信息
        all_functions = ckg_system.get_all_functions(project_id)
        all_classes = ckg_system.get_all_classes(project_id)
        
        if use_vector_search:
            # LLM筛选 + 向量搜索增强
            logger.info("启用LLM筛选 + 向量搜索增强")
            
            # 1. LLM筛选
            llm_functions, llm_classes = llm_service.filter_relevant_functions(
                problem_description, all_functions, all_classes
            )
            
            # 2. 向量搜索
            vector_functions, vector_classes = ckg_system.vector_search_functions(
                project_id, problem_description
            )
            
            # 3. 合并结果
            functions, classes = ckg_system.merge_search_results(
                llm_functions, llm_classes,
                vector_functions, vector_classes
            )
        else:
            # 仅使用LLM筛选
            logger.info("使用传统LLM筛选")
            functions, classes = llm_service.filter_relevant_functions(
                problem_description, all_functions, all_classes
            )
        # 6. LLM 分析代码相关性
        feature_analyses = llm_service.analyze_code_relevance(
            problem_description, functions, classes
        )
        
        if not feature_analyses:
            # 如果没有找到相关代码，返回基本信息
            feature_analyses = [FeatureAnalysis(
                feature_description=f"未找到与 '{problem_description}' 直接相关的代码实现",
                implementation_location=[]
            )]
        
        # 7. 生成执行计划
        execution_plan = llm_service.generate_execution_plan(code_files, project_stats)
        
        # 8. 构建基础响应
        response_data = {
            "feature_analysis": feature_analyses,
            "execution_plan_suggestion": execution_plan
        }
        
        # 9. 如果需要功能验证
        if generate_tests and feature_analyses and feature_analyses[0].implementation_location:
            try:
                if execute_tests:
                    # 执行完整的测试生成和执行
                    logger.info("开始生成并执行测试代码...")
                    verification_result = llm_service.generate_and_execute_test(
                        problem_description, 
                        feature_analyses,
                        code_files
                    )
                else:
                    # 仅生成测试代码
                    test_code = llm_service.generate_test_code(problem_description, feature_analyses)
                    verification_result = {
                        "generated_test_code": test_code,
                        "execution_result": {
                            "tests_passed": None,
                            "log": "测试代码已生成，但未执行（execute_tests=False）",
                            "note": "可以设置 execute_tests=true 来执行测试代码"
                        }
                    }
                
                if verification_result["generated_test_code"]:
                    verification = FunctionalVerification(
                        generated_test_code=verification_result["generated_test_code"],
                        execution_result=verification_result["execution_result"]
                    )
                    
                    # 返回扩展响应
                    return ExtendedAnalyzeResponse(
                        **response_data,
                        functional_verification=verification
                    )
                    
            except Exception as e:
                logger.error(f"功能验证时出错: {e}")
                # 降级到仅生成测试代码
                try:
                    test_code = llm_service.generate_test_code(problem_description, feature_analyses)
                    verification = FunctionalVerification(
                        generated_test_code=test_code,
                        execution_result={
                            "tests_passed": False,
                            "log": f"执行测试时出错: {str(e)}",
                            "error": str(e)
                        }
                    )
                    
                    return ExtendedAnalyzeResponse(
                        **response_data,
                        functional_verification=verification
                    )
                except:
                    logger.warning("生成测试代码也失败了")
        
        # 10. 记录处理时间
        processing_time = time.time() - start_time
        logger.info(f"代码分析完成，耗时: {processing_time:.2f}秒")
        
        return ExtendedAnalyzeResponse(**response_data, functional_verification=None)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"代码分析过程中出错: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # 清理向量索引（如果使用了向量搜索）
        if use_vector_search and 'project_id' in locals():
            try:
                vector_search_service.clear_project_index(project_id)
            except Exception as e:
                logger.error(f"清理向量索引失败: {e}")


@app.post("/analyze-v4", response_model=ExtendedAnalyzeResponse)
async def analyze_code_v4(
    problem_description: str = Form(..., description="功能需求描述"),
    code_zip: UploadFile = File(..., description="包含项目代码的ZIP文件"),
    generate_tests: bool = Form(False, description="是否生成测试代码"),
    execute_tests: bool = Form(False, description="是否执行测试代码")
):
    """方案4：纯向量搜索方案"""
    start_time = time.time()
    temp_dir = None
    
    try:
        # 1. 验证输入
        if not problem_description:
            raise HTTPException(
                status_code=400,
                detail="问题描述不能为空"
            )
        
        if not code_zip.filename or not code_zip.filename.endswith('.zip'):
            raise HTTPException(
                status_code=400,
                detail="请上传ZIP格式的代码文件"
            )
        
        # 检查文件大小
        if code_zip.size and code_zip.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # 2. 读取并处理ZIP文件
        zip_content = await code_zip.read()
        
        # 3. 解压和扫描代码文件
        code_files, project_stats, file_hash = await file_processor.process_uploaded_zip(
            zip_content, code_zip.filename
        )
        
        if not code_files:
            raise HTTPException(
                status_code=400,
                detail="ZIP文件中未找到支持的代码文件"
            )
        
        logger.info(f"成功处理 {len(code_files)} 个代码文件")
        
        # 4. 构建代码知识图谱
        project_id = ckg_system.build_ckg(
            code_files, 
            file_hash, 
            f"分析项目_{int(time.time())}", 
            problem_description
        )
        
        # 5. 纯向量搜索
        logger.info("使用纯向量搜索")
        functions, classes = ckg_system.vector_search_functions(project_id, problem_description)
        
        # 6. LLM 分析代码相关性
        feature_analyses = llm_service.analyze_code_relevance(
            problem_description, functions, classes
        )
        
        if not feature_analyses:
            # 如果没有找到相关代码，返回基本信息
            feature_analyses = [FeatureAnalysis(
                feature_description=f"未找到与 '{problem_description}' 直接相关的代码实现",
                implementation_location=[]
            )]
        
        # 7. 生成执行计划
        execution_plan = llm_service.generate_execution_plan(code_files, project_stats)
        
        # 8. 构建基础响应
        response_data = {
            "feature_analysis": feature_analyses,
            "execution_plan_suggestion": execution_plan
        }
        
        # 9. 功能验证
        if generate_tests and feature_analyses and feature_analyses[0].implementation_location:
            try:
                if execute_tests:
                    # 执行完整的测试生成和执行
                    logger.info("执行完整的测试生成和执行")
                    verification_result = llm_service.generate_and_execute_test(
                        problem_description, 
                        feature_analyses,
                        code_files
                    )
                else:
                    # 仅生成测试代码
                    logger.info("仅生成测试代码")
                    test_code = llm_service.generate_test_code(problem_description, feature_analyses)
                    verification_result = {
                        "generated_test_code": test_code,
                        "execution_result": {
                            "tests_passed": None,
                            "log": "测试代码已生成，但未执行（execute_tests=False）",
                            "note": "可以设置 execute_tests=true 来执行测试代码"
                        }
                    }
                                
                if verification_result["generated_test_code"]:
                    # 确保 execution_result 符合 TestResult 模型
                    exec_result = verification_result["execution_result"]
                    test_result = TestResult(
                        tests_passed=exec_result.get("tests_passed"),
                        log=exec_result.get("log", ""),
                        note=exec_result.get("note")
                    )
                    
                    verification = FunctionalVerification(
                        generated_test_code=verification_result["generated_test_code"],
                        execution_result=test_result
                    )
                    
                    # 返回扩展响应
                    return ExtendedAnalyzeResponse(
                        **response_data,
                        functional_verification=verification
                    )
                else:
                    logger.warning("测试代码为空，不返回functional_verification")
                    
            except Exception as e:
                # 无法执行则仅生成测试代码
                try:
                    test_code = llm_service.generate_test_code(problem_description, feature_analyses)
                    test_result = TestResult(
                        tests_passed=False,
                        log=f"执行测试时出错: {str(e)}",
                        note=f"错误详情: {str(e)}"
                    )
                    verification = FunctionalVerification(
                        generated_test_code=test_code,
                        execution_result=test_result
                    )
                    
                    return ExtendedAnalyzeResponse(
                        **response_data,
                        functional_verification=verification
                    )
                except Exception as inner_e:
                    logger.error(f"生成测试代码也失败了: {str(inner_e)}")
        
        # 10. 处理时间
        processing_time = time.time() - start_time
        logger.info(f"代码分析完成: {processing_time:.2f}秒")
        
        return ExtendedAnalyzeResponse(**response_data, functional_verification=None)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"代码分析过程中出错: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # 清理向量索引（方案4总是使用向量搜索）
        if 'project_id' in locals():
            try:
                vector_search_service.clear_project_index(project_id)
            except Exception as e:
                logger.error(f"清理向量索引失败: {e}")


# def cleanup_project_data(project_id: int):
#     """后台清理项目数据"""
#     try:
#         # 延迟清理，给用户一些时间查看结果
#         import time
#         time.sleep(300)  # 5分钟后清理
        
#         ckg_system.cleanup_project(project_id)
#         logger.info(f"已清理项目 {project_id} 的数据")
#     except Exception as e:
#         logger.error(f"清理项目数据失败: {e}")

# 异常处理器
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "请求的资源不存在", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"内部服务器错误: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "内部服务器错误，请稍后重试"}
    )

if __name__ == "__main__":
    # 配置日志
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        encoding="utf-8"
    )
    
    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)
    
    # 启动应用
    uvicorn.run(
        "main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
