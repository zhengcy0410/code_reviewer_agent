
import json
from typing import List, Dict, Any, Optional
from loguru import logger
import dashscope
from dashscope import Generation
import openai
from app.core.config import settings
from app.models.response_models import FeatureAnalysis, FunctionInfo
# 设置 API Keys
OPENAI_AVAILABLE = True
DASHSCOPE_AVAILABLE = True
dashscope.api_key = settings.DASHSCOPE_API_KEY
openai.api_key = settings.OPENAI_API_KEY

class LLMService:
    """LLM 分析服务 - 支持多种模型提供商"""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        
        if self.provider == "openai":
            self.model = "gpt-3.5-turbo"  
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("使用 OpenAI ChatGPT 模型")
                
        if self.provider == "qwen":
            self.model = "qwen-max"
            self.provider = "qwen"
            logger.info("使用通义千问模型")
    
    def _call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """调用 LLM API - 支持多个提供商"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            if self.provider == "openai":
                return self._call_openai(messages)
            else:
                return self._call_qwen(messages)
                
        except Exception as e:
            logger.error(f"调用 LLM 时出错: {e}")
            return ""
    
    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """调用 OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_tokens=4000,
                timeout=30  
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.error("OpenAI API 返回空响应")
                return ""
                
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            return ""
    
    def _call_qwen(self, messages: List[Dict[str, str]]) -> str:
        """调用通义千问 API"""
        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_tokens=4000,
                result_format='message'
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                logger.error(f"通义千问 API 调用失败: {response.message}")
                return ""
                
        except Exception as e:
            logger.error(f"通义千问 API 调用失败: {e}")
            return ""
    
    def extract_function_names_from_description(self, problem_description: str) -> List[str]:
        """从问题描述中提取可能的函数名 - 基于trae-agent-main的精确搜索方法"""
        system_prompt = """你是一个专业的代码分析助手。你的任务是从用户的问题描述中提取可能的函数名或类名，这些名称将用于在代码库中进行精确搜索。

请遵循以下规则：
1. 提取可能的函数名（如multiply, calculate_product等）
2. 提取可能的类名（如Calculator, MathUtils等）  
3. 基于功能描述推断常见的函数命名模式
4. 返回JSON格式的名称列表

示例：
输入："帮我查找乘积功能的位置"
输出：["multiply", "product", "calculate_product", "mul", "times"]

输入："查找计算器类的加法方法"
输出：["Calculator", "add", "plus", "addition", "sum"]
"""
        
        prompt = f"""请从以下问题描述中提取可能的函数名和类名：

问题描述：{problem_description}

请返回JSON格式的名称列表："""
        
        response = self._call_llm(prompt, system_prompt)
        
        try:
            # 尝试解析JSON响应
            function_names = json.loads(response)
            if isinstance(function_names, list):
                logger.info(f"LLM提取的函数名: {function_names}")
                return function_names
            else:
                # 如果不是列表，使用备用方法
                return self._extract_function_names_fallback(problem_description)
        except json.JSONDecodeError:
            logger.warning("LLM 返回的不是有效JSON，使用备用方法")
            return self._extract_function_names_fallback(problem_description)
    
    def _extract_function_names_fallback(self, problem_description: str) -> List[str]:
        """备用函数名提取方法 - 基于关键词映射到可能的函数名"""
        import re
        
        # 提取可能的函数名模式
        function_patterns = [
            r'函数\s*[：:]\s*(\w+)',  # "函数：multiply"
            r'方法\s*[：:]\s*(\w+)',  # "方法：calculate"
            r'`(\w+)`',               # "`multiply`"
            r'(\w+)\s*函数',          # "multiply函数"
            r'(\w+)\s*方法',          # "calculate方法"
            r'调用\s*(\w+)',          # "调用multiply"
            r'执行\s*(\w+)',          # "执行calculate"
            r'(\w+)\s*功能',          # "multiply功能"
        ]
        
        extracted_names = set()
        
        # 使用正则表达式提取
        for pattern in function_patterns:
            matches = re.findall(pattern, problem_description, re.IGNORECASE)
            extracted_names.update(matches)
        
        # 基于关键词映射到常见函数名
        keyword_to_functions = {
            '乘积': ['multiply', 'product', 'mul'],
            '加法': ['add', 'plus', 'sum'],
            '减法': ['subtract', 'minus', 'sub'],
            '除法': ['divide', 'div'],
            '登录': ['login', 'signin', 'authenticate'],
            '注册': ['register', 'signup'],
            '上传': ['upload'],
            '下载': ['download'],
            '删除': ['delete', 'remove'],
            '创建': ['create', 'add', 'insert'],
            '更新': ['update', 'modify'],
            '查询': ['query', 'search', 'find', 'get'],
            '计算': ['calculate', 'compute', 'calc'],
            '发送': ['send', 'post'],
            '接收': ['receive', 'get'],
            '验证': ['validate', 'verify', 'check']
        }
        
        for keyword, function_names in keyword_to_functions.items():
            if keyword in problem_description:
                extracted_names.update(function_names)
        
        # 过滤掉过短的名称
        result = [name for name in extracted_names if len(name) >= 3]
        logger.info(f"备用方法提取的函数名: {result}")
        return result
    
    def analyze_code_relevance(self, problem_description: str, functions: List[Dict], 
                              classes: List[Dict]) -> List[FeatureAnalysis]:
        """分析代码与问题的相关性"""
        if not functions and not classes:
            return []
        
        # 构建代码信息
        code_info = self._build_code_info(functions, classes)
        
        system_prompt = """你是一个专业的代码分析师。你的任务是分析给定的代码片段，找出与用户需求最相关的实现位置。

        请遵循以下规则：
        1. 仔细分析用户的需求描述
        2. 识别代码中实现该功能的关键函数和类
        3. 按照相关性排序，最相关的排在前面
        4. 为每个功能提供清晰的描述
        5. 返回标准的JSON格式

        输出格式示例：
        {
        "feature_analysis": [
            {
            "feature_description": "实现乘积计算功能",
            "implementation_location": [
                {
                "file": "math_utils.py",
                "function": "multiply",
                "lines": "15-20"
                }
            ]
            }
        ]
        }"""
        
        prompt = f"""用户需求：{problem_description}

        代码信息：
        {code_info}

        请分析上述代码，找出实现用户需求的关键位置，并返回JSON格式的分析结果："""
        
        response = self._call_llm(prompt, system_prompt)
        
        try:
            result = json.loads(response)
            if "feature_analysis" in result and isinstance(result["feature_analysis"], list):
                feature_analyses = []
                for item in result["feature_analysis"]:
                    if "feature_description" in item and "implementation_location" in item:
                        locations = []
                        for loc in item["implementation_location"]:
                            if all(key in loc for key in ["file", "function", "lines"]):
                                code_content = ""
                                
                                for func in functions:
                                    if func.get("name") == loc["function"]:
                                        if (func.get("file") == loc["file"] or 
                                            loc["file"] in ["unknown", ""] or
                                            func.get("file") == "" or
                                            func.get("file") is None):
                                            code_content = func.get("body", "")
                                            break
                                
                                # 如果在functions中没找到，再在classes中查找
                                if not code_content:
                                    for cls in classes:
                                        if cls.get("name") == loc["function"]:
                                            if (cls.get("file") == loc["file"] or 
                                                loc["file"] in ["unknown", ""] or
                                                cls.get("file") == "" or
                                                cls.get("file") is None):
                                                code_content = cls.get("body", "")
                                                break
                                
                                locations.append(FunctionInfo(
                                    file=loc["file"],
                                    function=loc["function"],
                                    lines=loc["lines"],
                                    code_content=code_content  
                                ))
                        
                        if locations:
                            feature_analyses.append(FeatureAnalysis(
                                feature_description=item["feature_description"],
                                implementation_location=locations
                            ))
                
                return feature_analyses
            else:
                return self._analyze_code_fallback(problem_description, functions, classes)
                
        except json.JSONDecodeError:
            return self._analyze_code_fallback(problem_description, functions, classes)
    
    def _build_code_info(self, functions: List[Dict], classes: List[Dict]) -> str:
        """构建代码信息字符串"""
        info_parts = []
        
        # 添加函数信息
        if functions:
            info_parts.append("=== 函数列表 ===")
            for func in functions[:20]:  # 限制数量避免过长
                # 安全获取字符串字段
                file_path = str(func.get('file_path', 'unknown'))
                name = str(func.get('name', 'unknown'))
                start_line = str(func.get('start_line', 'unknown'))
                end_line = str(func.get('end_line', 'unknown'))
                
                info_parts.append(f"文件: {file_path}")
                info_parts.append(f"函数: {name} (行 {start_line}-{end_line})")
                
                if func.get('parent_class'):
                    parent_class = str(func['parent_class'])
                    info_parts.append(f"所属类: {parent_class}")
                
                if func.get('body'):
                    body = func['body']
                    if isinstance(body, list):
                        body = '\n'.join(str(line) for line in body)
                    else:
                        body = str(body)
                    
                    body_lines = body.split('\n')[:5]
                    info_parts.append("代码预览:")
                    for line in body_lines:
                        info_parts.append(f"  {str(line)}")
                info_parts.append("")
        
        # 添加类信息
        if classes:
            info_parts.append("=== 类列表 ===")
            for cls in classes[:10]:  
                file_path = str(cls.get('file_path', 'unknown'))
                name = str(cls.get('name', 'unknown'))
                start_line = str(cls.get('start_line', 'unknown'))
                end_line = str(cls.get('end_line', 'unknown'))
                
                info_parts.append(f"文件: {file_path}")
                info_parts.append(f"类: {name} (行 {start_line}-{end_line})")
                
                if cls.get('methods'):
                    methods = cls['methods']
                    if isinstance(methods, list):
                        methods = '\n'.join(str(method) for method in methods)
                    else:
                        methods = str(methods)
                    info_parts.append("方法:")
                    info_parts.append(methods)
                info_parts.append("")
        
        return '\n'.join(info_parts)
    
    def _analyze_code_fallback(self, problem_description: str, functions: List[Dict], 
                              classes: List[Dict]) -> List[FeatureAnalysis]:
        """备用代码分析方法"""
        if functions or classes:
            locations = []
            
            # 处理函数结果
            for func in functions[:5]:  
                locations.append(FunctionInfo(
                    file=func.get('file', func.get('file_path', '')),
                    function=func.get('name', ''),
                    lines=f"{func.get('start_line', 0)}-{func.get('end_line', 0)}",
                    code_content=func.get('body', '')  # 添加code_content
                ))
            
            # 处理类结果
            for cls in classes[:3]:  
                locations.append(FunctionInfo(
                    file=cls.get('file', cls.get('file_path', '')),
                    function=cls.get('name', ''),
                    lines=f"{cls.get('start_line', 0)}-{cls.get('end_line', 0)}",
                    code_content=cls.get('body', '')  # 添加code_content
                ))
            
            if locations:
                return [FeatureAnalysis(
                    feature_description=f"找到与 '{problem_description}' 相关的代码实现",
                    implementation_location=locations
                )]
        
        logger.warning("未找到相关的代码实现")
        return []
    
    def generate_execution_plan(self, code_files: List[Dict[str, str]], 
                               project_stats: Dict[str, int]) -> str:
        """生成执行计划建议"""
        system_prompt = """你是一个项目架构专家。基于提供的代码文件信息，生成一个清晰的项目执行计划。

        请包含以下内容：
        1. 依赖安装命令
        2. 项目启动方式
        3. 主要入口文件
        4. API访问地址（如果适用）
        5. 其他重要说明

        保持简洁明了，适合开发者快速上手。"""
        
        # 分析项目类型
        languages = project_stats.get('languages', {})
        has_python = 'python' in languages
        has_js = any(lang in languages for lang in ['javascript', 'typescript'])
        has_java = 'java' in languages
        
        # 查找关键文件
        key_files = []
        for file_info in code_files[:10]:  # 只分析前10个文件
            file_path = file_info['file_path'].lower()
            if any(name in file_path for name in ['main', 'app', 'server', 'index', 'run']):
                key_files.append(file_info['file_path'])
        
        prompt = f"""请为以下项目生成执行计划：

        项目统计信息：
        - 总文件数: {project_stats.get('total_files', 0)}
        - 总代码行数: {project_stats.get('total_lines', 0)}
        - 编程语言分布: {languages}

        关键文件: {', '.join(key_files) if key_files else '无'}

        请生成简洁的执行计划建议："""
        
        response = self._call_llm(prompt, system_prompt)
        
        if not response:
            # 备用执行计划
            if has_python:
                return "要执行此项目，请首先运行 `pip install -r requirements.txt` 安装依赖，然后执行 `python main.py` 或 `python app.py` 启动项目。"
            elif has_js:
                return "要执行此项目，请首先运行 `npm install` 安装依赖，然后执行 `npm start` 或 `node index.js` 启动项目。"
            elif has_java:
                return "要执行此项目，请确保已安装 JDK，然后使用 `javac *.java` 编译源文件，最后使用 `java Main` 运行项目。"
            else:
                return "请根据项目的具体语言和框架，安装相应的依赖并启动项目。"
        
        return response
    
    def generate_test_code(self, problem_description: str, feature_analysis: List[FeatureAnalysis]) -> Optional[str]:
        """生成测试代码"""
        if not feature_analysis:
            return None
        
        # 首先检测函数的编程语言
        detected_language = self._detect_function_language(feature_analysis)
        logger.info(f"检测到的编程语言: {detected_language}")
        
        # 根据语言生成相应的系统提示词
        system_prompt = self._get_language_specific_system_prompt(detected_language)
        
        # 检查是否有代码内容，如果没有则尝试从文件中获取
        enhanced_analysis = self._enhance_analysis_with_code(feature_analysis)
        
        analysis_info = []
        total_functions = 0
        
        for analysis in enhanced_analysis:
            analysis_info.append(f"=== 功能: {analysis.feature_description} ===")
            
            for loc in analysis.implementation_location:
                total_functions += 1
                analysis_info.append(f"文件: {loc.file}")
                analysis_info.append(f"函数: {loc.function} (行 {loc.lines})")
                
                if hasattr(loc, 'code_content') and loc.code_content:
                    analysis_info.append("完整代码:")
                    analysis_info.append("```")
                    analysis_info.append(loc.code_content)
                    analysis_info.append("```")
                else:
                    analysis_info.append(f"警告: 未找到完整代码内容，仅有位置信息")
                    analysis_info.append(f"请确保函数 {loc.function} 在文件 {loc.file} 的第 {loc.lines} 行")
                
                analysis_info.append("")  
        
        # 根据语言生成相应的导入信息和用户提示词
        import_info, user_prompt_template = self._get_language_specific_prompt_template(detected_language, enhanced_analysis)
        
        prompt = user_prompt_template.format(
            problem_description=problem_description,
            total_functions=total_functions,
            analysis_info=chr(10).join(analysis_info),
            import_info=import_info
        )
        
        response = self._call_llm(prompt, system_prompt)
        return response if response else None
    
    def _enhance_analysis_with_code(self, feature_analysis: List[FeatureAnalysis]) -> List[FeatureAnalysis]:
        """增强分析结果"""
        enhanced_analyses = []
        
        for analysis in feature_analysis:
            enhanced_locations = []
            
            for loc in analysis.implementation_location:
                code_content = loc.code_content
                if code_content and code_content.strip():
                    enhanced_locations.append(loc)
                else:
                    logger.warning(f"函数 {loc.function} 在文件 {loc.file} 缺少完整代码内容")
                    # 确保所有字段都有值
                    enhanced_loc = FunctionInfo(
                        file=loc.file,
                        function=loc.function,
                        lines=loc.lines,
                        code_content=code_content or ""  
                    )
                    enhanced_locations.append(enhanced_loc)
            
            enhanced_analyses.append(FeatureAnalysis(
                feature_description=analysis.feature_description,
                implementation_location=enhanced_locations
            ))
        
        return enhanced_analyses
    
    def generate_and_execute_test(self, 
                                problem_description: str, 
                                feature_analysis: List[FeatureAnalysis],
                                project_files: List[Dict[str, str]]) -> Dict[str, Any]:
        """生成并执行测试代码"""
        from app.services.test_executor import SafeTestExecutor
        
        # 1. 检测函数的编程语言
        language = self._detect_function_language(feature_analysis)
        logger.info(f"检测到的编程语言: {language}")
        
        # 2. 生成测试代码（现在会根据语言生成相应的测试代码）
        test_code = self.generate_test_code(problem_description, feature_analysis)
        if not test_code:
            return {
                "generated_test_code": None,
                "execution_result": {
                    "tests_passed": False,
                    "log": "未能生成测试代码",
                    "error": "测试代码生成失败"
                }
            }
        
        # 3. 清理测试代码，去除markdown格式标记
        test_code = self._clean_test_code(test_code)
        
        # 4. 执行测试代码
        executor = SafeTestExecutor()
        execution_result = executor.execute_test_code(test_code, project_files, language)
        
        return {
            "generated_test_code": test_code,
            "execution_result": execution_result
        }
    
    def _clean_test_code(self, test_code: str) -> str:
        """清理测试代码，去除markdown格式标记"""
        if not test_code:
            return test_code
        import re
        
        # 去除开头的```python, ```javascript等标记
        test_code = re.sub(r'^```\w*\n', '', test_code, flags=re.MULTILINE)
        
        # 去除结尾的```标记
        test_code = re.sub(r'\n```$', '', test_code, flags=re.MULTILINE)
        test_code = test_code.strip()
        
        return test_code
    
    def _detect_main_language(self, project_files: List[Dict[str, str]]) -> str:
        """检测项目的主要编程语言"""
        language_count = {"python": 0, "javascript": 0, "other": 0}
        
        for file_info in project_files:
            file_path = file_info.get('file_path', '')
            if file_path.endswith('.py'):
                language_count["python"] += 1
            elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                language_count["javascript"] += 1
            else:
                language_count["other"] += 1
        
        # 返回文件数最多的语言
        return max(language_count, key=language_count.get) if max(language_count.values()) > 0 else "python"
    
    def _detect_function_language(self, feature_analysis: List[FeatureAnalysis]) -> str:
        """检测分析到的函数的编程语言"""
        language_count = {"python": 0, "javascript": 0, "other": 0}
        
        for analysis in feature_analysis:
            for loc in analysis.implementation_location:
                file_path = loc.file.lower()
                if file_path.endswith('.py'):
                    language_count["python"] += 1
                elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                    language_count["javascript"] += 1
                else:
                    language_count["other"] += 1
        
        return max(language_count, key=language_count.get) if max(language_count.values()) > 0 else "python"
    
    def analyze_files_with_llm(self, code_files: List[Dict[str, str]], problem_description: str) -> List[FeatureAnalysis]:
        """方案1：对每个文件和问题都调用一次LLM，直接返回FeatureAnalysis格式的结果"""
        system_prompt = """你是一个专业的代码分析师。你的任务是分析给定的代码文件，找出与用户需求相关的函数和类，并提供完整的代码内容用于生成测试。

        请遵循以下规则：
        1. 仔细分析每个代码文件的内容
        2. 识别与用户需求相关的函数和类
        3. 返回标准的JSON格式，包含功能描述和实现位置的详细信息
        4. **重要**: 必须在code_content字段中包含完整的函数代码，从函数定义开始到函数结束
        5. 如果没有找到相关内容，返回空的feature_analysis数组
        6. 确保code_content字段包含完整可执行的函数代码

        输出格式：
        {
        "feature_analysis": [
            {
            "feature_description": "具体功能描述",
            "implementation_location": [
                {
                "file": "path/to/file.py",
                "function": "function_name", 
                "lines": "10-20",
                "code_content": "def function_name(param1, param2):\n    # 完整的函数实现代码\n    return result"
                }
            ]
            }
        ]
        }

        注意：code_content字段是最重要的，必须包含从def开始到函数结束的完整代码。"""
        
        all_feature_analyses = []
        
        # 对每个文件调用LLM分析
        for file_info in code_files:
            file_path = file_info['file_path']
            content = file_info['content']
            
            if len(content) > 4000:
                content = content[:4000] + "...(内容截断)"
            
            prompt = f"""用户需求：{problem_description}

            代码文件：{file_path}
            文件内容：
            {content}

            请分析上述代码文件，找出与用户需求相关的函数和类，并返回JSON格式的功能分析结果："""
            
            response = self._call_llm(prompt, system_prompt)
            
            try:
                result = json.loads(response)
                if "feature_analysis" in result and isinstance(result["feature_analysis"], list):
                    for feature in result["feature_analysis"]:
                        if "feature_description" in feature and "implementation_location" in feature:
                            locations = []
                            for loc in feature["implementation_location"]:
                                if all(key in loc for key in ["file", "function", "lines"]):
                                    location_info = FunctionInfo(
                                        file=loc["file"],
                                        function=loc["function"],
                                        lines=loc["lines"],
                                        code_content=loc.get('code_content', '')  # 只保留code_content，移除parameters和return_type
                                    )
                                    locations.append(location_info)
                            
                            if locations:
                                all_feature_analyses.append(FeatureAnalysis(
                                    feature_description=feature["feature_description"],
                                    implementation_location=locations
                                ))
                            
            except json.JSONDecodeError as e:
                # backup_analysis = self._analyze_file_backup_v2(file_path, content, problem_description)
                # if backup_analysis:
                #     all_feature_analyses.extend(backup_analysis)
                continue
        
        logger.info(f"方案1 LLM分析完成，找到 {len(all_feature_analyses)} 个功能分析")
        return all_feature_analyses
    
    # def _analyze_file_backup_v2(self, file_path: str, content: str, problem_description: str) -> List[FeatureAnalysis]:
    #     """备用文件分析方法v2 - 直接返回FeatureAnalysis格式"""
    #     import re
        
    #     feature_analyses = []
        
    #     # 更精确的正则匹配来提取函数
    #     if file_path.endswith('.py'):
    #         # 改进的Python函数匹配，能更好地处理缩进和函数体
    #         lines = content.split('\n')
    #         i = 0
    #         while i < len(lines):
    #             line = lines[i]
    #             # 匹配函数定义
    #             func_match = re.match(r'^(\s*)def\s+(\w+)\s*\([^)]*\):', line)
    #             if func_match:
    #                 func_name = func_match.group(2)
    #                 func_indent = len(func_match.group(1))
                    
    #                 # 检查是否与问题相关
    #                 if self._is_relevant_to_problem(func_name, problem_description):
    #                     start_line = i + 1
    #                     func_lines = [line]
                        
    #                     # 提取完整函数体
    #                     j = i + 1
    #                     while j < len(lines):
    #                         current_line = lines[j]
    #                         # 如果是空行或注释，继续
    #                         if not current_line.strip() or current_line.strip().startswith('#'):
    #                             func_lines.append(current_line)
    #                             j += 1
    #                             continue
                            
    #                         # 如果缩进级别回到函数定义级别或更少，函数结束
    #                         current_indent = len(current_line) - len(current_line.lstrip())
    #                         if current_line.strip() and current_indent <= func_indent:
    #                             break
                                
    #                         func_lines.append(current_line)
    #                         j += 1
                        
    #                     end_line = j
    #                     func_code = '\n'.join(func_lines)
                        
    #                     # 提取参数信息
    #                     param_match = re.search(r'def\s+\w+\s*\(([^)]*)\):', line)
    #                     parameters = param_match.group(1) if param_match else ""
                        
    #                     location_info = FunctionInfo(
    #                         file=file_path,
    #                         function=func_name,
    #                         lines=f"{start_line}-{end_line}",
    #                         code_content=func_code,
    #                         parameters=f"参数: {parameters}" if parameters else "无参数",
    #                         return_type="从代码分析推断的返回类型"
    #                     )
                        
    #                     feature_analyses.append(FeatureAnalysis(
    #                         feature_description=f"找到与 '{problem_description}' 相关的函数: {func_name}",
    #                         implementation_location=[location_info]
    #                     ))
                        
    #                     i = j  # 跳过已处理的行
    #                     continue
                
    #             i += 1
        
    #     elif file_path.endswith('.js'):
    #         # 匹配JavaScript函数定义
    #         func_pattern = r'(function\s+(\w+)\s*\([^)]*\)\s*{[^}]*})'
    #         func_matches = re.finditer(func_pattern, content, re.MULTILINE | re.DOTALL)
            
    #         for match in func_matches:
    #             func_name = match.group(2)
    #             func_code = match.group(1).strip()
    #             start_line = content[:match.start()].count('\n') + 1
    #             end_line = start_line + func_code.count('\n')
                
    #             if self._is_relevant_to_problem(func_name, problem_description):
    #                 location_info = FunctionInfo(
    #                     file=file_path,
    #                     function=func_name,
    #                     lines=f"{start_line}-{end_line}",
    #                     code_content=func_code,
    #                     parameters="从代码中提取的参数信息",
    #                     return_type="推断的返回类型"
    #                 )
                    
    #                 feature_analyses.append(FeatureAnalysis(
    #                     feature_description=f"找到与 '{problem_description}' 相关的函数: {func_name}",
    #                     implementation_location=[location_info]
    #                 ))
        
    #     return feature_analyses
    
    # def _analyze_file_backup(self, file_path: str, content: str, problem_description: str) -> tuple[List[Dict], List[Dict]]:
    #     """备用文件分析方法 - 当LLM返回格式错误时使用"""
    #     functions = []
    #     classes = []
        
    #     # 简单的正则匹配来提取函数和类
    #     import re
        
    #     # 提取Python函数
    #     if file_path.endswith('.py'):
    #         # 匹配函数定义
    #         func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
    #         func_matches = re.finditer(func_pattern, content, re.MULTILINE)
            
    #         for match in func_matches:
    #             func_name = match.group(1)
    #             start_line = content[:match.start()].count('\n') + 1
                
    #             # 检查是否与问题相关
    #             if self._is_relevant_to_problem(func_name, problem_description):
    #                 functions.append({
    #                     'name': func_name,
    #                     'file_path': file_path,
    #                     'start_line': str(start_line),
    #                     'end_line': str(start_line + 5),  # 估算
    #                     'body': f"def {func_name}(...):",
    #                     'parent_class': None
    #                 })
            
    #         # 匹配类定义
    #         class_pattern = r'class\s+(\w+)\s*(?:\([^)]*\))?:'
    #         class_matches = re.finditer(class_pattern, content, re.MULTILINE)
            
    #         for match in class_matches:
    #             class_name = match.group(1)
    #             start_line = content[:match.start()].count('\n') + 1
                
    #             if self._is_relevant_to_problem(class_name, problem_description):
    #                 classes.append({
    #                     'name': class_name,
    #                     'file_path': file_path,
    #                     'start_line': str(start_line),
    #                     'end_line': str(start_line + 10),  # 估算
    #                     'body': f"class {class_name}:",
    #                     'methods': ""
    #                 })
        
    #     # 提取JavaScript函数
    #     elif file_path.endswith('.js'):
    #         # 匹配函数定义
    #         func_pattern = r'function\s+(\w+)\s*\([^)]*\)'
    #         func_matches = re.finditer(func_pattern, content, re.MULTILINE)
            
    #         for match in func_matches:
    #             func_name = match.group(1)
    #             start_line = content[:match.start()].count('\n') + 1
                
    #             if self._is_relevant_to_problem(func_name, problem_description):
    #                 functions.append({
    #                     'name': func_name,
    #                     'file_path': file_path,
    #                     'start_line': str(start_line),
    #                     'end_line': str(start_line + 5),
    #                     'body': f"function {func_name}(...)",
    #                     'parent_class': None
    #                 })
        
    #     return functions, classes
    
    def _is_relevant_to_problem(self, identifier: str, problem_description: str) -> bool:
        """判断标识符是否与问题相关"""
        # 简单的相关性检查
        keywords = ['乘积', 'multiply', 'product', 'calculate', 'compute']
        
        identifier_lower = identifier.lower()
        problem_lower = problem_description.lower()
        
        # 检查标识符是否包含相关关键词
        for keyword in keywords:
            if keyword.lower() in identifier_lower or keyword.lower() in problem_lower:
                if keyword.lower() in identifier_lower:
                    return True
        
        return False
    
    def extract_english_keywords(self, problem_description: str) -> List[str]:
        """方案2：调用LLM获取问题相关的英文关键词"""
        system_prompt = """你是一个专业的代码搜索助手。你的任务是从用户的中文问题描述中提取相关的英文关键词，这些关键词将用于在代码库中进行模糊搜索。

        请遵循以下规则：
        1. 提取与功能相关的英文关键词
        2. 包含可能的函数名、变量名、类名等
        3. 考虑常见的编程命名约定
        4. 返回JSON格式的关键词列表

        示例：
        输入："帮我查找将数据插入到mysql的相关函数位置"
        输出：["insert", "mysql", "database", "db", "save", "add", "create", "data", "record", "table"]

        输入："查找用户登录验证的代码"
        输出：["login", "auth", "authenticate", "user", "verify", "check", "validate", "password", "credential"]"""
                
        prompt = f"""请从以下问题描述中提取相关的英文关键词：

        问题描述：{problem_description}

        请返回JSON格式的关键词列表："""
        
        response = self._call_llm(prompt, system_prompt)
        
        try:
            keywords = json.loads(response)
            if isinstance(keywords, list):
                return keywords
            else:
                return self._extract_keywords_fallback(problem_description)
        except json.JSONDecodeError:
            return self._extract_keywords_fallback(problem_description)
    
    def _extract_keywords_fallback(self, problem_description: str) -> List[str]:
        """临时提取方法"""
        keyword_mapping = {
            '插入': ['insert', 'add', 'create', 'save'],
            'mysql': ['mysql', 'database', 'db', 'sql'],
            '数据': ['data', 'record', 'info'],
            '登录': ['login', 'auth', 'authenticate', 'signin'],
            '用户': ['user', 'account', 'member'],
            '验证': ['verify', 'validate', 'check', 'auth'],
            '查询': ['query', 'search', 'find', 'get', 'select'],
            '更新': ['update', 'modify', 'edit', 'change'],
            '删除': ['delete', 'remove', 'del'],
            '计算': ['calculate', 'compute', 'calc'],
            '处理': ['process', 'handle', 'deal'],
            '发送': ['send', 'post', 'submit'],
            '接收': ['receive', 'get', 'fetch']
        }
        
        keywords = set()
        for chinese_word, english_words in keyword_mapping.items():
            if chinese_word in problem_description:
                keywords.update(english_words)
        
        result = list(keywords) if keywords else ['data', 'function', 'method']
        logger.info(f"{result}")
        return result
    
    def filter_relevant_functions(self, problem_description: str, all_functions: List[Dict], all_classes: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """方案3：使用LLM从所有函数中筛选相关的"""
        system_prompt = """你是一个专业的代码分析师。你的任务是从给定的所有函数和类中筛选出与用户需求最相关的部分。

请遵循以下规则：
1. 仔细分析用户需求
2. 从函数名、类名、代码内容等角度判断相关性
3. 按照相关性排序，最相关的排在前面
4. 返回筛选后的函数和类列表
5. 如果某个函数或类不相关，不要包含在结果中

输出格式：
{
  "relevant_functions": [...],
  "relevant_classes": [...]
}"""
        
        # 构建函数和类的信息摘要
        functions_summary = []
        for func in all_functions[:50]:  # 限制数量避免token过多
            summary = f"函数: {func['name']} (文件: {func.get('file_path', 'unknown')})"
            if func.get('parent_class'):
                summary += f" [类: {func['parent_class']}]"
            functions_summary.append(summary)
        
        classes_summary = []
        for cls in all_classes[:20]:  # 限制数量
            summary = f"类: {cls['name']} (文件: {cls.get('file_path', 'unknown')})"
            if cls.get('methods'):
                summary += f" [方法: {cls['methods'][:100]}...]"
            classes_summary.append(summary)
        
        prompt = f"""用户需求：{problem_description}

所有函数列表：
{chr(10).join(functions_summary)}

所有类列表：
{chr(10).join(classes_summary)}

请从上述函数和类中筛选出与用户需求最相关的部分，并返回JSON格式的结果："""
        
        response = self._call_llm(prompt, system_prompt)
        
        try:
            result = json.loads(response)
            relevant_functions = result.get("relevant_functions", [])
            relevant_classes = result.get("relevant_classes", [])
            
            # 根据LLM筛选结果，从原始列表中找到对应的完整信息
            filtered_functions = []
            filtered_classes = []
            
            # 筛选函数
            if relevant_functions:
                for rel_func in relevant_functions:
                    func_name = rel_func.get('name') if isinstance(rel_func, dict) else str(rel_func)
                    for func in all_functions:
                        if func['name'] == func_name:
                            filtered_functions.append(func)
                            break
            
            # 筛选类
            if relevant_classes:
                for rel_cls in relevant_classes:
                    cls_name = rel_cls.get('name') if isinstance(rel_cls, dict) else str(rel_cls)
                    for cls in all_classes:
                        if cls['name'] == cls_name:
                            filtered_classes.append(cls)
                            break
            
            logger.info(f"LLM筛选结果: {len(filtered_functions)} 个相关函数，{len(filtered_classes)} 个相关类")
            return filtered_functions, filtered_classes
            
        except json.JSONDecodeError:
            logger.warning("LLM 返回的不是有效JSON，返回前几个结果")
            # 备用方案：返回前几个结果
            return all_functions[:10], all_classes[:5]
    
    def _get_language_specific_system_prompt(self, language: str) -> str:
        """根据编程语言生成相应的系统提示词"""
        base_rules = """你是一个专业的测试工程师。基于用户需求和完整的函数代码，生成相应的单元测试代码。

        请遵循以下通用规则：    
        1. 仔细分析提供的完整函数代码
        2. 理解函数的参数、返回值和业务逻辑
        3. 为每个函数生成独立的测试类或测试方法
        4. 生成完整可执行的测试代码，包含必要的导入语句
        5. 测试正常情况、边界情况和异常情况
        6. 使用适当的测试框架和断言方法
        7. 为每个测试用例添加清晰的注释说明
        8. 确保测试代码结构清晰，易于理解和维护
        9. 如果有多个函数，为每个函数创建对应的测试方法
        10. 重要：只返回纯代码，不要包含任何markdown格式标记（如```python或```）"""
        
        if language == "javascript":
            return base_rules + """

        **JavaScript/TypeScript 特定要求：**
        - 使用 Jest 测试框架
        - 使用 ES6+ 语法和模块导入 (import/export)
        - 使用 describe() 和 it() 组织测试
        - 使用 expect() 断言
        - 对于异步函数使用 async/await 或 Promise
        - 谨慎使用mock，只在必要时使用
        - 测试文件应该以 .test.js 或 .test.ts 结尾
        - **重要**: 如果函数没有被导出，请创建一个简化的、独立的测试函数来测试相同的逻辑
        - **重要**: 避免复杂的依赖mock，优先测试纯函数或创建简化版本
        - **重要**: 确保所有导入的模块和函数都是实际存在且可访问的
        
        示例格式：
        ```javascript
        import { functionName } from './filename';
        
        describe('functionName', () => {
          it('should return expected result', () => {
            expect(functionName(input)).toBe(expected);
          });
          
          it('should handle edge cases', () => {
            expect(functionName(edgeCase)).toBe(expectedEdgeResult);
          });
        });
        ```
        
        对于复杂的函数（如包含大量依赖的函数），请创建简化的测试：
        ```javascript
        // 如果原函数太复杂，创建一个简化版本来测试核心逻辑
        function simplifiedFunction(input) {
          // 提取原函数的核心逻辑
          return processedInput;
        }
        
        describe('simplifiedFunction (testing core logic)', () => {
          it('should process input correctly', () => {
            expect(simplifiedFunction(testInput)).toBe(expectedOutput);
          });
        });
        ```"""
        
        elif language == "python":
            return base_rules + """

        **Python 特定要求：**
        - 使用 unittest 或 pytest 测试框架
        - 使用标准的 import 语句
        - 使用 TestCase 类或 pytest 函数
        - 使用 assertEqual, assertTrue 等断言方法
        - 对于异常测试使用 assertRaises 或 pytest.raises
        - 使用 unittest.mock 进行模拟
        
        示例格式：
        ```python
        import unittest
        from filename import function_name
        
        class TestFunctionName(unittest.TestCase):
            def test_normal_case(self):
                self.assertEqual(function_name(input), expected)
        ```"""
        
        else:
            return base_rules + f"""

        **{language.upper()} 特定要求：**
        - 使用该语言的主流测试框架
        - 遵循该语言的命名约定和代码风格
        - 使用适当的导入和模块系统
        - 使用该语言标准的断言方法"""
    
    def _get_language_specific_prompt_template(self, language: str, enhanced_analysis: List[FeatureAnalysis]) -> tuple[str, str]:
        """根据语言生成导入信息和用户提示词模板"""
        
        if language == "javascript":
            # 收集JavaScript/TypeScript文件
            file_names = set()
            for analysis in enhanced_analysis:
                for loc in analysis.implementation_location:
                    if loc.file:
                        # 移除扩展名，保持相对路径
                        module_name = loc.file.replace('.js', '').replace('.ts', '').replace('.jsx', '').replace('.tsx', '')
                        file_names.add(module_name)
            
            import_info = f"模块文件: {', '.join(file_names)}" if file_names else ""
            
            user_prompt_template = """用户需求: {problem_description}

详细代码分析结果 (共 {total_functions} 个函数):
{analysis_info}

{import_info}

请基于上述完整的JavaScript/TypeScript函数代码，生成对应的Jest单元测试代码。

**重要指导原则：**
1. **仔细检查函数是否被导出** - 如果函数没有export关键字，请不要尝试导入它
2. **对于未导出的复杂函数** - 创建一个简化的、独立的测试函数来测试相同的核心逻辑
3. **避免复杂的Mock** - 只有在必要且简单的情况下才使用mock
4. **确保可执行性** - 生成的测试代码必须能够实际运行，不要包含不存在的导入

测试代码要求：
1. 使用ES6 import语法，但只导入实际存在且被导出的函数
2. 使用describe()和it()组织测试结构
3. 为每个函数创建独立的测试套件
4. 覆盖主要的测试场景（正常情况、边界情况、异常情况）
5. 使用expect()断言验证结果
6. 包含必要的测试数据和预期结果
7. 测试代码结构清晰，便于执行和维护

**示例处理方式：**
- 如果函数被导出：直接导入并测试
- 如果函数未被导出：创建简化版本测试核心逻辑
- 如果函数太复杂：分解为更小的、可测试的部分

请生成完整的、可执行的Jest测试代码："""
        
        else:  # Python 和其他语言
            # 收集Python文件
            file_names = set()
            for analysis in enhanced_analysis:
                for loc in analysis.implementation_location:
                    if loc.file:
                        module_name = loc.file.replace('.py', '')
                        file_names.add(module_name)
            
            import_info = f"导入模块: {', '.join(file_names)}" if file_names else ""
            
            user_prompt_template = """用户需求: {problem_description}

详细代码分析结果 (共 {total_functions} 个函数):
{analysis_info}

{import_info}

请基于上述完整的函数代码，生成对应的单元测试代码。确保测试代码能够：
1. 正确导入被测试的函数 - 重要：从上面提到的模块中导入函数，例如：from test_sample import calculate_sum, Calculator
2. 为每个函数创建独立的测试方法
3. 覆盖主要的测试场景（正常情况、边界情况、异常情况）
4. 使用合适的断言验证结果
5. 包含必要的测试数据和预期结果
6. 测试代码结构清晰，便于执行和维护
7. 确保导入语句与实际的文件名匹配

请生成完整的测试代码："""
        
        return import_info, user_prompt_template