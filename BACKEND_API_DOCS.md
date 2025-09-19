# 纯后端Wiki生成接口文档

## 概述

这个纯后端接口提供了一个完整的Wiki生成解决方案，可以为任何GitHub、GitLab或Bitbucket仓库自动生成结构化的技术文档。

## API端点

### POST `/api/generate_wiki`

生成完整的仓库Wiki文档。

#### 请求参数

```json
{
  "repo_url": "https://github.com/user/repo",           // 必需：仓库URL
  "repo_type": "github",                                // 可选：仓库类型 (github|gitlab|bitbucket|local)
  "token": "your_access_token",                         // 可选：私有仓库访问令牌
  "local_path": "/path/to/local/repo",                  // 可选：本地仓库路径
  "provider": "google",                                 // 可选：AI模型提供商
  "model": "gemini-2.5-flash",                         // 可选：具体模型名称
  "custom_model": "custom-model-name",                  // 可选：自定义模型名称
  "language": "zh",                                     // 可选：生成语言 (en|zh|ja|es|kr|vi|pt-br|fr|ru)
  "is_comprehensive": true,                             // 可选：是否生成详细Wiki (true|false)
  "excluded_dirs": "node_modules,dist,build",          // 可选：排除的目录
  "excluded_files": "*.log,*.tmp",                     // 可选：排除的文件模式
  "included_dirs": "src,docs",                         // 可选：仅包含的目录
  "included_files": "*.py,*.js,*.md",                  // 可选：仅包含的文件模式
  "authorization_code": "your_auth_code"                // 可选：授权码
}
```

#### 响应格式

```json
{
  "success": true,                                      // 是否成功
  "message": "Wiki generated successfully with 8 pages", // 状态消息
  "wiki_structure": {                                   // Wiki结构
    "id": "wiki",
    "title": "项目Wiki标题",
    "description": "项目描述",
    "pages": [                                          // 页面列表
      {
        "id": "page-1",
        "title": "系统架构",
        "content": "",                                  // 页面内容(在generated_pages中)
        "filePaths": ["src/main.py", "docs/arch.md"],
        "importance": "high",
        "relatedPages": ["page-2", "page-3"]
      }
    ],
    "sections": [                                       // 章节结构（详细模式）
      {
        "id": "section-1", 
        "title": "核心功能",
        "pages": ["page-1", "page-2"]
      }
    ],
    "rootSections": ["section-1", "section-2"]
  },
  "generated_pages": {                                  // 生成的页面内容
    "page-1": {
      "id": "page-1",
      "title": "系统架构", 
      "content": "# 系统架构\n\n## 概述\n...",     // 实际的Markdown内容
      "filePaths": ["src/main.py"],
      "importance": "high",
      "relatedPages": ["page-2"]
    }
  },
  "cache_key": "/path/to/cache/file.json",             // 缓存文件路径
  "errors": ["可能的错误信息"]                            // 错误列表（如果有）
}
```

## 使用示例

### 1. 基本使用 - 生成GitHub仓库Wiki

```bash
curl -X POST "http://localhost:8001/api/generate_wiki" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/AsyncFuncAI/deepwiki-open",
    "provider": "google",
    "language": "zh"
  }'
```

### 2. 生成私有仓库Wiki（带访问令牌）

```bash
curl -X POST "http://localhost:8001/api/generate_wiki" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/private/repo",
    "token": "ghp_your_github_token",
    "provider": "openai",
    "model": "gpt-4",
    "language": "en"
  }'
```

### 3. 生成本地仓库Wiki

```bash
curl -X POST "http://localhost:8001/api/generate_wiki" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "file:///path/to/local/repo",
    "repo_type": "local",
    "local_path": "/path/to/local/repo",
    "provider": "ollama",
    "model": "qwen:7b",
    "language": "zh"
  }'
```

### 4. 使用文件过滤器

```bash
curl -X POST "http://localhost:8001/api/generate_wiki" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/user/repo",
    "included_dirs": "src,docs,api",
    "excluded_files": "*.test.js,*.spec.py",
    "language": "zh",
    "is_comprehensive": false
  }'
```

### 5. Python脚本调用

```python
import requests

def generate_wiki(repo_url, language="zh", provider="google"):
    response = requests.post(
        "http://localhost:8001/api/generate_wiki",
        json={
            "repo_url": repo_url,
            "language": language,
            "provider": provider,
            "is_comprehensive": True
        },
        timeout=300
    )
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            print(f"✅ Wiki生成成功: {result['message']}")
            return result
        else:
            print(f"❌ 生成失败: {result['message']}")
    else:
        print(f"❌ API错误: {response.status_code}")
    
    return None

# 使用示例
wiki_data = generate_wiki("https://github.com/user/repo")
if wiki_data:
    pages = wiki_data["generated_pages"]
    for page_id, page in pages.items():
        print(f"页面: {page['title']}")
        print(f"内容长度: {len(page['content'])} 字符")
```

## 核心特性

### 1. 支持多种仓库类型
- **GitHub**: 公开和私有仓库
- **GitLab**: GitLab.com和自托管实例
- **Bitbucket**: Bitbucket.org仓库  
- **Local**: 本地文件系统仓库

### 2. 多AI模型支持
- **Google Gemini**: gemini-2.5-flash等
- **OpenAI**: gpt-4, gpt-3.5-turbo等
- **OpenRouter**: 多种第三方模型
- **Ollama**: 本地部署模型
- **Azure AI**: Azure OpenAI服务

### 3. 多语言生成
支持9种语言的文档生成：
- 中文 (zh)
- 英文 (en) 
- 日文 (ja)
- 西班牙文 (es)
- 韩文 (kr)
- 越南文 (vi)
- 葡萄牙文 (pt-br)
- 法文 (fr)
- 俄文 (ru)

### 4. 智能内容生成
- **结构化分析**: 自动分析仓库结构并确定Wiki框架
- **相关文件识别**: 为每个页面自动识别相关源文件
- **Mermaid图表**: 自动生成架构图、流程图等可视化内容
- **代码引用**: 精确引用源文件和行号
- **交叉引用**: 页面间的智能关联

### 5. 文件过滤系统
- **排除模式**: 排除特定目录和文件模式
- **包含模式**: 仅处理指定目录和文件
- **预设过滤**: 自动排除常见的构建产物和依赖目录

### 6. 缓存机制
- **自动缓存**: 生成的Wiki自动缓存到本地
- **增量更新**: 支持基于缓存的增量更新
- **缓存管理**: 提供缓存查询和清理API

## 高级配置

### 环境变量

```bash
# AI模型API密钥
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key  
OPENROUTER_API_KEY=your_openrouter_api_key

# Ollama配置
OLLAMA_HOST=http://localhost:11434

# 认证配置
DEEPWIKI_AUTH_MODE=true
DEEPWIKI_AUTH_CODE=your_secret_code

# 服务配置
PORT=8001
SERVER_BASE_URL=http://localhost:8001
```

### 文件过滤配置

在 `api/config/repo.json` 中配置默认过滤规则：

```json
{
  "file_filters": {
    "excluded_dirs": [
      "node_modules", "dist", "build", ".git", 
      "__pycache__", ".venv", "venv"
    ],
    "excluded_files": [
      "*.log", "*.tmp", "*.cache", "package-lock.json",
      "yarn.lock", "*.pyc", "*.pyo"
    ]
  }
}
```

## 错误处理

### 常见错误及解决方案

1. **401 Unauthorized**: 检查访问令牌是否正确
2. **400 Bad Request**: 检查仓库URL格式
3. **500 Internal Server Error**: 检查AI模型API密钥配置
4. **Timeout**: 大型仓库可能需要更长时间，可以增加超时设置

### 错误响应格式

```json
{
  "success": false,
  "message": "错误描述",
  "errors": ["详细错误信息列表"]
}
```

## 性能优化

### 1. 仓库大小限制
- 建议仓库文件数量 < 1000个
- 单个文件大小 < 1MB
- 总仓库大小 < 100MB

### 2. 并发控制
- 同时生成请求数量限制
- 页面生成队列管理
- 资源使用监控

### 3. 缓存策略
- 基于仓库URL+语言+模型的缓存键
- 自动过期机制
- 磁盘空间管理

## 扩展开发

### 添加新的仓库类型

1. 在 `data_pipeline.py` 中添加新的获取函数
2. 更新 `get_repository_structure` 函数
3. 添加相应的认证逻辑

### 添加新的AI模型

1. 在 `config/generator.json` 中添加模型配置
2. 实现对应的客户端类
3. 更新模型选择逻辑

### 自定义内容模板

可以通过修改 `generate_page_content_internal` 函数中的提示词模板来自定义生成内容的格式和风格。

## 测试工具

使用提供的测试脚本：

```bash
python test_wiki_generation_api.py \
  --repo-url https://github.com/user/repo \
  --provider google \
  --language zh \
  --save-result
```

## 部署建议

### Docker部署

```dockerfile
# 使用现有的Dockerfile
docker build -t deepwiki-open .
docker run -p 8001:8001 \
  -e GOOGLE_API_KEY=your_key \
  -v ~/.adalflow:/root/.adalflow \
  deepwiki-open
```

### 生产环境配置

1. **负载均衡**: 使用Nginx或类似工具进行负载均衡
2. **数据库**: 考虑使用Redis等缓存数据库
3. **监控**: 添加API监控和日志记录
4. **安全**: 启用HTTPS和访问控制

## 许可证

MIT License - 详见项目根目录的LICENSE文件。