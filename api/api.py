import os
import logging
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
import google.generativeai as genai
import asyncio
import re

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Streaming API",
    description="API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Helper function to get adalflow root path
def get_adalflow_default_root_path():
    return os.path.expanduser(os.path.join("~", ".adalflow"))

# --- Pydantic Models ---
class WikiPage(BaseModel):
    """
    Model for a wiki page.
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str # Should ideally be Literal['high', 'medium', 'low']
    relatedPages: List[str]

class ProcessedProjectEntry(BaseModel):
    id: str  # Filename
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str # Renamed from type to repo_type for clarity with existing models
    submittedAt: int # Timestamp
    language: str # Extracted from filename

class RepoInfo(BaseModel):
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None


class WikiSection(BaseModel):
    """
    Model for the wiki sections.
    """
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """
    Model for the overall wiki structure.
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None

class WikiCacheData(BaseModel):
    """
    Model for the data to be stored in the wiki cache.
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  #compatible for old cache
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class WikiCacheRequest(BaseModel):
    """
    Model for the request body when saving wiki cache.
    """
    repo: RepoInfo
    language: str
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str

class WikiExportRequest(BaseModel):
    """
    Model for requesting a wiki export.
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format (markdown or json)")

class WikiGenerationRequest(BaseModel):
    """
    Model for requesting wiki generation.
    """
    repo_url: str = Field(..., description="URL of the repository")
    repo_type: str = Field("github", description="Type of repository (github, gitlab, bitbucket, local)")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    local_path: Optional[str] = Field(None, description="Local path for local repositories")
    provider: str = Field("google", description="Model provider (google, openai, openrouter, ollama, azure)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")
    custom_model: Optional[str] = Field(None, description="Custom model name if using custom model")
    language: str = Field("en", description="Language for content generation")
    is_comprehensive: bool = Field(False, description="Whether to generate comprehensive or concise wiki")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")
    authorization_code: Optional[str] = Field(None, description="Authorization code for protected operations")
    is_refresh: bool = Field(False, description="Whether to refresh the cache")

class WikiGenerationResponse(BaseModel):
    """
    Model for wiki generation response.
    """
    success: bool = Field(..., description="Whether the generation was successful")
    message: str = Field(..., description="Status message")
    # wiki_structure: Optional[WikiStructureModel] = Field(None, description="Generated wiki structure")
    # generated_pages: Optional[Dict[str, WikiPage]] = Field(None, description="Generated wiki pages")
    cache_key: Optional[str] = Field(None, description="Cache key for the generated wiki")
    errors: Optional[List[str]] = Field(None, description="List of errors if any occurred")

# --- Model Configuration Models ---
class Model(BaseModel):
    """
    Model for LLM model configuration
    """
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")

class Provider(BaseModel):
    """
    Model for LLM provider configuration
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(..., description="List of available models for this provider")
    supportsCustomModel: Optional[bool] = Field(False, description="Whether this provider supports custom models")

class ModelConfig(BaseModel):
    """
    Model for the entire model configuration
    """
    providers: List[Provider] = Field(..., description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")

class AuthorizationConfig(BaseModel):
    code: str = Field(..., description="Authorization code")

from api.config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE
from api.data_pipeline import get_repository_structure
from api.rag import RAG

@app.get("/lang/config")
async def get_lang_config():
    return configs["lang_config"]

@app.get("/auth/status")
async def get_auth_status():
    """
    Check if authentication is required for the wiki.
    """
    return {"auth_required": WIKI_AUTH_MODE}

@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """
    Check authorization code.
    """
    return {"success": WIKI_AUTH_CODE == request.code}

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    Get available model providers and their models.

    This endpoint returns the configuration of available model providers and their
    respective models that can be used throughout the application.

    Returns:
        ModelConfig: A configuration object containing providers and their models
    """
    try:
        logger.info("Fetching model configurations")

        # Create providers from the config file
        providers = []
        default_provider = configs.get("default_provider", "google")

        # Add provider configuration based on config.py
        for provider_id, provider_config in configs["providers"].items():
            models = []
            # Add models from config
            for model_id in provider_config["models"].keys():
                # Get a more user-friendly display name if possible
                models.append(Model(id=model_id, name=model_id))

            # Add provider with its models
            providers.append(
                Provider(
                    id=provider_id,
                    name=f"{provider_id.capitalize()}",
                    supportsCustomModel=provider_config.get("supportsCustomModel", False),
                    models=models
                )
            )

        # Create and return the full configuration
        config = ModelConfig(
            providers=providers,
            defaultProvider=default_provider
        )
        return config

    except Exception as e:
        logger.error(f"Error creating model configuration: {str(e)}")
        # Return some default configuration in case of error
        return ModelConfig(
            providers=[
                Provider(
                    id="google",
                    name="Google",
                    supportsCustomModel=True,
                    models=[
                        Model(id="gemini-2.5-flash", name="Gemini 2.5 Flash")
                    ]
                )
            ],
            defaultProvider="google"
        )

@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    Export wiki content as Markdown or JSON.

    Args:
        request: The export request containing wiki pages and format

    Returns:
        A downloadable file in the requested format
    """
    try:
        logger.info(f"Exporting wiki for {request.repo_url} in {request.format} format")

        # Extract repository name from URL for the filename
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # Get current timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # Generate Markdown content
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON format
            # Generate JSON content
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # Create response with appropriate headers for file download
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"Error exporting wiki: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/local_repo/structure")
async def get_local_repo_structure(path: str = Query(None, description="Path to local repository")):
    """Return the file tree and README content for a local repository."""
    if not path:
        return JSONResponse(
            status_code=400,
            content={"error": "No path provided. Please provide a 'path' query parameter."}
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Directory not found: {path}"}
        )

    try:
        logger.info(f"Processing local repository at: {path}")
        file_tree_lines = []
        readme_content = ""

        for root, dirs, files in os.walk(path):
            # Exclude hidden dirs/files and virtual envs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'node_modules' and d != '.venv']
            for file in files:
                if file.startswith('.') or file == '__init__.py' or file == '.DS_Store':
                    continue
                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                file_tree_lines.append(rel_file)
                # Find README.md (case-insensitive)
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README.md: {str(e)}")
                        readme_content = ""

        file_tree_str = '\n'.join(sorted(file_tree_lines))
        return {"file_tree": file_tree_str, "readme": readme_content}
    except Exception as e:
        logger.error(f"Error processing local repository: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing local repository: {str(e)}"}
        )

def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate Markdown export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        Markdown content as string
    """
    # Start with metadata
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add table of contents
    markdown += "## Table of Contents\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"



        # Add related pages
        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                # Find the title of the related page
                related_page = next((p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(f"[{related_page.title}](#{related_id})")

            if related_titles:
                markdown += "Related topics: " + ", ".join(related_titles) + "\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown

def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate JSON export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        JSON content as string
    """
    # Create a dictionary with metadata and pages
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # Convert to JSON string with pretty formatting
    return json.dumps(export_data, indent=2)

# Import the simplified chat implementation
from api.simple_chat import chat_completions_stream
from api.websocket_wiki import handle_websocket_chat

# Add the chat_completions_stream endpoint to the main app
app.add_api_route("/chat/completions/stream", chat_completions_stream, methods=["POST"])

# Add the WebSocket endpoint
app.add_websocket_route("/ws/chat", handle_websocket_chat)

# --- Wiki Cache Helper Functions ---

WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)

def get_wiki_cache_path(owner: str, repo: str, repo_type: str, language: str) -> str:
    """Generates the file path for a given wiki cache."""
    filename = f"deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json"
    return os.path.join(WIKI_CACHE_DIR, filename)

async def read_wiki_cache(owner: str, repo: str, repo_type: str, language: str) -> Optional[WikiCacheData]:
    """Reads wiki cache data from the file system."""
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from {cache_path}: {e}")
            return None
    return None

async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """Saves wiki cache data to the file system."""
    cache_path = get_wiki_cache_path(data.repo.owner, data.repo.repo, data.repo.type, data.language)
    logger.info(f"Attempting to save wiki cache. Path: {cache_path}")
    try:
        payload = WikiCacheData(
            wiki_structure=data.wiki_structure,
            generated_pages=data.generated_pages,
            repo=data.repo,
            provider=data.provider,
            model=data.model
        )
        # Log size of data to be cached for debugging (avoid logging full content if large)
        try:
            payload_json = payload.model_dump_json()
            payload_size = len(payload_json.encode('utf-8'))
            logger.info(f"Payload prepared for caching. Size: {payload_size} bytes.")
        except Exception as ser_e:
            logger.warning(f"Could not serialize payload for size logging: {ser_e}")


        logger.info(f"Writing cache file to: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving wiki cache to {cache_path}: {e.strerror} (errno: {e.errno})", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False

# --- Wiki Cache API Endpoints ---

@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content")
):
    """
    Retrieves cached wiki data (structure and generated pages) for a repository.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        language = configs["lang_config"]["default"]

    logger.info(f"Attempting to retrieve wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language)
    if cached_data:
        return cached_data
    else:
        # Return 200 with null body if not found, as frontend expects this behavior
        # Or, raise HTTPException(status_code=404, detail="Wiki cache not found") if preferred
        logger.info(f"Wiki cache not found for {owner}/{repo} ({repo_type}), lang: {language}")
        return None

@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """
    Stores generated wiki data (structure and pages) to the server-side cache.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]

    if not supported_langs.__contains__(request_data.language):
        request_data.language = configs["lang_config"]["default"]

    logger.info(f"Attempting to save wiki cache for {request_data.repo.owner}/{request_data.repo.repo} ({request_data.repo.type}), lang: {request_data.language}")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki cache saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save wiki cache")

@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content"),
    authorization_code: Optional[str] = Query(None, description="Authorization code")
):
    """
    Deletes a specific wiki cache from the file system.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        raise HTTPException(status_code=400, detail="Language is not supported")

    if WIKI_AUTH_MODE:
        logger.info("check the authorization code")
        if WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(status_code=401, detail="Authorization code is invalid")

    logger.info(f"Attempting to delete wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"Successfully deleted wiki cache: {cache_path}")
            return {"message": f"Wiki cache for {owner}/{repo} ({language}) deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting wiki cache {cache_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete wiki cache: {str(e)}")
    else:
        logger.warning(f"Wiki cache not found, cannot delete: {cache_path}")
        raise HTTPException(status_code=404, detail="Wiki cache not found")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api"
    }

@app.post("/api/generate_wiki", response_model=WikiGenerationResponse)
async def generate_wiki(request: WikiGenerationRequest):
    """
    Generate a complete wiki for a repository.
    
    This endpoint provides a pure backend solution for generating wiki documentation
    for a given repository. It handles the entire process from repository analysis
    to content generation.
    
    Args:
        request: Wiki generation request parameters
        
    Returns:
        WikiGenerationResponse: Complete wiki structure and generated content
    """
    try:
        logger.info(f"Starting wiki generation for {request.repo_url}")
        
        # Check authorization if required
        if WIKI_AUTH_MODE:
            if not request.authorization_code or WIKI_AUTH_CODE != request.authorization_code:
                raise HTTPException(status_code=401, detail="Invalid authorization code")
        
        # Validate language
        supported_langs = configs["lang_config"]["supported_languages"]
        if request.language not in supported_langs:
            request.language = configs["lang_config"]["default"]
        
        language = "English"
        if request.language == "zh":
            language = "Mandarin Chinese (中文)"

        if request.is_refresh:
            await delete_wiki_cache(
                extract_owner_from_url(request.repo_url),
                extract_repo_from_url(request.repo_url),
                request.repo_type,
                request.language
            )
        else:
            # Check if wiki already exists in cache
            cached_data = await read_wiki_cache(
                extract_owner_from_url(request.repo_url),
                extract_repo_from_url(request.repo_url),
                request.repo_type,
                request.language
            )
            
            if cached_data:
                logger.info("Found existing wiki cache, returning cached data")
                return WikiGenerationResponse(
                    success=True,
                    message="Wiki retrieved from cache",
                    # wiki_structure=cached_data.wiki_structure,
                    # generated_pages=cached_data.generated_pages,
                    cache_key=get_wiki_cache_path(
                        extract_owner_from_url(request.repo_url),
                        extract_repo_from_url(request.repo_url),
                        request.repo_type,
                        request.language
                    )
                )

        # Step 1: Get repository structure
        logger.info("Fetching repository structure...")
        repo_structure = await get_repository_structure_internal(
            request.repo_url,
            request.repo_type,
            request.token,
            request.local_path
        )
        
        if not repo_structure:
            raise HTTPException(status_code=400, detail="Failed to fetch repository structure")
        
        file_tree, readme_content = repo_structure
        
        # Step 2: Determine wiki structure using RAG
        logger.info("Determining wiki structure...")
        wiki_structure = await determine_wiki_structure_internal(
            file_tree,
            readme_content,
            extract_owner_from_url(request.repo_url),
            extract_repo_from_url(request.repo_url),
            request
        )
        
        if not wiki_structure:
            raise HTTPException(status_code=500, detail="Failed to determine wiki structure")
        
        # Step 3: Generate content for all pages
        logger.info(f"Generating content for {len(wiki_structure.pages)} pages...")
        generated_pages = {}
        errors = []
        
        for page in wiki_structure.pages:
            try:
                content = await generate_page_content_internal(
                    page,
                    language_name=language,
                    request=request,
                )
                generated_pages[page.id] = WikiPage(
                    id=page.id,
                    title=page.title,
                    content=content,
                    filePaths=page.filePaths,
                    importance=page.importance,
                    relatedPages=page.relatedPages
                )
                logger.info(f"Generated content for page: {page.title}")
            except Exception as e:
                error_msg = f"Failed to generate content for page {page.title}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Add empty content page to maintain structure
                generated_pages[page.id] = WikiPage(
                    id=page.id,
                    title=page.title,
                    content=f"Error generating content: {str(e)}",
                    filePaths=page.filePaths,
                    importance=page.importance,
                    relatedPages=page.relatedPages
                )
        
        # Step 4: Save to cache
        try:
            cache_request = WikiCacheRequest(
                repo=RepoInfo(
                    owner=extract_owner_from_url(request.repo_url),
                    repo=extract_repo_from_url(request.repo_url),
                    type=request.repo_type,
                    token=request.token,
                    localPath=request.local_path,
                    repoUrl=request.repo_url
                ),
                language=request.language,
                wiki_structure=wiki_structure,
                generated_pages=generated_pages,
                provider=request.provider,
                model=request.model or "default"
            )
            await save_wiki_cache(cache_request)
            logger.info("Wiki cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save wiki cache: {str(e)}")
            errors.append(f"Failed to save cache: {str(e)}")
        
        cache_key = get_wiki_cache_path(
            extract_owner_from_url(request.repo_url),
            extract_repo_from_url(request.repo_url),
            request.repo_type,
            request.language
        )
        
        return WikiGenerationResponse(
            success=True,
            message=f"Wiki generated successfully with {len(generated_pages)} pages",
            # wiki_structure=wiki_structure,
            # generated_pages=generated_pages,
            cache_key=cache_key,
            errors=errors if errors else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating wiki: {str(e)}", exc_info=True)
        return WikiGenerationResponse(
            success=False,
            message=f"Failed to generate wiki: {str(e)}",
            errors=[str(e)]
        )

@app.get("/")
async def root():
    """Root endpoint to check if the API is running and list available endpoints dynamically."""
    # Collect routes dynamically from the FastAPI app
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            # Skip docs and static routes
            if route.path in ["/openapi.json", "/docs", "/redoc", "/favicon.ico"]:
                continue
            # Group endpoints by first path segment
            path_parts = route.path.strip("/").split("/")
            group = path_parts[0].capitalize() if path_parts[0] else "Root"
            method_list = list(route.methods - {"HEAD", "OPTIONS"})
            for method in method_list:
                endpoints.setdefault(group, []).append(f"{method} {route.path}")

    # Optionally, sort endpoints for readability
    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "Welcome to Streaming API",
        "version": "1.0.0",
        "endpoints": endpoints
    }

# --- Processed Projects Endpoint --- (New Endpoint)
@app.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """
    Lists all processed projects found in the wiki cache directory.
    Projects are identified by files named like: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json
    """
    project_entries: List[ProcessedProjectEntry] = []
    # WIKI_CACHE_DIR is already defined globally in the file

    try:
        if not os.path.exists(WIKI_CACHE_DIR):
            logger.info(f"Cache directory {WIKI_CACHE_DIR} not found. Returning empty list.")
            return []

        logger.info(f"Scanning for project cache files in: {WIKI_CACHE_DIR}")
        filenames = await asyncio.to_thread(os.listdir, WIKI_CACHE_DIR) # Use asyncio.to_thread for os.listdir

        for filename in filenames:
            if filename.startswith("deepwiki_cache_") and filename.endswith(".json"):
                file_path = os.path.join(WIKI_CACHE_DIR, filename)
                try:
                    stats = await asyncio.to_thread(os.stat, file_path) # Use asyncio.to_thread for os.stat
                    parts = filename.replace("deepwiki_cache_", "").replace(".json", "").split('_')

                    # Expecting repo_type_owner_repo_language
                    # Example: deepwiki_cache_github_AsyncFuncAI_deepwiki-open_en.json
                    # parts = [github, AsyncFuncAI, deepwiki-open, en]
                    if len(parts) >= 4:
                        repo_type = parts[0]
                        owner = parts[1]
                        language = parts[-1] # language is the last part
                        repo = "_".join(parts[2:-1]) # repo can contain underscores

                        project_entries.append(
                            ProcessedProjectEntry(
                                id=filename,
                                owner=owner,
                                repo=repo,
                                name=f"{owner}/{repo}",
                                repo_type=repo_type,
                                submittedAt=int(stats.st_mtime * 1000), # Convert to milliseconds
                                language=language
                            )
                        )
                    else:
                        logger.warning(f"Could not parse project details from filename: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue # Skip this file on error

        # Sort by most recent first
        project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
        logger.info(f"Found {len(project_entries)} processed project entries.")
        return project_entries

    except Exception as e:
        logger.error(f"Error listing processed projects from {WIKI_CACHE_DIR}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list processed projects from server cache.")

# Helper functions for wiki generation

def extract_owner_from_url(repo_url: str) -> str:
    """Extract owner/organization name from repository URL."""
    try:
        # Remove trailing slash and split by '/'
        parts = repo_url.rstrip('/').split('/')
        if len(parts) >= 2:
            return parts[-2]  # Second to last part is usually the owner
        return "unknown"
    except Exception:
        return "unknown"

def extract_repo_from_url(repo_url: str) -> str:
    """Extract repository name from repository URL."""
    try:
        # Remove trailing slash and split by '/'
        parts = repo_url.rstrip('/').split('/')
        if len(parts) >= 1:
            repo_name = parts[-1]  # Last part is usually the repo name
            # Remove .git extension if present
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            return repo_name
        return "unknown"
    except Exception:
        return "unknown"

async def get_repository_structure_internal(
    repo_url: str,
    repo_type: str,
    token: Optional[str] = None,
    local_path: Optional[str] = None
) -> Optional[tuple[str, str]]:
    """Internal function to get repository structure."""
    try:
        if repo_type == "local" and local_path:
            # Handle local repository
            import os
            if not os.path.isdir(local_path):
                raise ValueError(f"Directory not found: {local_path}")
            
            file_tree_lines = []
            readme_content = ""
            
            for root, dirs, files in os.walk(local_path):
                # Exclude hidden dirs/files and common build directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.venv', 'venv']]
                for file in files:
                    if file.startswith('.') or file in ['__init__.py', '.DS_Store']:
                        continue
                    rel_dir = os.path.relpath(root, local_path)
                    rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                    file_tree_lines.append(rel_file)
                    
                    # Find README file
                    if file.lower() == 'readme.md' and not readme_content:
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                readme_content = f.read()
                        except Exception as e:
                            logger.warning(f"Could not read README.md: {str(e)}")
            
            file_tree = '\n'.join(sorted(file_tree_lines))
            return file_tree, readme_content
        else:
            # Use existing data_pipeline function for remote repositories
            structure_data = get_repository_structure(repo_url, repo_type, token)
            if structure_data and len(structure_data) >= 2:
                return structure_data[0], structure_data[1]  # file_tree, readme
            return None
    except Exception as e:
        logger.error(f"Error getting repository structure: {str(e)}")
        return None

async def determine_wiki_structure_internal(
    file_tree: str,
    readme: str,
    owner: str,
    repo: str,
    request: WikiGenerationRequest
) -> Optional[WikiStructureModel]:
    """Internal function to determine wiki structure using RAG."""
    try:
        # Create RAG instance
        #rag = RAG(provider=request.provider, model=request.model)
        
        # Prepare the structure determination prompt
        language_name = "English"
        if request.language == "zh":
            language_name = "Mandarin Chinese (中文)"

        structure_prompt = f"""Analyze this repository {owner}/{repo} and create a wiki structure for it.

1. The complete file tree of the project:
<file_tree>
{file_tree}
</file_tree>

2. The README file of the project:
<readme>
{readme}
</readme>

I want to create a wiki for this repository. Determine the most logical structure for a wiki based on the repository's content.

IMPORTANT: The wiki content will be generated in {language_name} language.

When designing the wiki structure, include pages that would benefit from visual diagrams, such as:
- Architecture overviews
- Data flow descriptions
- Component relationships
- Process workflows
- State machines
- Class hierarchies

{f"""
Create a structured wiki with the following main sections:
- Overview (general information about the project)
- System Architecture (how the system is designed)
- Core Features (key functionality)
- Data Management/Flow: If applicable, how data is stored, processed, accessed, and managed (e.g., database schema, data pipelines, state management).
- Frontend Components (UI elements, if applicable.)
- Backend Systems (server-side components)
- Model Integration (AI model connections)
- Deployment/Infrastructure (how to deploy, what's the infrastructure like)
- Extensibility and Customization: If the project architecture supports it, explain how to extend or customize its functionality (e.g., plugins, theming, custom modules, hooks).

Each section should contain relevant pages. For example, the "Frontend Components" section might include pages for "Home Page", "Repository Wiki Page", "Ask Component", etc.

Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <sections>
    <section id="section-1">
      <title>[Section title]</title>
      <pages>
        <page_ref>page-1</page_ref>
        <page_ref>page-2</page_ref>
      </pages>
      <subsections>
        <section_ref>section-2</section_ref>
      </subsections>
    </section>
    <!-- More sections as needed -->
  </sections>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description of what this page will cover]</description>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to a relevant file]</file_path>
        <!-- More file paths as needed -->
      </relevant_files>
      <related_pages>
        <related>page-2</related>
        <!-- More related page IDs as needed -->
      </related_pages>
      <parent_section>section-1</parent_section>
    </page>
    <!-- More pages as needed -->
  </pages>
</wiki_structure>
""" if request.is_comprehensive else f"""
Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description of what this page will cover]</description>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to a relevant file]</file_path>
        <!-- More file paths as needed -->
      </relevant_files>
      <related_pages>
        <related>page-2</related>
        <!-- More related page IDs as needed -->
      </related_pages>
    </page>
    <!-- More pages as needed -->
  </pages>
</wiki_structure>
"""}

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid XML structure specified above
- DO NOT wrap the XML in code blocks (no ``` or ```xml)
- DO NOT include any explanation text before or after the XML
- Ensure the XML is properly formatted and valid
- Start directly with <wiki_structure> and end with </wiki_structure>

IMPORTANT:
1. Create { "8-12" if request.is_comprehensive else "4-6" } pages that would make a { "comprehensive" if request.is_comprehensive else "concise" } wiki for this repository
2. Each page should focus on a specific aspect of the codebase (e.g., architecture, key features, setup)
3. The relevant_files should be actual files from the repository that would be used to generate that page
4. Return ONLY valid XML with the structure specified above, with no markdown code block delimiters """
        
        # Use simple chat to get structure
        from api.simple_chat import ChatCompletionRequest as SimpleChatRequest
        from api.simple_chat import ChatMessage as SimpleChatMessage
        from api.websocket_wiki import handle_websocket_chat
        import json
        
        chat_request = SimpleChatRequest(
            repo_url=request.repo_url,
            messages=[SimpleChatMessage(role="user", content=structure_prompt)],
            type=request.repo_type,
            token=request.token,
            provider=request.provider,
            model=request.model,
            language=request.language,
            excluded_dirs=request.excluded_dirs,
            excluded_files=request.excluded_files,
            included_dirs=request.included_dirs,
            included_files=request.included_files
        )
        
        # Create a mock WebSocket for internal use
        class MockWebSocket:
            def __init__(self):
                self.messages = []
                
            async def accept(self):
                pass
                
            async def receive_json(self):
                return chat_request.model_dump()
                
            async def send_text(self, text):
                self.messages.append(text)
                
            async def close(self):
                pass
        
        mock_ws = MockWebSocket()
        
        # Use the websocket handler to get response
        await handle_websocket_chat(mock_ws)
        
        # Combine all messages to get the response
        response_text = ''.join(mock_ws.messages)
        
        # Parse XML response
        response_text = response_text.replace('```xml', '').replace('```', '').strip()
        
        # Extract wiki structure from XML
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(response_text)
            
            # Extract basic info
            title = root.find('title').text if root.find('title') is not None else f"{owner}/{repo} Wiki"
            description = root.find('description').text if root.find('description') is not None else "Auto-generated wiki documentation"
            
            # Extract pages
            pages = []
            pages_element = root.find('pages')
            if pages_element is not None:
                for page_elem in pages_element.findall('page'):
                    page_id = page_elem.get('id', f'page-{len(pages) + 1}')
                    page_title = page_elem.find('title').text if page_elem.find('title') is not None else 'Untitled'
                    importance = page_elem.find('importance').text if page_elem.find('importance') is not None else 'medium'
                    
                    # Extract file paths
                    file_paths = []
                    relevant_files = page_elem.find('relevant_files')
                    if relevant_files is not None:
                        for file_path_elem in relevant_files.findall('file_path'):
                            if file_path_elem.text:
                                file_paths.append(file_path_elem.text)
                    
                    # Extract related pages
                    related_pages = []
                    related_pages_elem = page_elem.find('related_pages')
                    if related_pages_elem is not None:
                        for related_elem in related_pages_elem.findall('related'):
                            if related_elem.text:
                                related_pages.append(related_elem.text)
                    
                    pages.append(WikiPage(
                        id=page_id,
                        title=page_title,
                        content="",  # Will be generated later
                        filePaths=file_paths,
                        importance=importance,
                        relatedPages=related_pages
                    ))
            
            # Extract sections if comprehensive mode
            sections = []
            root_sections = []
            if request.is_comprehensive:
                sections_element = root.find('sections')
                if sections_element is not None:
                    for section_elem in sections_element.findall('section'):
                        section_id = section_elem.get('id', f'section-{len(sections) + 1}')
                        section_title = section_elem.find('title').text if section_elem.find('title') is not None else 'Untitled Section'
                        
                        # Extract page references
                        section_pages = []
                        pages_elem = section_elem.find('pages')
                        if pages_elem is not None:
                            for page_ref in pages_elem.findall('page_ref'):
                                if page_ref.text:
                                    section_pages.append(page_ref.text)
                        
                        sections.append(WikiSection(
                            id=section_id,
                            title=section_title,
                            pages=section_pages
                        ))
                        
                        # Assume all sections are root sections for simplicity
                        root_sections.append(section_id)
            
            return WikiStructureModel(
                id="wiki",
                title=title,
                description=description,
                pages=pages,
                sections=sections if sections else None,
                rootSections=root_sections if root_sections else None
            )
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            logger.error(f"Response text: {response_text[:500]}...")
            return None
            
    except Exception as e:
        logger.error(f"Error determining wiki structure: {str(e)}")
        return None

async def generate_page_content_internal(
    page: WikiPage,
    language_name,
    request: WikiGenerationRequest,
) -> str:
    """Internal function to generate content for a wiki page."""
    try:        
        # Create content generation prompt
        content_prompt = f"""You are an expert technical writer and software architect.
Your task is to generate a comprehensive and accurate technical wiki page in Markdown format about a specific feature, system, or module within a given software project.

You will be given:
1. The "[WIKI_PAGE_TOPIC]" for the page you need to create.
2. A list of "[RELEVANT_SOURCE_FILES]" from the project that you MUST use as the sole basis for the content. You have access to the full content of these files. You MUST use AT LEAST 5 relevant source files for comprehensive coverage - if fewer are provided, search for additional related files in the codebase.

CRITICAL STARTING INSTRUCTION:
The very first thing on the page MUST be a `<details>` block listing ALL the `[RELEVANT_SOURCE_FILES]` you used to generate the content. There MUST be AT LEAST 5 source files listed - if fewer were provided, you MUST find additional related files to include.
Format it exactly like this:
<details>
<summary>Relevant source files</summary>

Remember, do not provide any acknowledgements, disclaimers, apologies, or any other preface before the `<details>` block. JUST START with the `<details>` block.
The following files were used as context for generating this wiki page:

{chr(10).join(['- [' + path + '](' + path + ')' for path in page.filePaths])}
<!-- Add additional relevant files if fewer than 5 were provided -->
</details>

Immediately after the `<details>` block, the main title of the page should be a H1 Markdown heading: `# {page.title}`.

Based ONLY on the content of the `[RELEVANT_SOURCE_FILES]`:

1.  **Introduction:** Start with a concise introduction (1-2 paragraphs) explaining the purpose, scope, and high-level overview of "${page.title}" within the context of the overall project. If relevant, and if information is available in the provided files, link to other potential wiki pages using the format `[Link Text](#page-anchor-or-id)`.

2.  **Detailed Sections:** Break down "{page.title}" into logical sections using H2 (`##`) and H3 (`###`) Markdown headings. For each section:
    *   Explain the architecture, components, data flow, or logic relevant to the section's focus, as evidenced in the source files.
    *   Identify key functions, classes, data structures, API endpoints, or configuration elements pertinent to that section.

3.  **Mermaid Diagrams:**
    *   EXTENSIVELY use Mermaid diagrams (e.g., `flowchart TD`, `sequenceDiagram`, `classDiagram`, `erDiagram`, `graph TD`) to visually represent architectures, flows, relationships, and schemas found in the source files.
    *   Ensure diagrams are accurate and directly derived from information in the `[RELEVANT_SOURCE_FILES]`.
    *   Provide a brief explanation before or after each diagram to give context.
    *   CRITICAL: All diagrams MUST follow strict vertical orientation:
       - Use "graph TD" (top-down) directive for flow diagrams
       - NEVER use "graph LR" (left-right)
       - Maximum node width should be 3-4 words
       - For sequence diagrams:
         - Start with "sequenceDiagram" directive on its own line
         - Define ALL participants at the beginning
         - Use descriptive but concise participant names
         - Use the correct arrow types:
           - ->> for request/asynchronous messages
           - -->> for response messages
           - -x for failed messages
         - Include activation boxes using +/- notation
         - Add notes for clarification using "Note over" or "Note right of"

4.  **Tables:**
    *   Use Markdown tables to summarize information such as:
        *   Key features or components and their descriptions.
        *   API endpoint parameters, types, and descriptions.
        *   Configuration options, their types, and default values.
        *   Data model fields, types, constraints, and descriptions.

5.  **Code Snippets (ENTIRELY OPTIONAL):**
    *   Include short, relevant code snippets (e.g., Python, Java, JavaScript, SQL, JSON, YAML) directly from the `[RELEVANT_SOURCE_FILES]` to illustrate key implementation details, data structures, or configurations.
    *   Ensure snippets are well-formatted within Markdown code blocks with appropriate language identifiers.

6.  **Source Citations (EXTREMELY IMPORTANT):**
    *   For EVERY piece of significant information, explanation, diagram, table entry, or code snippet, you MUST cite the specific source file(s) and relevant line numbers from which the information was derived.
    *   Place citations at the end of the paragraph, under the diagram/table, or after the code snippet.
    *   Use the exact format: `Sources: [filename.ext:start_line-end_line]()` for a range, or `Sources: [filename.ext:line_number]()` for a single line. Multiple files can be cited: `Sources: [file1.ext:1-10](), [file2.ext:5](), [dir/file3.ext]()` (if the whole file is relevant and line numbers are not applicable or too broad).
    *   If an entire section is overwhelmingly based on one or two files, you can cite them under the section heading in addition to more specific citations within the section.
    *   IMPORTANT: You MUST cite AT LEAST 5 different source files throughout the wiki page to ensure comprehensive coverage.

7.  **Technical Accuracy:** All information must be derived SOLELY from the `[RELEVANT_SOURCE_FILES]`. Do not infer, invent, or use external knowledge about similar systems or common practices unless it's directly supported by the provided code. If information is not present in the provided files, do not include it or explicitly state its absence if crucial to the topic.

8.  **Clarity and Conciseness:** Use clear, professional, and concise technical language suitable for other developers working on or learning about the project. Avoid unnecessary jargon, but use correct technical terms where appropriate.

9.  **Conclusion/Summary:** End with a brief summary paragraph if appropriate for "{page.title}", reiterating the key aspects covered and their significance within the project.

IMPORTANT: Generate the content in {language_name} language.

Remember:
- Ground every claim in the provided source files.
- Prioritize accuracy and direct representation of the code's functionality and structure.
- Structure the document logically for easy understanding by other developers"""
        
        # Use simple chat to generate content
        from api.simple_chat import ChatCompletionRequest as SimpleChatRequest
        from api.simple_chat import ChatMessage as SimpleChatMessage
        from api.websocket_wiki import handle_websocket_chat
        
        chat_request = SimpleChatRequest(
            repo_url=request.repo_url,
            messages=[SimpleChatMessage(role="user", content=content_prompt)],
            type=request.repo_type,
            token=request.token,
            provider=request.provider,
            model=request.model,
            language=request.language,
            excluded_dirs=request.excluded_dirs,
            excluded_files=request.excluded_files,
            included_dirs=request.included_dirs,
            included_files=request.included_files
        )
        
        # Create a mock WebSocket for internal use
        class MockWebSocket:
            def __init__(self):
                self.messages = []
                
            async def accept(self):
                pass
                
            async def receive_json(self):
                return chat_request.model_dump()
                
            async def send_text(self, text):
                self.messages.append(text)
                
            async def close(self):
                pass
        
        mock_ws = MockWebSocket()
        
        # Use the websocket handler to get response
        await handle_websocket_chat(mock_ws)
        
        # Combine all messages to get the content
        content = ''.join(mock_ws.messages)
        
        # Clean up markdown delimiters
        content = re.sub(r'^```markdown\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```\s*$', '', content, flags=re.IGNORECASE)

        return content
        
    except Exception as e:
        logger.error(f"Error generating page content for {page.title}: {str(e)}")
        return f"Error generating content: {str(e)}"
