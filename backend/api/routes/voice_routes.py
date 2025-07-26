# api/routes/voice_routes.py - Voice processing endpoints
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import base64
import logging

# Import orchestrator
from orchestration.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/voice", tags=["voice"])

# Request/Response models
class VoiceProcessRequest(BaseModel):
    """Generic voice processing request"""
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    text: Optional[str] = Field("", description="Text input if no audio")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class VoiceProcessResponse(BaseModel):
    """Generic voice processing response"""
    response: str
    voice_data: Optional[Dict[str, Any]] = None
    insights: Optional[list] = None
    memory_stored: Optional[bool] = None
    memories_retrieved: Optional[list] = None
    error: Optional[str] = None

# Dependency to get orchestrator instance
def get_orchestrator() -> Orchestrator:
    """Get the orchestrator instance from app state"""
    from main import app
    if not hasattr(app.state, 'orchestrator'):
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return app.state.orchestrator

@router.post("/process", response_model=VoiceProcessResponse)
async def process_voice(
    request: VoiceProcessRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """
    Process voice input through the orchestrator
    
    This endpoint simply forwards the request to the orchestrator
    and returns its response.
    """
    try:
        # Build orchestrator input
        orchestrator_input = {
            "text": request.text or "",
            "audio_data": request.audio_data,
            "metadata": request.metadata
        }
        
        # Process through orchestrator
        result = await orchestrator.process(orchestrator_input)
        
        # Return orchestrator response directly
        return VoiceProcessResponse(
            response=result.get("response", ""),
            voice_data=result.get("voice_data"),
            insights=result.get("insights"),
            memory_stored=result.get("memory_stored"),
            memories_retrieved=result.get("memories_retrieved"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/upload")
# async def upload_voice_file(
#     file: UploadFile = File(...),
#     metadata: Optional[str] = None,
#     orchestrator: Orchestrator = Depends(get_orchestrator)
# ):
#     """
#     Upload a voice file for processing
    
#     Converts the file to base64 and forwards to orchestrator.
#     """
#     try:
#         # Read and encode file
#         audio_bytes = await file.read()
#         audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
#         # Parse metadata if provided
#         import json
#         parsed_metadata = {}
#         if metadata:
#             try:
#                 parsed_metadata = json.loads(metadata)
#             except:
#                 pass
        
#         # Add file info to metadata
#         parsed_metadata["filename"] = file.filename
#         parsed_metadata["content_type"] = file.content_type
        
#         # Build orchestrator input
#         orchestrator_input = {
#             "text": "",
#             "audio_data": audio_base64,
#             "metadata": parsed_metadata
#         }
        
#         # Process through orchestrator
#         result = await orchestrator.process(orchestrator_input)
        
#         # Return result
#         return {
#             "response": result.get("response", ""),
#             "voice_data": result.get("voice_data"),
#             "insights": result.get("insights"),
#             "memory_stored": result.get("memory_stored"),
#             "memories_retrieved": result.get("memories_retrieved"),
#             "error": result.get("error")
#         }
        
#     except Exception as e:
#         logger.error(f"File upload error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/health")
# async def voice_health_check(orchestrator: Orchestrator = Depends(get_orchestrator)):
#     """Check voice service health"""
#     try:
#         # Get agent pool stats
#         pool_stats = orchestrator.agent_pool.get_pool_stats()
        
#         return {
#             "status": "healthy",
#             "voice_agent_stats": pool_stats.get("voice", {}),
#         }
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e)
#         }