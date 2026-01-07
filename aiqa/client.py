# aiqa/client.py
import os
import logging
from functools import lru_cache
from typing import Optional, TYPE_CHECKING, Any, Dict
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import requests

from .constants import AIQA_TRACER_NAME, LOG_TAG

logger = logging.getLogger(LOG_TAG)

# Compatibility import for TraceIdRatioBased sampler
# In older OpenTelemetry versions it was TraceIdRatioBasedSampler
# In newer versions (>=1.24.0) it's TraceIdRatioBased
TraceIdRatioBased: Optional[Any] = None
try:
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
except ImportError:
    try:
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBasedSampler as TraceIdRatioBased
    except ImportError:
        logger.warning(
            f"Could not import TraceIdRatioBased or TraceIdRatioBasedSampler from "
            "opentelemetry.sdk.trace.sampling. AIQA tracing may not work correctly. "
            "Please ensure opentelemetry-sdk>=1.24.0 is installed. "
            "Try: pip install --upgrade opentelemetry-sdk"
        )
        # Set to None so we can check later
        TraceIdRatioBased = None

from .http_utils import get_server_url, get_api_key, build_headers, format_http_error

class AIQAClient:
    """
    Singleton client for AIQA tracing.
    
    This class manages the tracing provider, exporter, and enabled state.
    Access via get_aiqa_client() which returns the singleton instance.
    """
    _instance: Optional['AIQAClient'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._provider: Optional[TracerProvider] = None
            cls._instance._exporter = None # reduce circular import issues by not importing for typecheck here
            cls._instance._enabled: bool = True
            cls._instance._initialized: bool = False
        return cls._instance
    
    @property
    def provider(self) -> Optional[TracerProvider]:
        """Get the tracer provider."""
        return self._provider
    
    @provider.setter
    def provider(self, value: Optional[TracerProvider]) -> None:
        """Set the tracer provider."""
        self._provider = value
    
    @property
    def exporter(self) -> Optional[Any]:
        """Get the span exporter."""
        return self._exporter
    
    @exporter.setter
    def exporter(self, value: Optional[Any]) -> None:
        """Set the span exporter."""
        self._exporter = value
    
    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set the enabled state.
        
        When disabled:
        - Tracing does not create spans
        - Export does not send spans
        """
        logger.info(f"AIQA tracing {'enabled' if value else 'disabled'}")
        self._enabled = value
    
    def shutdown(self) -> None:
        """
        Shutdown the tracer provider and exporter.
        It is not necessary to call this function. 
        Use this to clean up resources at the end of all tracing.
        
        This will also set enabled=False to prevent further tracing attempts.
        """
        try:
            logger.info(f"AIQA tracing shutting down")
            # Disable tracing to prevent attempts to use shut-down system
            self.enabled = False
            if self._provider:
                self._provider.shutdown()
            if self._exporter:
                self._exporter.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down tracing: {e}")
            # Still disable even if shutdown had errors
            self.enabled = False



# Global singleton instance
client: AIQAClient = AIQAClient()

# Component tag to add to all spans (can be set via AIQA_COMPONENT_TAG env var or programmatically)
_component_tag: str = ""


def get_component_tag() -> str:
    """Get the current component tag."""
    return _component_tag


def set_component_tag(tag: Optional[str]) -> None:
    """Set the component tag programmatically (overrides environment variable)."""
    global _component_tag
    _component_tag = tag or ""


@lru_cache(maxsize=1)
def get_aiqa_client() -> AIQAClient:
    """
    Initialize and return the AIQA client singleton.
    
    This function is called automatically when WithTracing is first used, so you typically
    don't need to call it explicitly. However, you can call it manually if you want to:
    - Check if tracing is enabled (client.enabled)
    - Initialize before the first @WithTracing usage
    - Access the client object for advanced usage
    
    The function loads environment variables (AIQA_SERVER_URL, AIQA_API_KEY, AIQA_COMPONENT_TAG)
    and initializes the tracing system.
    
    The client object manages the tracing system state. Tracing is done by the WithTracing 
    decorator. Experiments are run by the ExperimentRunner class.
    
    The function is idempotent - calling it multiple times is safe and will only
    initialize once.
    
    Example:
        from aiqa import get_aiqa_client, WithTracing
        
        # Optional: Initialize explicitly (usually not needed)
        client = get_aiqa_client()
        if client.enabled:
            print(f"Tracing is enabled")
        
        @WithTracing
        def my_function():
            pass  # Initialization happens automatically here if not done above
    """
    global client
    try:
        _init_tracing()
    except Exception as e:
        logger.error(f"Failed to initialize AIQA tracing: {e}")
        logger.warning(f"AIQA tracing is disabled. Your application will continue to run without tracing.")
    return client

def _init_tracing() -> None:
    """Initialize tracing system and load configuration from environment variables."""
    global client
    if client._initialized:
        return
    
    try:
        server_url = get_server_url()
        api_key = get_api_key()
        
        if not api_key:
            client.enabled = False
            logger.warning(
                f"AIQA tracing is disabled: missing required environment variables: AIQA_API_KEY"
            )
            client._initialized = True
            return
        
        # Initialize component tag from environment variable
        set_component_tag(os.getenv("AIQA_COMPONENT_TAG", None))
        
        provider = trace.get_tracer_provider()

        # Get sampling rate from environment (default: 1.0 = sample all)
        sampling_rate = 1.0
        if env_rate := os.getenv("AIQA_SAMPLING_RATE"):
            try:
                rate = float(env_rate)
                sampling_rate = max(0.0, min(1.0, rate))  # Clamp to [0, 1]
            except ValueError:
                logger.warning(f"Invalid AIQA_SAMPLING_RATE value '{env_rate}', using default 1.0")

        # If it's still the default proxy, install a real SDK provider
        if not isinstance(provider, TracerProvider):
            if TraceIdRatioBased is None:
                raise ImportError(
                    "TraceIdRatioBased sampler is not available. "
                    "Please install opentelemetry-sdk>=1.24.0"
                )
            
            # Create sampler based on trace-id for deterministic sampling
            sampler = TraceIdRatioBased(sampling_rate)
            provider = TracerProvider(sampler=sampler)
            trace.set_tracer_provider(provider)

        # Idempotently add your processor
        _attach_aiqa_processor(provider)
        client.provider = provider
        
        # Log successful initialization
        logger.info(f"AIQA initialized and tracing (sampling rate: {sampling_rate:.2f}, server: {server_url})")
        client._initialized = True
        
    except Exception as e:
        logger.error(f"Error initializing AIQA tracing: {e}")
        client._initialized = True  # Mark as initialized even on error to prevent retry loops
        raise

def _attach_aiqa_processor(provider: TracerProvider) -> None:
    """Attach AIQA span processor to the provider. Idempotent - safe to call multiple times."""
    from .aiqa_exporter import AIQASpanExporter
    
    try:
        # Check if already attached
        for p in provider._active_span_processor._span_processors:
            if isinstance(getattr(p, "exporter", None), AIQASpanExporter):
                logger.debug(f"AIQA span processor already attached, skipping")
                return

        exporter = AIQASpanExporter(
            server_url=os.getenv("AIQA_SERVER_URL"),
            api_key=os.getenv("AIQA_API_KEY"),
            # max_buffer_spans will be read from AIQA_MAX_BUFFER_SPANS env var by the exporter
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        global client
        client.exporter = exporter
        logger.debug(f"AIQA span processor attached successfully")
    except Exception as e:
        logger.error(f"Error attaching AIQA span processor: {e}")
        # Re-raise to let _init_tracing handle it - it will log and continue
        raise



def get_aiqa_tracer() -> trace.Tracer:
    """
    Get the AIQA tracer with version from __init__.py __version__.
    This should be used instead of trace.get_tracer() so that the version is set.
    """
    try:
        # Import here to avoid circular import
        from . import VERSION
        # Compatibility: version parameter may not be supported in older OpenTelemetry versions
        # Try with version parameter (newer OpenTelemetry versions)
        return trace.get_tracer(AIQA_TRACER_NAME, version=VERSION)
    except Exception as e:
        # Log issue but still return a tracer
        logger.info(f"Issue getting AIQA tracer with version: {e}, using fallback")
        return trace.get_tracer(AIQA_TRACER_NAME)


def get_organisation(
    organisation_id: str,
    server_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get organisation information based on API key via an API call.
    
    Args:
        organisation_id: ID of the organisation to retrieve
        server_url: Optional server URL (defaults to AIQA_SERVER_URL env var)
        api_key: Optional API key (defaults to AIQA_API_KEY env var)
    
    Returns:
        Organisation object as a dictionary
    """
    url = get_server_url(server_url)
    key = get_api_key(api_key)
    headers = build_headers(key)
    
    response = requests.get(
        f"{url}/organisation/{organisation_id}",
        headers=headers,
    )
    
    if not response.ok:
        raise Exception(format_http_error(response, "get organisation"))
    
    return response.json()


def get_api_key_info(
    api_key_id: str,
    server_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get API key information via an API call.
    
    Args:
        api_key_id: ID of the API key to retrieve
        server_url: Optional server URL (defaults to AIQA_SERVER_URL env var)
        api_key: Optional API key (defaults to AIQA_API_KEY env var)
    
    Returns:
        ApiKey object as a dictionary
    """
    url = get_server_url(server_url)
    key = get_api_key(api_key)
    headers = build_headers(key)
    
    response = requests.get(
        f"{url}/api-key/{api_key_id}",
        headers=headers,
    )
    
    if not response.ok:
        raise Exception(format_http_error(response, "get api key info"))
    
    return response.json()