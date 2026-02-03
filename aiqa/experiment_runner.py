"""
ExperimentRunner - runs experiments on datasets and scores results
"""

import os
import time
import asyncio
from opentelemetry import context as otel_context
from opentelemetry.trace import Status, StatusCode, set_span_in_context
from .constants import LOG_TAG
from .http_utils import build_headers, get_server_url, get_api_key, format_http_error
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from .tracing import WithTracing
from .span_helpers import set_span_attribute, flush_tracing, get_active_trace_id
from .client import get_aiqa_client, get_aiqa_tracer, get_component_tag
from .object_serialiser import serialize_for_span
from .tracing_llm_utils import _extract_and_set_token_usage, _extract_and_set_provider_and_model
from .llm_as_judge import score_llm_metric_local, get_model_from_server, call_llm_fallback
import requests
from .types import MetricResult, ScoreThisInputOutputMetricType, Example, Result, Metric, CallLLMType

# Type aliases for engine/scoring functions to improve code completion and clarity
from typing import TypedDict

# Function that processes input and parameters to produce an output (sync or async)
CallMyCodeType = Callable[[Any, Dict[str, Any]], Union[Any, Awaitable[Any]]]

# Function that scores a given output, using input, example, and parameters (usually async)
# Returns a dictionary with score/message/etc.
ScoreThisOutputType = Callable[[Any, Any, Dict[str, Any], Dict[str, Any]], Awaitable[Dict[str, Any]]]


def _metric_score_key(metric: Dict[str, Any]) -> str:
    """Key for scores in API: server expects metric name (fallback to id)."""
    return (metric.get("name") or metric.get("id")) or ""


class ExperimentRunner:
    """
    The ExperimentRunner is the main class for running experiments on datasets.
    It can create an experiment, run it, and score the results.
    Handles setting up environment variables and passing parameters to the engine function.
    """

    def __init__(
        self,
        dataset_id: str,
        experiment_id: Optional[str] = None,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organisation_id: Optional[str] = None,
        llm_call_fn: Optional[CallLLMType] = None,
    ):
        """
        Initialize the ExperimentRunner.

        Args:
            dataset_id: ID of the dataset to run experiments on
            experiment_id: Usually unset, and a fresh experiment is created with a random ID
            server_url: URL of the AIQA server (defaults to AIQA_SERVER_URL env var)
            api_key: API key for authentication (defaults to AIQA_API_KEY env var)
            organisation_id: Optional organisation ID for the experiment. If not provided, will be
                            derived from the dataset when needed.
            llm_call_fn: Optional async function that takes (system_prompt, user_message) and returns
                        raw content string (typically JSON). If not provided, will check for OPENAI_API_KEY
                        or ANTHROPIC_API_KEY environment variables.
        """
        self.dataset_id = dataset_id
        self.experiment_id = experiment_id
        self.server_url = get_server_url(server_url)
        self.api_key = get_api_key(api_key)
        self.organisation = organisation_id
        self.experiment: Optional[Dict[str, Any]] = None
        self.scores: List[Dict[str, Any]] = []
        self.llm_call_fn = llm_call_fn
        self._dataset_cache: Optional[Dict[str, Any]] = None

    def _get_headers(self) -> Dict[str, str]:
        """Build HTTP headers for API requests."""
        return build_headers(self.api_key)

    def get_dataset(self) -> Dict[str, Any]:
        """
        Fetch the dataset to get its metrics.

        Returns:
            The dataset object with metrics and other information
        """
        if self._dataset_cache is not None:
            return self._dataset_cache

        response = requests.get(
            f"{self.server_url}/dataset/{self.dataset_id}",
            headers=self._get_headers(),
        )

        if not response.ok:
            raise Exception(format_http_error(response, "fetch dataset"))

        dataset = response.json()
        self._dataset_cache = dataset
        
        # If organisation_id wasn't set, derive it from the dataset
        if not self.organisation and dataset.get("organisation"):
            self.organisation = dataset.get("organisation")
        
        return dataset

    def get_example(self, example_id: str) -> Dict[str, Any]:
        """
        Fetch an example by ID.
        """
        response = requests.get(
            f"{self.server_url}/example/{example_id}",
            headers=self._get_headers(),
        )
        return response.json()

    def get_examples_for_dataset(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Fetch example inputs from the dataset.

        Args:
            limit: Maximum number of examples to fetch (default: 10000)

        Returns:
            List of example objects
        """
        params = {
            "dataset": self.dataset_id,
            "limit": str(limit),
        }
        if self.organisation:
            params["organisation"] = self.organisation

        response = requests.get(
            f"{self.server_url}/example",
            params=params,
            headers=self._get_headers(),
        )

        if not response.ok:
            raise Exception(format_http_error(response, "fetch example inputs"))

        data = response.json()
        return data.get("hits", [])

    def create_experiment(
        self, experiment_setup: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an experiment if one does not exist.

        Args:
            experiment_setup: Optional setup for the experiment object. You may wish to set:
                - name (recommended for labelling the experiment)
                - parameters

        Returns:
            The created experiment object
        """
        # Ensure we have the organisation ID - try to get it from the dataset if not set
        if not self.organisation:
            dataset = self.get_dataset()
            self.organisation = dataset.get("organisation")
        
        if not self.organisation or not self.dataset_id:
            raise Exception("Organisation and dataset ID are required to create an experiment. Organisation can be derived from the dataset or set via organisation_id parameter.")

        if not experiment_setup:
            experiment_setup = {}

        # Fill in if not set
        experiment_setup = {
            **experiment_setup,
            "organisation": self.organisation,
            "dataset": self.dataset_id,
            "results": [],
            "summaries": {},
        }

        print(f"Creating experiment")
        response = requests.post(
            f"{self.server_url}/experiment",
            json=experiment_setup,
            headers=self._get_headers(),
        )

        if not response.ok:
            raise Exception(format_http_error(response, "create experiment"))

        experiment = response.json()
        self.experiment_id = experiment["id"]
        self.experiment = experiment
        return experiment

    async def score_and_store(
        self,
        example: Example,
        output: Any,
        result: Result,
        trace_id: Optional[str] = None,
    ) -> Result:
        """
        Ask the server to score an example result. Stores the score for later summary calculation.

        Args:
            example: The example object
            output: The output from running the engine on the example
            result: The result object for locally calculated scores

        Returns:
            The score result from the server
        """
        # Do we have an experiment ID? If not, we need to create the experiment first
        if not self.experiment_id:
            self.create_experiment()
        example_id = example.get("id")
        if not example_id:
            raise ValueError("Example must have an 'id' field")
        if result is None:
            result = {"example": example_id, "scores": {}, "messages": {}, "errors": {}}
        scores = result.get("scores") or {}
        
        print(f"Scoring and storing example: {example_id}")
        print(f"Scores: {scores}")

        # Run synchronous requests.post in a thread pool to avoid blocking
        # Server expects output = raw output to score, not the result dict; scores keyed by metric name
        def _do_request():
            return requests.post(
                f"{self.server_url}/experiment/{self.experiment_id}/example/{example_id}/scoreAndStore",
                json={
                    "output": output,
                    "trace": trace_id,
                    "scores": scores,
                },
                headers=self._get_headers(),
            )

        response = await asyncio.to_thread(_do_request)

        if not response.ok:
            raise Exception(format_http_error(response, "score and store"))

        json_result = response.json()
        print(f"scoreAndStore response: {json_result}")
        return json_result

    async def run(
        self,
        call_my_code: CallMyCodeType,
        scorer_for_metric_id: Optional[Dict[str, ScoreThisInputOutputMetricType]] = None,
    ) -> None:
        """
        Run an engine function on all examples and score the results.

        Args:
            engine: Function that takes input, returns output (can be async)
            scorer: Optional function that scores the output given the example
        """
        examples = self.get_examples_for_dataset()

        for example in examples:
            try:
                scores = await self.run_example(example, call_my_code, scorer_for_metric_id)
                if scores:
                    self.scores.append(
                        {
                            "example": example,
                            "result": scores,
                            "scores": scores,
                        }
                    )
            except Exception as e:
                print(f"Error processing example {example.get('id', 'unknown')}: {e}")
                # Continue with next example instead of failing entire run

    async def run_example(
        self,
        example: Example,
        call_my_code: CallMyCodeType,
        scorer_for_metric_id: Optional[Dict[str, ScoreThisInputOutputMetricType]] = None,
    ) -> List[Result]:
        """
        Run the engine on an example with the experiment's parameters, score the result, and store it.

        Spans: one root "RunExample" span (input, call_my_code, output) and one child "ScoreExample"
        span for scoring, so the server sees a clear call_my_code vs scoring split (aligned with client-go).

        Args:
            example: The example to run. See Example.ts type
            call_my_code: Function that takes input and parameters, returns output (can be async)
            scorer_for_metric_id: Optional dictionary of metric IDs to functions that score the output given the example and parameters

        Returns:
            List of one result (for API compatibility).
        """
        if not self.experiment:
            self.create_experiment()
        if not self.experiment:
            raise Exception("Failed to create experiment")

        parameters_here = self.experiment.get("parameters") or {}
        input_data = example.get("input")
        if not input_data and example.get("spans") and len(example["spans"]) > 0:
            input_data = example["spans"][0].get("attributes", {}).get("input")
        if not input_data:
            print(f"Warning: Example has no input field or spans with input attribute: {example}")

        example_id = example.get("id")
        if not example_id:
            raise ValueError("Example must have an 'id' field")

        print(f"Running with parameters: {parameters_here}")
        original_env_vars: Dict[str, Optional[str]] = {}
        for key, value in parameters_here.items():
            if value:
                original_env_vars[key] = os.environ.get(key)
                os.environ[key] = str(value)
        try:
            start = time.time() * 1000

            run_trace_id_ref: List[Optional[str]] = [None]

            # Wrap engine to match run_example signature (input, parameters)
            # Root span so server can find it by parent:unset; trace ID is sent to scoreAndStore
            def set_trace_id(tid: Optional[str]) -> None:
                run_trace_id_ref[0] = tid

            @WithTracing(root=True)
            async def wrapped_engine(input_data, parameters, set_trace_id: Callable[[Optional[str]], None]):
                trace_id_here = get_active_trace_id()
                set_trace_id(trace_id_here)
                result = call_my_code(input_data, parameters)
                # Handle async functions
                if hasattr(result, "__await__"):
                    result = await result
                return result

            output = wrapped_engine(input_data, parameters_here, set_trace_id)
            if hasattr(output, "__await__"):
                output = await output
            duration = int((time.time() * 1000) - start)
            print(f"Output: {output}")

            dataset_metrics = self.get_dataset().get("metrics", [])
            specific_metrics = example.get("metrics", [])
            metrics = [*dataset_metrics, *specific_metrics]
            result: Result = {"example": example_id, "scores": {}, "messages": {}, "errors": {}}
            for metric in metrics:
                metric_id = metric.get("id")
                score_key = _metric_score_key(metric)
                if not metric_id or not score_key:
                    continue
                scorer = scorer_for_metric_id.get(metric_id) if scorer_for_metric_id else None
                if scorer:
                    metric_result = await scorer(input_data, output, metric)
                elif metric.get("type") == "llm":
                    metric_result = await self._score_llm_metric(input_data, output, example, metric)
                else:
                    continue
                if not metric_result:
                    result["errors"][score_key] = "Scoring function returned None"
                    continue
                result["scores"][score_key] = metric_result.get("score")
                result["messages"][score_key] = metric_result.get("message")
                result["errors"][score_key] = metric_result.get("error")
            result["scores"]["duration"] = duration
            await flush_tracing()
            print(f"Call scoreAndStore ... for example: {example_id} with scores: {result['scores']}")
            result = await self.score_and_store(example, output, result, trace_id=run_trace_id_ref[0])
            print(f"scoreAndStore returned: {result}")
            return [result]
        finally:
            for key, original_value in original_env_vars.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def get_summaries(self) -> Dict[str, Any]:
        """
        Get summaries from the experiment.

        Returns:
            Dictionary of metric names to summary statistics
        """
        if not self.experiment_id:
            raise ValueError("No experiment ID available. Create an experiment first.")
        
        response = requests.get(
            f"{self.server_url}/experiment/{self.experiment_id}",
            headers=self._get_headers(),
        )
    
        if not response.ok:
            raise Exception(format_http_error(response, "fetch summary results"))

        experiment2 = response.json()
        return experiment2.get("summaries", {})

    async def _score_llm_metric(
        self,
        input_data: Any,
        output: Any,
        example: Example,
        metric: Metric,
    ) -> MetricResult:
        """
        Score an LLM metric by fetching model API key from server if needed.
        
        Args:
            input_data: The input data to score
            output: The output to score
            example: The example object
            metric: The metric definition
            
        Returns:
            MetricResult object with score:[0,1], message (optional), and error (optional)
        """
        # If model is specified, try to fetch API key from server
        model_id = metric.get("model")
        api_key = None
        provider = metric.get("provider")
        
        if model_id:
            model_data = await get_model_from_server(
                model_id, self.server_url, self._get_headers()
            )
            if model_data:
                # Server returns 'apiKey' (camelCase)
                api_key = model_data.get("apiKey")
                # If provider not set in metric, try to get it from model
                if not provider and model_data.get("provider"):
                    provider = model_data.get("provider")
        
        # Create a custom llm_call_fn if we have an API key from the model
        llm_call_fn = self.llm_call_fn
        if api_key and not llm_call_fn:
            async def _model_llm_call(system_prompt: str, user_message: str) -> str:
                return await call_llm_fallback(system_prompt, user_message, api_key, provider)
            llm_call_fn = _model_llm_call
        
        return await score_llm_metric_local(
            input_data, output, example, metric, llm_call_fn
        )


