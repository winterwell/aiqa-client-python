"""
ExperimentRunner - runs experiments on datasets and scores results
"""

import os
import time
import asyncio
from .constants import LOG_TAG
from .http_utils import build_headers, get_server_url, get_api_key, format_http_error
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from .tracing import WithTracing
from .span_helpers import set_span_attribute, flush_tracing
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



def _filter_input_for_run(input_data: Any) -> Dict[str, Any]:
    """Tracing:Filter input - drop most, keep just ids"""
    if not isinstance(input_data, dict):
        return {}
    self_obj = input_data.get("self")
    if not self_obj:
        return {}
    return {
        "dataset": getattr(self_obj, "dataset_id", None),
        "experiment": getattr(self_obj, "experiment_id", None),
    }


def _filter_input_for_run_example(
    self: "ExperimentRunner", 
    example: Dict[str, Any],
    call_my_code: Any = None,
    score_this_output: Any = None,
) -> Dict[str, Any]:
    """Filter input for run_example method to extract dataset, experiment, and example IDs."""
    result = _filter_input_for_run({"self": self})
    if isinstance(example, dict):
        result["example"] = example.get("id")
    return result


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

    def get_example_inputs(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Fetch example inputs from the dataset.

        Args:
            limit: Maximum number of examples to fetch (default: 10000)

        Returns:
            List of example objects
        """
        params = {
            "dataset_id": self.dataset_id,
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
                - comparison_parameters

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
            "summary_results": {},
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
            example_id = example.get("id")
            if not example_id:
                raise ValueError("Example must have an 'id' field")
            result = Result(exampleId=example_id, scores={}, messages={}, errors={})
        scores = result.get("scores") or {}

        
        
        print(f"Scoring and storing example: {example_id}")
        print(f"Scores: {scores}")

        # Run synchronous requests.post in a thread pool to avoid blocking
        def _do_request():
            return requests.post(
                f"{self.server_url}/experiment/{self.experiment_id}/example/{example_id}/scoreAndStore",
                json={
                    "output": result,
                    "traceId": example.get("traceId"),
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

    @WithTracing(filter_input=_filter_input_for_run)
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
        examples = self.get_example_inputs()

        # Wrap engine to match run_example signature (input, parameters)
        async def wrapped_engine(input_data, parameters):
            result = call_my_code(input_data, parameters)
            # Handle async functions
            if hasattr(result, "__await__"):
                result = await result
            return result

        for example in examples:
            try:
                scores = await self.run_example(example, wrapped_engine, scorer_for_metric_id)
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

    @WithTracing(filter_input=_filter_input_for_run_example)
    async def run_example(
        self,
        example: Example,
        call_my_code: CallMyCodeType,
        scorer_for_metric_id: Optional[Dict[str, ScoreThisInputOutputMetricType]] = None,
    ) -> List[Result]:
        """
        Run the engine on an example with the given parameters (looping over comparison parameters),
        and score the result. Also calls scoreAndStore to store the result in the server.

        Args:
            example: The example to run. See Example.ts type
            call_my_code: Function that takes input and parameters, returns output (can be async)
            scorer_for_metric_id: Optional dictionary of metric IDs to functions that score the output given the example and parameters

        Returns:
            One set of scores for each comparison parameter set. If no comparison parameters,
            returns an array of one.
        """
        # Ensure experiment exists
        if not self.experiment:
            self.create_experiment()
        if not self.experiment:
            raise Exception("Failed to create experiment")

        # Make the parameters
        parameters_fixed = self.experiment.get("parameters") or {}
        # If comparison_parameters is empty/undefined, default to [{}] so we run at least once
        parameters_loop = self.experiment.get("comparison_parameters") or [{}]

        # Handle both spans array and input field
        input_data = example.get("input")
        if not input_data and example.get("spans") and len(example["spans"]) > 0:
            input_data = example["spans"][0].get("attributes", {}).get("input")

        if not input_data:
            print(f"Warning: Example has no input field or spans with input attribute: {example}"
            )
            # Run engine anyway -- this could make sense if it's all about the parameters

        # Set example.id on the root span (created by @WithTracing decorator)
        # This ensures the root span from the trace has example=Example.id set
        example_id = example.get("id")
        if not example_id:
            raise ValueError("Example must have an 'id' field")
        set_span_attribute("example", example_id)
        
        all_scores: List[Dict[str, Any]] = []
        dataset_metrics = self.get_dataset().get("metrics", [])
        specific_metrics = example.get("metrics", [])
        metrics = [*dataset_metrics, *specific_metrics]
        # This loop should not be parallelized - it should run sequentially, one after the other
        # to avoid creating interference between the runs.
        for parameters in parameters_loop:
            parameters_here = {**parameters_fixed, **parameters}
            print(f"Running with parameters: {parameters_here}")

            # Save original env var values for cleanup
            original_env_vars: Dict[str, Optional[str]] = {}
            # Set env vars from parameters_here
            for key, value in parameters_here.items():
                if value:
                    original_env_vars[key] = os.environ.get(key)
                    os.environ[key] = str(value)

            try:
                start = time.time() * 1000  # milliseconds
                output = call_my_code(input_data, parameters_here)
                # Handle async functions
                if hasattr(output, "__await__"):
                    output = await output
                end = time.time() * 1000  # milliseconds
                duration = int(end - start)

                print(f"Output: {output}")
                # Score it
                result = Result(exampleId=example_id, scores={}, messages={}, errors={})
                for metric in metrics:
                    metric_id = metric.get("id")
                    if not metric_id:
                        print(f"Warning: Metric missing 'id' field, skipping: {metric}")
                        continue
                    scorer = scorer_for_metric_id.get(metric_id) if scorer_for_metric_id else None
                    if scorer:
                        metric_result = await scorer(input_data, output, metric)
                    elif metric.get("type") == "llm":
                        metric_result = await self._score_llm_metric(input_data, output, example, metric)
                    else:
                        metric_type = metric.get("type", "unknown")
                        print(f"Skipping metric: {metric_id} {metric_type} - no scorer")
                        continue
                    
                    # Handle None metric_result (e.g., if scoring failed)
                    if not metric_result:
                        print(f"Warning: Metric {metric_id} returned None result, skipping")
                        result["errors"][metric_id] = "Scoring function returned None"
                        continue
                    
                    result["scores"][metric_id] = metric_result.get("score")
                    result["messages"][metric_id] = metric_result.get("message")
                    result["errors"][metric_id] = metric_result.get("error")
                # Always add duration to scores as a system metric
                result["scores"]["duration"] = duration

                # Flush spans before scoreAndStore to ensure they're indexed in ES
                # This prevents race condition where scoreAndStore looks up spans before they're indexed
                await flush_tracing()

                print(f"Call scoreAndStore ... for example: {example_id} with scores: {result['scores']}")
                result = await self.score_and_store(example, output, result)
                print(f"scoreAndStore returned: {result}")
                all_scores.append(result)
            finally:
                # Restore original env var values
                for key, original_value in original_env_vars.items():
                    if original_value is None:
                        # Variable didn't exist before, remove it
                        os.environ.pop(key, None)
                    else:
                        # Restore original value
                        os.environ[key] = original_value

        return all_scores

    def get_summary_results(self) -> Dict[str, Any]:
        """
        Get summary results from the experiment.

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
        return experiment2.get("summary_results", {})

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
                api_key = model_data.get("api_key")
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


