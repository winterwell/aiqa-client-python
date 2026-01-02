"""
ExperimentRunner - runs experiments on datasets and scores results
"""

import os
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
import requests


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
    ):
        """
        Initialize the ExperimentRunner.

        Args:
            dataset_id: ID of the dataset to run experiments on
            experiment_id: Usually unset, and a fresh experiment is created with a random ID
            server_url: URL of the AIQA server (defaults to AIQA_SERVER_URL env var)
            api_key: API key for authentication (defaults to AIQA_API_KEY env var)
            organisation_id: Organisation ID for the experiment
        """
        self.dataset_id = dataset_id
        self.experiment_id = experiment_id
        self.server_url = (server_url or os.getenv("AIQA_SERVER_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("AIQA_API_KEY", "")
        self.organisation = organisation_id
        self.experiment: Optional[Dict[str, Any]] = None
        self.scores: List[Dict[str, Any]] = []

    def _get_headers(self) -> Dict[str, str]:
        """Build HTTP headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
        return headers

    def get_dataset(self) -> Dict[str, Any]:
        """
        Fetch the dataset to get its metrics.

        Returns:
            The dataset object with metrics and other information
        """
        response = requests.get(
            f"{self.server_url}/dataset/{self.dataset_id}",
            headers=self._get_headers(),
        )

        if not response.ok:
            error_text = response.text if hasattr(response, "text") else "Unknown error"
            raise Exception(
                f"Failed to fetch dataset: {response.status_code} {response.reason} - {error_text}"
            )

        return response.json()

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
            error_text = response.text if hasattr(response, "text") else "Unknown error"
            raise Exception(
                f"Failed to fetch example inputs: {response.status_code} {response.reason} - {error_text}"
            )

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
        if not self.organisation or not self.dataset_id:
            raise Exception("Organisation and dataset ID are required to create an experiment")

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

        print("Creating experiment")
        response = requests.post(
            f"{self.server_url}/experiment",
            json=experiment_setup,
            headers=self._get_headers(),
        )

        if not response.ok:
            error_text = response.text if hasattr(response, "text") else "Unknown error"
            raise Exception(
                f"Failed to create experiment: {response.status_code} {response.reason} - {error_text}"
            )

        experiment = response.json()
        self.experiment_id = experiment["id"]
        self.experiment = experiment
        return experiment

    def score_and_store(
        self,
        example: Dict[str, Any],
        result: Any,
        scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ask the server to score an example result. Stores the score for later summary calculation.

        Args:
            example: The example object
            result: The output from running the engine on the example
            scores: Optional pre-computed scores

        Returns:
            The score result from the server
        """
        # Do we have an experiment ID? If not, we need to create the experiment first
        if not self.experiment_id:
            self.create_experiment()

        if scores is None:
            scores = {}

        print(f"Scoring and storing example: {example['id']}")
        print(f"Scores: {scores}")

        response = requests.post(
            f"{self.server_url}/experiment/{self.experiment_id}/example/{example['id']}/scoreAndStore",
            json={
                "output": result,
                "traceId": example.get("traceId"),
                "scores": scores,
            },
            headers=self._get_headers(),
        )

        if not response.ok:
            error_text = response.text if hasattr(response, "text") else "Unknown error"
            raise Exception(
                f"Failed to score and store: {response.status_code} {response.reason} - {error_text}"
            )

        json_result = response.json()
        print(f"scoreAndStore response: {json_result}")
        return json_result

    async def run(
        self,
        engine: Callable[[Any], Union[Any, Awaitable[Any]]],
        scorer: Optional[
            Callable[[Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]
        ] = None,
    ) -> None:
        """
        Run an engine function on all examples and score the results.

        Args:
            engine: Function that takes input, returns output (can be async)
            scorer: Optional function that scores the output given the example
        """
        examples = self.get_example_inputs()

        # Wrap engine to match run_example signature (input, parameters)
        def wrapped_engine(input_data, parameters):
            return engine(input_data)

        # Wrap scorer to match run_example signature (output, example, parameters)
        async def wrapped_scorer(output, example, parameters):
            if scorer:
                return await scorer(output, example)
            return {}

        for example in examples:
            scores = await self.run_example(example, wrapped_engine, wrapped_scorer)
            if scores:
                self.scores.append(
                    {
                        "example": example,
                        "result": scores,
                        "scores": scores,
                    }
                )

    async def run_example(
        self,
        example: Dict[str, Any],
        call_my_code: Callable[[Any, Dict[str, Any]], Union[Any, Awaitable[Any]]],
        score_this_output: Optional[
            Callable[[Any, Dict[str, Any], Dict[str, Any]], Awaitable[Dict[str, Any]]]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the engine on an example with the given parameters (looping over comparison parameters),
        and score the result. Also calls scoreAndStore to store the result in the server.

        Args:
            example: The example to run
            call_my_code: Function that takes input and parameters, returns output (can be async)
            score_this_output: Optional function that scores the output given the example and parameters

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
            print(
                f"Warning: Example has no input field or spans with input attribute: {example}"
            )
            # Run engine anyway -- this could make sense if it's all about the parameters

        all_scores: List[Dict[str, Any]] = []
        # This loop should not be parallelized - it should run sequentially, one after the other
        # to avoid creating interference between the runs.
        for parameters in parameters_loop:
            parameters_here = {**parameters_fixed, **parameters}
            print(f"Running with parameters: {parameters_here}")

            # Set env vars from parameters_here
            for key, value in parameters_here.items():
                if value:
                    os.environ[key] = str(value)

            start = time.time() * 1000  # milliseconds
            output = call_my_code(input_data, parameters_here)
            # Handle async functions
            if hasattr(output, "__await__"):
                import asyncio

                output = await output
            end = time.time() * 1000  # milliseconds
            duration = int(end - start)

            print(f"Output: {output}")

            scores: Dict[str, Any] = {}
            if score_this_output:
                scores = await score_this_output(output, example, parameters_here)

            scores["duration"] = duration

            # TODO: this call as async and wait for all to complete before returning
            print(f"Call scoreAndStore ... for example: {example['id']} with scores: {scores}")
            result = self.score_and_store(example, output, scores)
            print(f"scoreAndStore returned: {result}")
            all_scores.append(result)

        return all_scores

    def get_summary_results(self) -> Dict[str, Any]:
        """
        Get summary results from the experiment.

        Returns:
            Dictionary of metric names to summary statistics
        """
        response = requests.get(
            f"{self.server_url}/experiment/{self.experiment_id}",
            headers=self._get_headers(),
        )

        if not response.ok:
            error_text = response.text if hasattr(response, "text") else "Unknown error"
            raise Exception(
                f"Failed to fetch summary results: {response.status_code} {response.reason} - {error_text}"
            )

        experiment2 = response.json()
        return experiment2.get("summary_results", {})

