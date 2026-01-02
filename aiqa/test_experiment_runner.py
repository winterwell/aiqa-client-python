"""
Example usage of the ExperimentRunner class.
"""

import asyncio
import os
from dotenv import load_dotenv
from aiqa import ExperimentRunner

# Load environment variables
load_dotenv()


# A dummy test engine that returns a dummy response
async def my_engine(input_data):
    """
    Example engine function that simulates an API call.
    Note: For run(), the engine only takes input_data.
    For run_example(), you can use an engine that takes (input_data, parameters).
    """
    # Imitate an OpenAI API response
    # Sleep for random about 0.5 - 1 seconds
    import random

    sleep_time = random.random() * 0.5 + 0.5
    await asyncio.sleep(sleep_time)
    return {
        "choices": [
            {
                "message": {
                    "content": f"hello {input_data}",
                },
            },
        ],
    }


async def scorer(output, example):
    """
    Example scorer function that scores the output.
    In a real scenario, you would use the metrics from the dataset.
    Note: For run(), the scorer only takes (output, example).
    For run_example(), you can use a scorer that takes (output, example, parameters).
    """
    # This is a simple example - in practice, you'd use the metrics from the dataset
    # and call the scoring functions accordingly
    scores = {}
    # Add your scoring logic here
    return scores


async def example_basic_usage():
    """
    Basic example of using ExperimentRunner.
    """
    if not os.getenv("AIQA_API_KEY"):
        print("Warning: AIQA_API_KEY environment variable is not set. Example may fail.")

    dataset_id = "your-dataset-id-here"
    organisation_id = "your-organisation-id-here"

    experiment_runner = ExperimentRunner(
        dataset_id=dataset_id,
        organisation_id=organisation_id,
    )

    # Get metrics from the dataset
    dataset = experiment_runner.get_dataset()
    metrics = dataset.get("metrics", [])
    print(f"Found {len(metrics)} metrics in dataset: {[m['name'] for m in metrics]}")

    # Create scorer that scores all metrics from the dataset
    # (In practice, you'd implement this based on your metrics)
    async def dataset_scorer(output, example):
        # Use the metrics from the dataset to score
        # This is a placeholder - implement based on your actual metrics
        return await scorer(output, example)

    # Get example inputs
    example_inputs = experiment_runner.get_example_inputs()
    print(f"Processing {len(example_inputs)} examples")

    # Run experiments on each example
    for example in example_inputs:
        result = await experiment_runner.run_example(example, my_engine, dataset_scorer)
        if result and len(result) > 0:
            print(f"Scored example {example['id']}: {result}")
        else:
            print(f"No results for example {example['id']}")

    # Get summary results
    summary_results = experiment_runner.get_summary_results()
    print(f"Summary results: {summary_results}")


async def example_with_experiment_setup():
    """
    Example of creating an experiment with custom setup.
    """
    dataset_id = "your-dataset-id-here"
    organisation_id = "your-organisation-id-here"

    experiment_runner = ExperimentRunner(
        dataset_id=dataset_id,
        organisation_id=organisation_id,
    )

    # Create experiment with custom parameters
    experiment = experiment_runner.create_experiment(
        {
            "name": "My Custom Experiment",
            "parameters": {
                "model": "gpt-4",
                "temperature": 0.7,
            },
            "comparison_parameters": [
                {"temperature": 0.5},
                {"temperature": 0.9},
            ],
        }
    )

    print(f"Created experiment: {experiment['id']}")

    # Now run the experiment
    await experiment_runner.run(my_engine, scorer)


async def example_stepwise():
    """
    Example of running experiments step by step (more control).
    """
    dataset_id = "your-dataset-id-here"
    organisation_id = "your-organisation-id-here"

    experiment_runner = ExperimentRunner(
        dataset_id=dataset_id,
        organisation_id=organisation_id,
    )

    # Get the dataset
    dataset = experiment_runner.get_dataset()
    metrics = dataset.get("metrics", [])
    print(f"Found {len(metrics)} metrics in dataset")

    # Create scorer for run_example (takes parameters)
    async def my_scorer(output, example, parameters):
        # Implement your scoring logic here
        # Note: run_example() passes parameters, so this scorer can use them
        return {"score": 0.8}  # Placeholder

    # Get examples
    examples = experiment_runner.get_example_inputs(limit=100)
    print(f"Processing {len(examples)} examples")

    # Process each example individually
    for example in examples:
        try:
            result = await experiment_runner.run_example(example, my_engine, my_scorer)
            print(f"Example {example['id']} completed: {result}")
        except Exception as e:
            print(f"Example {example['id']} failed: {e}")

    # Get final summary
    summary = experiment_runner.get_summary_results()
    print(f"Final summary: {summary}")


if __name__ == "__main__":
    # Uncomment the example you want to run:
    # asyncio.run(example_basic_usage())
    # asyncio.run(example_with_experiment_setup())
    # asyncio.run(example_stepwise())
    print("Please uncomment one of the examples above to run it.")
    print("Make sure to set your dataset_id and organisation_id in the example functions.")

