

from typing import TypedDict, Dict, Optional, Awaitable, Callable, Any, Union, Annotated, List
from numbers import Number

class Metric(TypedDict):
  """Definition of a metric to score. See Metric.ts for more details."""
  id: str
  type: str
  name: Optional[str] = None
  description: Optional[str] = None
  provider: Optional[str] = None
  model: Optional[str] = None
  prompt: Optional[str] = None
  code: Optional[str] = None

class Example(TypedDict):
  """Definition of an example to score. See Example.ts for more details."""
  id: str
  input: Optional[str] = None
  spans: Optional[List[Dict[str, Any]]] = None
  metrics: Optional[List[Metric]] = None

class MetricResult(TypedDict):
  """Result of evaluating a metric on an output (i.e. a single metric for a single example)."""
  score: Annotated[Number, "Numeric score for the metric evaluation, typically a float in the range [0, 1]"]
  message: Optional[str] = None
  error: Optional[str] = None

class Result(TypedDict):
  """Result of evaluating a set of metrics on an output (i.e. the full set of metrics for a single example)."""
  exampleId: str
  scores: Dict[str, Number]
  messages: Optional[Dict[str, str]] = None
  errors: Optional[Dict[str, str]] = None


# Function that processes input and parameters to produce an output (sync or async)
# Args:
#   input: The input data for the example (typically a dict with the example's input fields)
#   parameters: Dictionary of parameters to pass to the function (e.g., model settings, temperature)
# Returns:
#   The output result, which can be any type. If the function is async, returns an Awaitable.
CallMyCodeType = Callable[[Any, Dict[str, Any]], Union[Any, Awaitable[Any]]]

# Function that calls an LLM with a system prompt and user message (async)
# Args:
#   system_prompt: The system prompt/instructions for the LLM
#   user_message: The "user" message containing the content to process (e.g., input and output to score)
# Returns:
#   The raw response content string from the LLM (typically JSON for scoring)
CallLLMType = Callable[[str, str], Awaitable[str]]

# Function that scores a given output, using input, example, and parameters (usually async)
# Args:
#   input: The input data for the example (typically a dict with the example's input fields)
#   output: The output to score
#   metric: The metric to score
# Returns:
#   MetricResult object with score:[0,1], message (optional), and error (optional)
ScoreThisInputOutputMetricType = Callable[[Any, Any, Metric], Awaitable[MetricResult]]