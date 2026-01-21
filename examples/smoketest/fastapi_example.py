from fastapi import FastAPI, Request
from pydantic import BaseModel
import asyncio

from dotenv import load_dotenv
import os
load_dotenv()

# Import tracing decorators/utilities from your aiqa client
from aiqa import WithTracing, set_span_attribute, flush_tracing

app = FastAPI()

class MathInput(BaseModel):
    x: int
    y: int

@app.get("/")
@WithTracing
def root():
    return {"message": "Hello from FastAPI + AIQA tracing!"}

@app.post("/sum")
@WithTracing
def sum_numbers(data: MathInput):
    result = data.x + data.y
    set_span_attribute("operation", "sum")
    set_span_attribute("x", data.x)
    set_span_attribute("y", data.y)
    return {"result": result}

@app.post("/mul")
@WithTracing
async def multiply_numbers(data: MathInput):
    await asyncio.sleep(0.1)
    result = data.x * data.y
    set_span_attribute("operation", "multiply")
    set_span_attribute("x", data.x)
    set_span_attribute("y", data.y)
    return {"result": result}

@app.get("/flush")
async def flush_endpoint():
    await flush_tracing()
    return {"status": "flushed"}

# If you want to run directly: uvicorn fastapi_example:app --reload

