from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import os
from typing import Optional
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Business Strategy Generator API",
    description="API that generates business strategies based on industry, company size, and goals using OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to specific origins like ["http://localhost:4200"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")


# Input model
class StrategyRequest(BaseModel):
    industry: str = Field(..., description="The industry of the business")
    size: str = Field(..., description="The size of the business (small, medium, large)")
    goal: str = Field(..., description="The primary business goal")
    additional_context: Optional[str] = Field(None, description="Any additional context or constraints")
    # Add St. Gallen Model specific fields
    persona: str = Field(..., description="The role of the person making the decision")
    market: str = Field(..., description="The market scope for the business")
    technology_adoption: str = Field(..., description="The level of technology adoption")
    knowledge_domain: str = Field(..., description="The focus area of expertise")
    decision_description: str = Field(..., description="Description of the decision to be made")


# Output model
class StrategyResponse(BaseModel):
    strategy: dict = Field(..., description="Generated business strategy as a JSON object")


# Helper function to generate prompt based on parameters
def generate_prompt(industry: str, size: str, goal: str, additional_context: Optional[str] = None) -> str:
    prompt = f"""
    Generate a detailed business strategy for a {size} business in the {industry} industry.
    The primary goal is: {goal}.
    """

    if additional_context:
        prompt += f"\nAdditional context: {additional_context}"

    return prompt.strip()


# Dependency for OpenAI API client
async def get_openai_client():
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client



# Endpoint for Post requests
@app.post("/overview", response_model=StrategyResponse)
async def generate_strategy(
        request: StrategyRequest,
        client: httpx.AsyncClient = Depends(get_openai_client)
) -> StrategyResponse:
    """
    Generate strategic advice based on the St. Gallen Management Model.
    """
    # Build the prompt using the request parameters
    prompt = (
        "As a management consultant expert in the St. Gallen Management Model, provide strategic advice for:\n\n"
        f"Industry: {request.industry}\n"
        f"Role: {request.persona}\n"
        f"Market scope: {request.market}\n"
        f"Company size: {request.size}\n"
        f"Tech adoption: {request.technology_adoption}\n"
        f"Focus area: {request.knowledge_domain}\n"
        f"Primary goal: {request.goal}\n"
        f"Decision: {request.decision_description}\n"
    )

    if request.additional_context:
        prompt += f"\nAdditional context: {request.additional_context}\n\n"
    else:
        prompt += "\n\n"

    prompt += (
        "Format response as JSON with these sections:\n"
        "- Overview: Brief context and challenge summary giving  3 or 5 management Challenges and opportunity areas\n"
       "Ex:{ managementChallenges: [{ name: value,description:value,strategy:value}]opportunityAreas:[{name:value,description:value,action:value}]}"

    )

    try:
        # Call the OpenAI API with the prompt
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4",  # You can also use "gpt-3.5-turbo" if needed
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a management consultant expert in the St. Gallen Management Model."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                        "temperature" : 1,
                    "max_tokens" : 5048,
                    "top_p" : 1,
                    "store" : True
            }
        )

        response.raise_for_status()
        data = response.json()

        # Extract the generated text from the response
        strategy_content = data["choices"][0]["message"]["content"]

        # Parse the JSON response from the model
        try:
            strategy_json = json.loads(strategy_content)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error decoding JSON from OpenAI response: {str(e)}. Received content: {strategy_content}"
            )

        # Add the vector store id to the JSON response
        strategy_json["vector_store_id"] = "vs_67dbd6e325c88191ba2315ac5b40f9b8"
        return StrategyResponse(strategy=strategy_json)

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"OpenAI API error: {e.response.text}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Request error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

    # Endpoint for Post requests
@app.post("/enviroment_analisy-economy", response_model=StrategyResponse)
async def generate_strategy(
            request: StrategyRequest,
            client: httpx.AsyncClient = Depends(get_openai_client)
    ) -> StrategyResponse:
        """
        Generate strategic advice based on the St. Gallen Management Model.
        """
        # Build the prompt using the request parameters
        prompt = (
            "As a management consultant expert in the St. Gallen Management Model, provide strategic advice for: \n\n"
            f"Industry: {request.industry}\n"
            f"Role: {request.persona}\n"
            f"Market scope: {request.market}\n"
            f"Company size: {request.size}\n"
            f"Tech adoption: {request.technology_adoption}\n"
            f"Focus area: {request.knowledge_domain}\n"
            f"Primary goal: {request.goal}\n"
            f"Decision: {request.decision_description}\n"
        )

        if request.additional_context:
            prompt += f"\nAdditional context: {request.additional_context}\n\n"
        else:
            prompt += "\n\n"

        prompt += (
            "Format response as JSON with these sections:\n"
            "Overview: Brief context and challenge summary - Environment analysis of the impact of decision\n"
            "Ex:{ impact: [{ name: value,description:value,strategy:value}] strict follow the example of data return only impact"
        )

        try:
            # Call the OpenAI API with the prompt
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",  # You can also use "gpt-3.5-turbo" if needed
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a management consultant expert in the St. Gallen Management Model."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature" : 1,
                    "max_tokens" : 5048,
                    "top_p" : 1,

                }
            )

            response.raise_for_status()
            data = response.json()

            # Extract the generated text from the response
            strategy_content = data["choices"][0]["message"]["content"]

            # Parse the JSON response from the model
            try:
                strategy_json = json.loads(strategy_content)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error decoding JSON from OpenAI response: {str(e)}. Received content: {strategy_content}"
                )

            # Add the vector store id to the JSON response
            strategy_json["vector_store_id"] = "vs_67dbd6e325c88191ba2315ac5b40f9b8"
            return StrategyResponse(strategy=strategy_json)

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"OpenAI API error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Request error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
@app.post("/enviroment_analisy-tech", response_model=StrategyResponse)
async def generate_strategy(
            request: StrategyRequest,
            client: httpx.AsyncClient = Depends(get_openai_client)
    ) -> StrategyResponse:
        """
        Generate strategic advice based on the St. Gallen Management Model.
        """
        # Build the prompt using the request parameters
        prompt = (
            "As a management consultant expert in the St. Gallen Management Model, provide strategic advice for:\n\n"
            f"Industry: {request.industry}\n"
            f"Role: {request.persona}\n"
            f"Market scope: {request.market}\n"
            f"Company size: {request.size}\n"
            f"Tech adoption: {request.technology_adoption}\n"
            f"Focus area: {request.knowledge_domain}\n"
            f"Primary goal: {request.goal}\n"
            f"Decision: {request.decision_description}\n"
        )

        if request.additional_context:
            prompt += f"\nAdditional context: {request.additional_context}\n\n"
        else:
            prompt += "\n\n"

        prompt += (
            "Format response as JSON with these sections:\n"
            "Overview: Brief context and challenge summary - Environment analysis of the impact of decision in technology sphere\n"
            "Ex:{ impact: [{ name: value,description:value,strategy:value}]"

        )

        try:
            # Call the OpenAI API with the prompt
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",  # You can also use "gpt-3.5-turbo" if needed
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a management consultant expert in the St. Gallen Management Model."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature" : 1,
                    "max_tokens" : 5048,
                    "top_p" : 1,
                    "store" : True
                }
            )

            response.raise_for_status()
            data = response.json()

            # Extract the generated text from the response
            strategy_content = data["choices"][0]["message"]["content"]

            # Parse the JSON response from the model
            try:
                strategy_json = json.loads(strategy_content)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error decoding JSON from OpenAI response: {str(e)}. Received content: {strategy_content}"
                )

            # Add the vector store id to the JSON response
            strategy_json["vector_store_id"] = "vs_67dbd6e325c88191ba2315ac5b40f9b8"
            return StrategyResponse(strategy=strategy_json)

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"OpenAI API error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Request error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )


@app.post("/enviroment_analisy-nature", response_model=StrategyResponse)
async def generate_strategy(
            request: StrategyRequest,
            client: httpx.AsyncClient = Depends(get_openai_client)
    ) -> StrategyResponse:
        """
        Generate strategic advice based on the St. Gallen Management Model.
        """
        # Build the prompt using the request parameters
        prompt = (
            "As a management consultant expert in the St. Gallen Management Model, provide strategic advice for:\n\n"
            f"Industry: {request.industry}\n"
            f"Role: {request.persona}\n"
            f"Market scope: {request.market}\n"
            f"Company size: {request.size}\n"
            f"Tech adoption: {request.technology_adoption}\n"
            f"Focus area: {request.knowledge_domain}\n"
            f"Primary goal: {request.goal}\n"
            f"Decision: {request.decision_description}\n"
        )

        if request.additional_context:
            prompt += f"\nAdditional context: {request.additional_context}\n\n"
        else:
            prompt += "\n\n"

        prompt += (
            "Format response as JSON with these sections:\n"
            "Overview: Brief context and challenge summary - Environment analysis of the impact of decision in nature sphere\n"
            "Ex:{ impact: [{ name: value,description:value,strategy:value}]"

        )

        try:
            # Call the OpenAI API with the prompt
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",  # You can also use "gpt-3.5-turbo" if needed
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a management consultant expert in the St. Gallen Management Model."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature" : 1,
                    "max_tokens" : 5048,
                    "top_p" : 1,
                    "store" : True
                }
            )

            response.raise_for_status()
            data = response.json()

            # Extract the generated text from the response
            strategy_content = data["choices"][0]["message"]["content"]

            # Parse the JSON response from the model
            try:
                strategy_json = json.loads(strategy_content)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error decoding JSON from OpenAI response: {str(e)}. Received content: {strategy_content}"
                )

            # Add the vector store id to the JSON response
            strategy_json["vector_store_id"] = "vs_67dbd6e325c88191ba2315ac5b40f9b8"
            return StrategyResponse(strategy=strategy_json)

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"OpenAI API error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Request error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
@app.post("/enviroment_analisy-society", response_model=StrategyResponse)
async def generate_strategy(
            request: StrategyRequest,
            client: httpx.AsyncClient = Depends(get_openai_client)
    ) -> StrategyResponse:
        """
        Generate strategic advice based on the St. Gallen Management Model.
        """
        # Build the prompt using the request parameters
        prompt = (
            "As a management consultant expert in the St. Gallen Management Model, provide strategic advice for:\n\n"
            f"Industry: {request.industry}\n"
            f"Role: {request.persona}\n"
            f"Market scope: {request.market}\n"
            f"Company size: {request.size}\n"
            f"Tech adoption: {request.technology_adoption}\n"
            f"Focus area: {request.knowledge_domain}\n"
            f"Primary goal: {request.goal}\n"
            f"Decision: {request.decision_description}\n"
        )

        if request.additional_context:
            prompt += f"\nAdditional context: {request.additional_context}\n\n"
        else:
            prompt += "\n\n"

        prompt += (
            "Format response as JSON with these sections:\n"
            "Overview: Brief context and challenge summary - Environment analysis of the impact of decision in society sphere\n"
            "Ex:{ impact: [{ name: value,description:value,strategy:value}]"

        )

        try:
            # Call the OpenAI API with the prompt
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",  # You can also use "gpt-3.5-turbo" if needed
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a management consultant expert in the St. Gallen Management Model."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature" : 1,
                    "max_tokens" : 5048,
                    "top_p" : 1,
                    "store" : True
                }
            )

            response.raise_for_status()
            data = response.json()

            # Extract the generated text from the response
            strategy_content = data["choices"][0]["message"]["content"]

            # Parse the JSON response from the model
            try:
                strategy_json = json.loads(strategy_content)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error decoding JSON from OpenAI response: {str(e)}. Received content: {strategy_content}"
                )

            # Add the vector store id to the JSON response
            strategy_json["vector_store_id"] = "vs_67dbd6e325c88191ba2315ac5b40f9b8"
            return StrategyResponse(strategy=strategy_json)

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"OpenAI API error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Request error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )

@app.post("/enviroment_analisy-interaction", response_model=StrategyResponse)
async def generate_strategy(
            request: StrategyRequest,
            client: httpx.AsyncClient = Depends(get_openai_client)
    ) -> StrategyResponse:
        """
        Generate strategic advice based on the St. Gallen Management Model.
        """
        # Build the prompt using the request parameters
        prompt = (
            "As a management consultant expert in the St. Gallen Management Model, provide strategic advice for:\n\n"
            f"Industry: {request.industry}\n"
            f"Role: {request.persona}\n"
            f"Market scope: {request.market}\n"
            f"Company size: {request.size}\n"
            f"Tech adoption: {request.technology_adoption}\n"
            f"Focus area: {request.knowledge_domain}\n"
            f"Primary goal: {request.goal}\n"
            f"Decision: {request.decision_description}\n"
        )

        if request.additional_context:
            prompt += f"\nAdditional context: {request.additional_context}\n\n"
        else:
            prompt += "\n\n"

        prompt += (
            "Format response as JSON with these sections:\n"
            "Overview: Brief context and challenge summary - interaction issue  analysis of the impact of decision give resources concerns and interests\n"
            "Ex:{resourse:[{name: value, description:value}],concernsIntests:{name:value,description:value}"
        )

        try:
            # Call the OpenAI API with the prompt
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",  # You can also use "gpt-3.5-turbo" if needed
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a management consultant expert in the St. Gallen Management Model."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature" : 1,
                    "max_tokens" : 5048,
                    "top_p" : 1,
                    "store" : True
                }
            )

            response.raise_for_status()
            data = response.json()

            # Extract the generated text from the response
            strategy_content = data["choices"][0]["message"]["content"]

            # Parse the JSON response from the model
            try:
                strategy_json = json.loads(strategy_content)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error decoding JSON from OpenAI response: {str(e)}. Received content: {strategy_content}"
                )

            # Add the vector store id to the JSON response
            strategy_json["vector_store_id"] = "vs_67dbd6e325c88191ba2315ac5b40f9b8"
            return StrategyResponse(strategy=strategy_json)

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"OpenAI API error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Request error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
@app.post("/organization-processes", response_model=StrategyResponse)
async def generate_strategy(
            request: StrategyRequest,
            client: httpx.AsyncClient = Depends(get_openai_client)
    ) -> StrategyResponse:
        """
        Generate strategic advice based on the St. Gallen Management Model.
        """
        # Build the prompt using the request parameters
        prompt = (
            "As a management consultant expert in the St. Gallen Management Model, provide strategic advice for:\n\n"
            f"Industry: {request.industry}\n"
            f"Role: {request.persona}\n"
            f"Market scope: {request.market}\n"
            f"Company size: {request.size}\n"
            f"Tech adoption: {request.technology_adoption}\n"
            f"Focus area: {request.knowledge_domain}\n"
            f"Primary goal: {request.goal}\n"
            f"Decision: {request.decision_description}\n"
        )

        if request.additional_context:
            prompt += f"\nAdditional context: {request.additional_context}\n\n"
        else:
            prompt += "\n\n"

        prompt += (
            "Format response as JSON with these sections:\n"
            "Overview: Brief context and challenge summary - organization process analysis of the impact of decision give  me management,business,suport process\n"
            "Ex:{managementProcess:[{name: value, description:value}],businessProcess:{name:value,description:value,suportProcess:{name:value,description:value}}"
        )

        try:
            # Call the OpenAI API with the prompt
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",  # You can also use "gpt-3.5-turbo" if needed
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a management consultant expert in the St. Gallen Management Model."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature" : 1,
                    "max_tokens" : 5048,
                    "top_p" : 1,
                    "store" : True
                }
            )

            response.raise_for_status()
            data = response.json()

            # Extract the generated text from the response
            strategy_content = data["choices"][0]["message"]["content"]

            # Parse the JSON response from the model
            try:
                strategy_json = json.loads(strategy_content)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error decoding JSON from OpenAI response: {str(e)}. Received content: {strategy_content}"
                )

            # Add the vector store id to the JSON response
            strategy_json["vector_store_id"] = "vs_67dbd6e325c88191ba2315ac5b40f9b8"
            return StrategyResponse(strategy=strategy_json)

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"OpenAI API error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Request error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
@app.post("/management-practices", response_model=StrategyResponse)
async def generate_strategy(
            request: StrategyRequest,
            client: httpx.AsyncClient = Depends(get_openai_client)
    ) -> StrategyResponse:
        """
        Generate strategic advice based on the St. Gallen Management Model.
        """
        # Build the prompt using the request parameters
        prompt = (
            "As a management consultant expert in the St. Gallen Management Model, provide strategic advice for:\n\n"
            f"Industry: {request.industry}\n"
            f"Role: {request.persona}\n"
            f"Market scope: {request.market}\n"
            f"Company size: {request.size}\n"
            f"Tech adoption: {request.technology_adoption}\n"
            f"Focus area: {request.knowledge_domain}\n"
            f"Primary goal: {request.goal}\n"
            f"Decision: {request.decision_description}\n"
        )

        if request.additional_context:
            prompt += f"\nAdditional context: {request.additional_context}\n\n"
        else:
            prompt += "\n\n"

        prompt += (
            "Format response as JSON with these sections:\n"
            "Overview: Brief context and challenge summary - management pratices analysis of the impact of decision give  me key chalanges,recomended aproaches bases on Role,Market scope,Tech adoption,Company size \n"
            "Ex:{keyChallages:[{name: value, description:value}],recommendedAproaches:{name:value,description:value}}"

        )

        try:
            # Call the OpenAI API with the prompt
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",  # You can also use "gpt-3.5-turbo" if needed
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a management consultant expert in the St. Gallen Management Model."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature" : 1,
                    "max_tokens" : 5048,
                    "top_p" : 1,
                    "store" : True
                }
            )

            response.raise_for_status()
            data = response.json()

            # Extract the generated text from the response
            strategy_content = data["choices"][0]["message"]["content"]

            # Parse the JSON response from the model
            try:
                strategy_json = json.loads(strategy_content)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error decoding JSON from OpenAI response: {str(e)}. Received content: {strategy_content}"
                )

            # Add the vector store id to the JSON response
            strategy_json["vector_store_id"] = "vs_67dbd6e325c88191ba2315ac5b40f9b8"
            return StrategyResponse(strategy=strategy_json)

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"OpenAI API error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Request error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
# Add documentation endpoint
@app.get("/docs/openapi.json", include_in_schema=False)
async def get_openapi_schema():
    return app.openapi()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
