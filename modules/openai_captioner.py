import base64
import json
import requests
from typing import List, Optional
from PIL import Image
import io
import time


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64


def caption_image_with_openai(
    image: Image.Image,
    api_endpoint: str,
    api_key: str,
    model: str = "gpt-4o",
    prompt: str = "Please describe this image in detail.",
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: int = 60,
    verbose: bool = False
) -> str:
    """
    Caption an image using an OpenAI-compatible API endpoint.
    
    Args:
        image: PIL Image to caption
        api_endpoint: URL of the OpenAI-compatible API endpoint
        api_key: API key for authentication
        model: Model name to use (default: gpt-4o)
        prompt: Prompt to send with the image
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Request timeout in seconds
        verbose: Whether to print verbose output
    
    Returns:
        Generated caption as string
    """
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(image)
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    if verbose:
        print(f"Sending request to {api_endpoint}")
        print(f"Using model: {model}")
    
    # Make the request with retry logic
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if verbose:
                    print("Request successful")
                
                # Extract the caption from the response
                if "choices" in result and len(result["choices"]) > 0:
                    caption = result["choices"][0]["message"]["content"]
                    return caption.strip()
                else:
                    raise ValueError("No caption found in API response")
                    
            else:
                error_msg = f"API request failed with status {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
                    except:
                        error_msg += f": {response.text}"
                
                if attempt < max_retries - 1:
                    if verbose:
                        print(f"{error_msg}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception(error_msg)
                    
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"Request timed out. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception("Request timed out after all retry attempts")
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"Request failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"Request failed after all retry attempts: {e}")
    
    raise Exception("Failed to get caption after all retry attempts")


def load_openai_models() -> None:
    """
    Placeholder function for consistency with other captioner modules.
    OpenAI-compatible APIs don't require local model loading.
    """
    pass 