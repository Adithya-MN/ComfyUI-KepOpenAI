from typing import Tuple
import torch
from openai import Client as OpenAIClient
from .lib import image

class ImageWithPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Image": ("IMAGE", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Generate a high quality caption for the image. The most important aspects of the image should be described first. If needed, weights can be applied to the caption in the following format: '(word or phrase:weight)', where the weight should be a float less than 2.",
                    },
                ),
                "max_tokens": ("INT", {"min": 1, "max": 2048, "default": 77}),
                "OPENAI_API_KEY": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gpt-4-vision-preview"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_completion"
    CATEGORY = "OpenAI"

    def generate_completion(
        self, 
        Image: torch.Tensor, 
        prompt: str, 
        max_tokens: int,
        OPENAI_API_KEY: str,
        model: str
    ) -> Tuple[str]:
        
        # Create client with provided API key
        client = OpenAIClient(api_key=OPENAI_API_KEY)
        
        # Convert image to base64
        b64image = image.pil2base64(image.tensor2pil(Image))
        
        # Generate completion
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64image}"},
                        },
                    ],
                }
            ],
        )
        
        if len(response.choices) == 0:
            raise Exception("No response from OpenAI API")

        return (response.choices[0].message.content,)
