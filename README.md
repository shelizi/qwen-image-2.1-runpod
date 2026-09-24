# Qwen-Image 2.1 RunPod Serverless

[![Runpod](https://api.runpod.io/badge/shelizi/qwen-image-2.1-runpod)](https://console.runpod.io/hub/listing/shelizi/qwen-image-2.1-runpod)

RunPod Serverless worker for [KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF](https://huggingface.co/KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF), using the **Q4_K_M** diffusion file.

The container runs `stable-diffusion.cpp` on CUDA. It does not bake the weights into the image. On startup it loads only these three files:

| Role | File | Size |
| --- | --- | --- |
| Diffusion | `qwen-image-2.1-UC-Q4_K_M.gguf` | 4.6 GiB |
| Text encoder | `Qwen/Qwen3-VL-8B-Instruct-GGUF` `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | 5.0 GB |
| VAE | `vae/qwen_image_2.1_vae_bf16.safetensors` | 644 MiB |

The Hugging Face repo also contains the other quants and a 16 GiB BF16 text encoder. This worker does not download those.

## Deploy on RunPod

1. In [RunPod Settings](https://console.runpod.io/user/settings), connect GitHub and allow this repository.
2. Open Serverless and create an endpoint with **Import Git Repository**.
3. Select `qwen-image-2.1-runpod`, branch `main`, Dockerfile at the repository root. Endpoint type: **Queue**.
4. Pick a GPU with at least 24 GB VRAM (for example RTX 4090, L4, A5000, or A40). Weights are about 14 GB, and sampling needs more.
5. Attach a Network Volume mounted at `/runpod-volume` if you want the download to survive worker restarts. Without it, a new machine downloads the three files again.
6. Set the execution timeout to at least 1800 seconds. The first job waits while the files download.
7. Deploy. RunPod builds the image from GitHub. A new GitHub release updates an existing endpoint.

`SD_OFFLOAD=0` is the default, so all models load directly into GPU VRAM for maximum speed. Set `SD_OFFLOAD=1` if running on smaller GPUs (< 16 GB VRAM) where the text encoder needs to stay in system RAM.

## Request

```bash
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a ceramic teapot on a wooden table, soft daylight",
      "negative_prompt": "",
      "width": 1024,
      "height": 1024,
      "steps": 25,
      "cfg_scale": 6,
      "seed": -1
    }
  }'
```

`seed` below 0 is random. Width and height must be multiples of 32, from 256 through 2048. Steps must be from 1 through 60. The default sampler is Euler.

The result field `image` is a PNG encoded as base64. Pass `image` as base64 to use that picture as a reference edit. Long jobs should use `/run` and then poll `/status`, because `/runsync` can time out before a cold worker finishes.

## License

This worker code is MIT. The Qwen-Image 2.1 weights use the Qwen Research License and are for non-commercial use.
