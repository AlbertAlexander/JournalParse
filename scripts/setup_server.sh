#!/bin/bash

# Install Ollama
curl https://ollama.ai/install.sh | sh

# Verify GPU
nvidia-smi

# Pull models you want
ollama pull mistral
ollama pull llama3.3
ollama pull qwen:7b

ollama run llama3.3
systemctl --user start ollama