#!/bin/bash

# Install Ollama
curl https://ollama.ai/install.sh | sh

# Verify GPU
nvidia-smi

# Pull models you want
ollama pull llama3.3
ollama pull cogito:70b

#ollama run llama3.3
ollama run cogito:70b
systemctl --user start ollama