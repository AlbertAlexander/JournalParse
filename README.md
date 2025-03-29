This project is a multi-layer system for anonymizing and analyzing journal entries.

Guide:
1. Spin up a server (instructions below for Lambda, but any cloud server will do). Install ollama, pull your preferred models, and start the ollama server. No GPU is needed, but it will make processing 5-10x faster. Note that larger models aren't necessarily better, particularly for the anonymization task.
2. Join all journal entries into a single text file and run privacy_analyzer-ai.py. Inspect the output to make sure replacement wasn't overaggressive. 
3. Use the output of privacy_analyzer-ai.py as the input for analyzer.py, visualize_emotions.py, and whatever other uses you may have.

Note that the analyzer assumes a modest context window (2000 tokens) and splits the text accordingly, processing each chunk independently for whatever query you run. You can expand the context window to your model's limits if you like.

TBA: 
Multi-layer interpretation (summaries of summaries)
Fully quantitative structured output for analysis and visualization


## Server Setup
1. Copy 'scripts/.env.example' to '.env'
2. Add your Lambda API key to .env
3. Update SSH key path in .env
4. Make scripts executable:
  ```bash
   chmod +x scripts/lambda_control.sh scripts/setup_server.sh
   ```

## Server Usage
Start server:
bash
./scripts/lambda_control.sh start


Stop server:
bash
./scripts/lambda_control.sh stop
