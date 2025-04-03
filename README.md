This project is a multi-layer system for anonymizing and analyzing journal entries.

## Server Setup
lambda_control.sh will automatically spin up a Lambda GH200 server VM, install ollama, pull models, start the ollama server, and configure port forwarding at localhost:11434. Other cloud services are fine but Lambda does not require a billing history or quota request for instant GPU access.No GPU is needed, but it will make processing 5-10x faster. Note that larger models aren't necessarily better, particularly for the anonymization task.

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

## Anonymization
You will need a single text file as input for privacy_analyzer-ai.py, which will chop up the text, batch LLM prompting, and parse and join JSON outputs in the pseudonymized_output directory. 

Inspect the output to make sure replacement wasn't overaggressive. Find and replace anonymized tokens as needed. You may need to restore to their original values since AI is sometimes unreliable, but I found this to be much easier than working with transformers.

Overwriting via mulitiple passes isn't tested, so you may want to move outputs back to the root directory to reprocess. 

Usage:
python privacy_analyzer-ai.py --input_file <path_to_input_file> --mode remote --model <model_name>



## Analysis
Use the output of privacy_analyzer-ai.py as the input for analyzer.py, visualize_emotions.py, and whatever other uses you may have.

Note that the analyzer assumes a modest context window (2000 tokens) and splits the text accordingly, processing each chunk independently for whatever query you run. You can expand the context window to your model's limits if you like.

TBA: 
Multi-layer interpretation (summaries of summaries)
Fully quantitative structured output for analysis and visualization

