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

Verify:

Restore ssh connection:
ssh -i <path_to_key> ubuntu@<ip_address>
ssh -i ~/.ssh/lambda_key -L 11434:localhost:11434 ubuntu@192.222.51.174

Stop server:
bash
./scripts/lambda_control.sh stop

## Anonymization
You will need a single text file as input for privacy_analyzer-ai.py, which will chop up the text, batch LLM prompting, and parse and join JSON outputs in the pseudonymized_output directory. 

Inspect the output to make sure replacement wasn't overaggressive. Find and replace anonymized tokens as needed. You may need to restore to their original values since AI is sometimes unreliable, but I found this to be much easier than working with transformers.

### Parameters
python privacyanalyzer-ai.py
- `file`: Path to the text file to process
- `--mode`: Use 'local' or 'remote' Ollama (default: local)
- `--host`: Ollama host (default: localhost)
- `--port`: Ollama port (default: 11434)
- `--model`: Ollama model to use (default: llama3.3:latest)
- `--resume-from`: Resume processing from a specific chunk index (0-based)
- `--output-dir`: Specify an existing output directory for resumption

## Resume Mode
There is some defensive parsing to handle unexpected LLM outputs, but if things break, you can use resume mode to continue appending from the last chunk. When resuming processing:

1. The tool loads progress from the specified output directory
2. It skips chunks that were already processed
3. New results are appended to existing output files
4. Processing continues from the specified chunk index

Note: for resume mode to function, you must specify an output directory with the following files:
- Progress file (file_stem_progress.json)
- Pseudonymized file (file_stem_pseudonymized.txt)
- Mapping file (file_stem_mapping.json)

## Output Files

The tool generates several output files:

- `*_pseudonymized.txt`: The pseudonymized text. This doesn't work well, and you will likely get better results from defining manual substitutions. You will likely also find false positives in substitutions.json that you want to delete. TBA: find and replace from substitutions.json
- `*_mapping.json`: Mapping between original terms and pseudonyms
- `*_progress.json`: Progress tracking for resumption
- `error_logs/`: Directory containing error logs for failed chunks

## Analysis
Use the output of privacy_analyzer-ai.py as the input for analyzer.py, visualize_emotions.py, and whatever other uses you may have.

Note that the analyzer assumes a modest context window (2000 tokens) and splits the text accordingly, processing each chunk independently for whatever query you run. You can expand the context window to your model's limits if you like.

TODO:
Test privacyanalyzer-ai.py resume mode
Multi-layer interpretation (summaries of summaries)
Fully quantitative structured output for analysis and visualization

