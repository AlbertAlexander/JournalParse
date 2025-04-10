#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List
import textwrap
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
import math

# Load environment variables from .env file
load_dotenv()

# Initialize console for rich text output
console = Console()

# Testing constants (temporary)
CHARS_PER_TOKEN = 4
MAX_API_TOKENS = 8000     # Reduced from 800000
MAX_CHUNK_TOKENS = 5000   # Reduced from 500000
OVERLAP_TOKENS = 500      # Reduced from 8000

def create_client():
    """Create and return an OpenAI client configured for Lambda's API."""
    api_key = os.getenv("LAMBDA_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] LAMBDA_API_KEY not found in .env file")
        sys.exit(1)
        
    return OpenAI(
        api_key=api_key,
        base_url="https://api.lambda.ai/v1"
    )

def list_available_models():
    """List all available models from Lambda API."""
    client = create_client()
    models = client.models.list()
    
    console.print("\n[bold cyan]Available Models:[/bold cyan]")
    for model in models.data:
        console.print(f"- {model.id}")
    console.print()

def read_file_content(file_path: str) -> str:
    """Read content from a file and return it as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        console.print(f"[bold red]Error reading file:[/bold red] {e}")
        sys.exit(1)

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text based on character count."""
    return len(text) // CHARS_PER_TOKEN

def chunk_large_file(file_path: str) -> List[str]:
    """
    Chunk a large file into manageable pieces with overlap.
    Returns a list of chunks that can be processed separately.
    """
    content = read_file_content(file_path)
    tokens = estimate_tokens(content)
    
    # If file is small enough, return it as is
    if tokens <= MAX_CHUNK_TOKENS:
        return [content]
    
    # Calculate chunk sizes in characters
    max_chunk_chars = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
    overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN
    
    # Split content into paragraphs to avoid breaking mid-paragraph
    paragraphs = content.split('\n\n')
    
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para) + 2  # +2 for the newlines
        
        # If adding this paragraph would exceed chunk size and we already have content
        if current_size + para_size > max_chunk_chars and current_size > 0:
            # Add current chunk to results
            chunks.append(current_chunk)
            
            # Calculate overlap - take last N characters
            overlap_text = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
            
            # Start new chunk with overlap
            current_chunk = overlap_text + "\n\n" + para
            current_size = len(current_chunk)
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n"
                current_size += 2
            current_chunk += para
            current_size += len(para)
    
    # Add the final chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # Log chunking info
    console.print(f"[bold yellow]Large file detected:[/bold yellow] {file_path}")
    console.print(f"[yellow]File split into {len(chunks)} chunks with ~{OVERLAP_TOKENS} tokens overlap[/yellow]")
    
    return chunks

def chat_with_context(
    question: str, 
    context_files: List[str], 
    model: str = "llama-4-scout-17b-16e-instruct",
    system_prompt: Optional[str] = None
):
    """Send a chat completion request with file contexts and display the response."""
    client = create_client()
    
    # Default system prompt if none provided
    if not system_prompt:
        system_prompt = (
            "You are a helpful assistant who receives context from files and answers questions "
            "based on that context. If the answer isn't in the context, say so rather than making "
            "things up."
        )
    
    # Prepare messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Process and chunk files
    file_chunks = []
    for file_path in context_files:
        chunks = chunk_large_file(file_path)
        filename = Path(file_path).name
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            file_chunks.append({
                "filename": filename,
                "chunk_num": i + 1,
                "total_chunks": len(chunks),
                "content": chunk
            })
    
    # Calculate total estimated tokens for all chunks
    total_estimated_tokens = sum(estimate_tokens(chunk["content"]) for chunk in file_chunks)
    
    # Warn if we're still over the limit after chunking
    if total_estimated_tokens > MAX_API_TOKENS:
        console.print(f"[bold red]Warning:[/bold red] Total content may exceed API limits ({total_estimated_tokens} estimated tokens)")
        console.print("[yellow]Processing chunks separately and combining results...[/yellow]")
        
        # Process in batches
        process_large_context_in_batches(client, file_chunks, question, model, system_prompt)
        return
    
    # Add all chunks to messages
    for chunk in file_chunks:
        chunk_indicator = ""
        if chunk["total_chunks"] > 1:
            chunk_indicator = f" (Chunk {chunk['chunk_num']}/{chunk['total_chunks']})"
            
        messages.append({
            "role": "user", 
            "content": f"Here's the content from file '{chunk['filename']}'{chunk_indicator}:\n\n{chunk['content']}\n\nPlease acknowledge receipt of this context."
        })
        messages.append({
            "role": "assistant",
            "content": f"I've received the content from '{chunk['filename']}'{chunk_indicator} and will use it to answer your questions."
        })
    
    # Add the user's question
    messages.append({"role": "user", "content": question})
    
    try:
        # Show a spinner while waiting for response
        with console.status(f"[bold green]Sending request to {model}... (est. {total_estimated_tokens} tokens)", spinner="dots"):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        
        # Display the response
        answer = response.choices[0].message.content
        console.print(Panel(Markdown(answer), border_style="cyan", title="Response", expand=False))
        
        # Display token usage
        console.print(f"\n[dim]Token usage: {response.usage.prompt_tokens} prompt + "
                     f"{response.usage.completion_tokens} completion = "
                     f"{response.usage.total_tokens} total tokens[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

def process_large_context_in_batches(client, file_chunks, question, model, system_prompt):
    """Process extremely large context by splitting into multiple API calls and combining results."""
    # Calculate how many chunks we can fit in each batch
    tokens_per_chunk = [estimate_tokens(chunk["content"]) for chunk in file_chunks]
    
    batches = []
    current_batch = []
    current_batch_tokens = 0
    
    # Create batches of chunks that fit within token limits
    for i, chunk in enumerate(file_chunks):
        chunk_tokens = tokens_per_chunk[i]
        
        # If adding this chunk would exceed limits, start a new batch
        if current_batch_tokens + chunk_tokens > MAX_API_TOKENS and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_batch_tokens = 0
        
        current_batch.append(chunk)
        current_batch_tokens += chunk_tokens
    
    # Add the final batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    console.print(f"[yellow]Split into {len(batches)} API requests[/yellow]")
    
    # Process each batch with appropriate context
    all_responses = []
    for i, batch in enumerate(batches):
        console.print(f"[cyan]Processing batch {i+1}/{len(batches)}...[/cyan]")
        
        # Create messages for this batch
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add batch context
        for chunk in batch:
            chunk_indicator = ""
            if chunk["total_chunks"] > 1:
                chunk_indicator = f" (Chunk {chunk['chunk_num']}/{chunk['total_chunks']})"
                
            messages.append({
                "role": "user", 
                "content": f"Here's the content from file '{chunk['filename']}'{chunk_indicator}:\n\n{chunk['content']}"
            })
            messages.append({
                "role": "assistant",
                "content": f"I've received the content."
            })
        
        # Add the user's question with batch context
        batch_question = question
        if len(batches) > 1:
            batch_question = f"This is part {i+1} of {len(batches)} of the context. {question}"
            
        messages.append({"role": "user", "content": batch_question})
        
        try:
            # Call the API
            with console.status(f"[bold green]Processing batch {i+1}/{len(batches)}...", spinner="dots"):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            
            # Store the response
            answer = response.choices[0].message.content
            all_responses.append(answer)
            
            # Display token usage for this batch
            console.print(f"[dim]Batch {i+1} tokens: {response.usage.total_tokens}[/dim]")
            
        except Exception as e:
            console.print(f"[bold red]Error processing batch {i+1}:[/bold red] {e}")
            all_responses.append(f"Error processing this section: {e}")
    
    # Combine and display all responses
    if len(all_responses) > 1:
        combined_response = "\n\n## Combined Analysis from Multiple Batches\n\n"
        for i, resp in enumerate(all_responses):
            combined_response += f"\n### Batch {i+1} Analysis\n\n{resp}\n\n"
        
        console.print(Panel(Markdown(combined_response), border_style="cyan", title="Combined Response", expand=False))
    else:
        console.print(Panel(Markdown(all_responses[0]), border_style="cyan", title="Response", expand=False))

def main():
    parser = argparse.ArgumentParser(description="CLI for Lambda's LLM API with file context support")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with the model using file context")
    chat_parser.add_argument("-f", "--files", nargs="+", help="Path to context file(s)", required=True)
    chat_parser.add_argument("-q", "--question", help="Question to ask")
    chat_parser.add_argument("-m", "--model", default="llama-4-scout-17b-16e-instruct", 
                           help="Model to use (default: llama-4-scout-17b-16e-instruct)")
    chat_parser.add_argument("-s", "--system", help="Custom system prompt")
    
    args = parser.parse_args()
    
    if args.command == "list-models":
        list_available_models()
    elif args.command == "chat":
        question = args.question
        if not question:
            # If no question provided, prompt interactively
            question = console.input("[bold cyan]Enter your question:[/bold cyan] ")
        
        chat_with_context(
            question=question,
            context_files=args.files,
            model=args.model,
            system_prompt=args.system
        )
    else:
        # Default to showing help if no command specified
        parser.print_help()

if __name__ == "__main__":
    main()
