#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"


# Simple relative path to .env
source "$PROJECT_ROOT/.env"

check_instance() {
    # Check for existing active instance
    INSTANCE_DATA=$(curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" \
                    https://cloud.lambdalabs.com/api/v1/instances)
    
    INSTANCE_ID=$(echo "$INSTANCE_DATA" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data['data'] else '')")
    INSTANCE_STATUS=$(echo "$INSTANCE_DATA" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['status'] if data['data'] else '')")
    INSTANCE_IP=$(echo "$INSTANCE_DATA" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['ip'] if data['data'] else '')")
    
    if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_STATUS" = "active" ]; then
        echo "✓ Found active instance: $INSTANCE_IP"
        echo "$INSTANCE_IP" > .instance_ip
        return 0
    else
        echo "No active instance found"
        return 1
    fi
}

cleanup_tunnel() {
    echo "🧹 Cleaning up port 11434..."
    
    # Check for Ollama app process
    if pgrep -f "Ollama.app" > /dev/null; then
        echo "Stopping Ollama app..."
        pkill -f "Ollama.app"
        sleep 1
    fi
    
    # Kill any Ollama processes
    if pgrep ollama > /dev/null; then
        echo "Killing Ollama processes..."
        pkill -9 ollama
        sleep 1
    fi
    
    # Force kill SSH tunnels
    local ssh_pids=$(lsof -ti:11434 -c ssh)
    if [ ! -z "$ssh_pids" ]; then
        echo "Force killing SSH tunnels..."
        kill -9 $ssh_pids
        sleep 1
    fi
    
    # Final verification
    if lsof -i:11434; then
        echo "❌ Failed to free port 11434"
        return 1
    else
        echo "✓ Port 11434 is now free"
        return 0
    fi
}

setup_tunnel() {
    local instance_ip=$1
    
    echo "🧹 Cleaning up port 11434..."
    # Kill any existing processes on port
    local port_pids=$(lsof -ti:11434)
    if [ ! -z "$port_pids" ]; then
        echo "Killing processes on port 11434..."
        kill -9 $port_pids
        sleep 2
    fi
    
    # Verify port is free
    if lsof -ti:11434; then
        echo "❌ Failed to free port 11434"
        return 1
    fi
    
    echo "🔌 Setting up SSH tunnel..."
    ssh -i "$SSH_KEY_PATH" -N -f -L 11434:localhost:11434 ubuntu@$instance_ip
    sleep 2
    
    echo "🔍 Verifying tunnel..."
    
    # Check if local Ollama is running
    if pgrep ollama > /dev/null; then
        echo "❌ Local Ollama is still running. Please stop it first:"
        echo "killall ollama"
        return 1
    fi
    
    # Check if we can reach Ollama API through tunnel
    if curl -s --connect-timeout 5 localhost:11434/api/tags > /dev/null; then
        # If we can reach API and local Ollama isn't running,
        # this must be the remote server
        echo "✓ Tunnel verified - connected to remote Ollama"
        return 0
    else
        echo "❌ Cannot reach Ollama API through tunnel"
        return 1
    fi
}

setup_ollama() {
    local instance_ip=$1
    local models=("llama3.3" "cogito:70b")
    
    # Install Ollama on remote server if needed
    echo "🔧 Installing Ollama on server..."
    ssh -i "$SSH_KEY_PATH" ubuntu@$instance_ip 'curl -fsSL https://ollama.com/install.sh | sh'
    
    if ! setup_tunnel "$instance_ip"; then
        echo "❌ Failed to establish tunnel to Lambda server"
        return 1
    fi
    
    echo "📦 Pulling models..."
    
    for model in "${models[@]}"; do
        echo "Pulling $model..."
        curl -s -X POST localhost:11434/api/pull -d "{\"name\": \"$model\"}" | \
            jq -r 'select(.status == "success") | "✓ Downloaded"' || echo "❌ Failed"
    done
    
    echo "✨ Setup complete! Ollama API available at localhost:11434"
}

start() {
    if check_instance; then
        echo "🔄 Using existing instance"
        INSTANCE_IP=$(cat .instance_ip)
    else
        echo "🚀 Starting new Lambda instance..."
        
        # Use exact name from Lambda list
        KEY_NAME="Lambda"
        echo "Debug: Using SSH key name: $KEY_NAME"
        
        # Get available regions with capacity
        echo "🔍 Checking for available GH200 instances..."
        REGIONS_DATA=$(curl -X GET "https://cloud.lambdalabs.com/api/v1/instance-types" \
            -H "Authorization: Bearer $LAMBDA_API_KEY" \
            -H 'accept: application/json')
        
        # Extract first region with capacity for gpu_1x_gh200
        REGION=$(echo "$REGIONS_DATA" | python3 -c '
import sys, json
data = json.load(sys.stdin)
regions = data["data"]["gpu_1x_gh200"]["regions_with_capacity_available"]
if regions:
    print(regions[0]["name"])
')

        if [ -z "$REGION" ]; then
            echo "❌ No GH200 capacity available in any region"
            exit 1
        fi
        
        echo "✨ Found capacity in region: $REGION"
        
        # Launch in the available region
        JSON="{\"region_name\": \"$REGION\", \"instance_type_name\": \"gpu_1x_gh200\", \"ssh_key_names\": [\"$KEY_NAME\"]}"
        echo "Debug: Sending JSON: $JSON"
        
        curl -H "Authorization: Bearer $LAMBDA_API_KEY" \
             -X POST \
             -d "$JSON" \
             https://cloud.lambdalabs.com/api/v1/instance-operations/launch
        
        # Use SSH_KEY_PATH from .env for SSH commands
        echo "🔌 Copying setup script..."
        scp -i "$SSH_KEY_PATH" setup_server.sh ubuntu@$INSTANCE_IP:~/
        
        echo "🔧 Running setup script..."
        ssh -i "$SSH_KEY_PATH" ubuntu@$INSTANCE_IP './setup_server.sh'
        
        echo "⏳ Waiting for instance to be ready..."
        
        # Poll until instance is ready
        while true; do
            STATUS=$(curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" \
                         https://cloud.lambdalabs.com/api/v1/instances | \
                         python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['status'])")
            
            if [ "$STATUS" = "active" ]; then
                break
            fi
            echo -n "."
            sleep 5
        done
        
        # Get IP once we know it's ready
        INSTANCE_IP=$(curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" \
                         https://cloud.lambdalabs.com/api/v1/instances | \
                         python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['ip'])")
        
        echo -e "\n✨ Instance ready at: $INSTANCE_IP"
        echo "$INSTANCE_IP" > .instance_ip
    fi
    nvidia-smi
    setup_ollama "$INSTANCE_IP"
    
    echo "✨ Setup complete! Ollama API available at localhost:11434"
    echo ""
    echo "To connect directly:"
    echo "ssh -i $SSH_KEY_PATH ubuntu@$INSTANCE_IP"
}

stop() {
    if ! check_instance; then
        echo "No active instance to stop"
        return
    fi
    
    cleanup_tunnel
    
    echo "🛑 Stopping Lambda instance..."
    curl -H "Authorization: Bearer $LAMBDA_API_KEY" \
         -X POST \
         -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}" \
         https://cloud.lambdalabs.com/api/v1/instance-operations/terminate
    
    echo "✅ Instance stopped"
}

model() {
    echo "Checking Ollama model..."
    ssh -i "$SSH_KEY_PATH" "$SSH_USER@$INSTANCE_IP" "curl -s http://localhost:11434/api/generate -d '{\"model\":\"llama3\",\"prompt\":\"What model are you?\",\"stream\":false}' | jq -r '.model + \": \" + .response'"
    echo "Model check complete."
}

case "$1" in
    "start")
        start
        ;;
    "stop")
        stop
        ;;
    "model")
        model
        ;;
    "setup")
        if ! check_instance; then
            echo "No active instance found. Run 'start' first"
            exit 1
        fi
        INSTANCE_IP=$(cat .instance_ip)
        setup_ollama "$INSTANCE_IP"
        ;;
    *)
        echo "Usage: ./lambda_control.sh [start|stop|model|setup]"
        ;;
esac
