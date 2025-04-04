#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"


# Simple relative path to .env
source "$PROJECT_ROOT/.env"

start() {
    echo "🚀 Starting Lambda instance..."
    
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
    
    # Wait for SSH to be available
    echo "🔄 Waiting for SSH access..."
    while ! ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
              -i ~/.ssh/lambda_key ubuntu@$INSTANCE_IP exit 2>/dev/null
    do
        echo -n "."
        sleep 2
    done
    
    echo -e "\n🔌 SSH ready!"
    
    # Copy and run setup
    echo "📦 Copying setup script..."
    scp -i "$SSH_KEY_PATH" "$SCRIPT_DIR/setup_server.sh" ubuntu@$INSTANCE_IP:~/
    
    echo "🔧 Running setup script..."
    ssh -i "$SSH_KEY_PATH" ubuntu@$INSTANCE_IP './setup_server.sh'
    
    echo "🌟 Setup complete!"
    echo "🔌 Connecting with port forwarding..."
    
    # Note: -N means "don't execute remote command" (just forward ports)
    #       -f means "go to background"
    ssh -i "$SSH_KEY_PATH" -N -f -L 11434:localhost:11434 ubuntu@$INSTANCE_IP
    
    echo "✨ Ollama API available at localhost:11434"
    echo ""
    echo "To connect directly:"
    echo "ssh -i $SSH_KEY_PATH ubuntu@$INSTANCE_IP"
    echo ""
    echo "To stop the instance:"
    echo "./scripts/lambda_control.sh stop"
}

stop() {
    echo "🛑 Stopping Lambda instance..."
    INSTANCE_ID=$(curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" \
                      https://cloud.lambdalabs.com/api/v1/instances | \
                 jq -r '.data[0].id')
    
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
    *)
        echo "Usage: ./lambda_control.sh [start|stop|model]"
        ;;
esac
