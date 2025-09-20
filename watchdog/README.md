# Go Redis Watchdog

A Redis-based process monitoring watchdog implementation in Go that periodically refreshes a key in Redis to indicate that a process is alive.

## Features

- **Redis Integration**: Uses Redis to store process heartbeat information
- **Configurable Intervals**: Customizable refresh interval and timeout duration
- **Error Handling**: Robust error handling with automatic retry logic for Redis connectivity issues
- **Graceful Shutdown**: Clean start/stop mechanisms with proper resource cleanup
- **Comprehensive Logging**: Detailed logging of status messages, errors, and connection issues
- **Recovery Mechanism**: Automatic reconnection attempts when Redis connection is lost

## Installation

1. Ensure Redis is installed and running:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   sudo systemctl start redis-server
   
   # Or using Docker
   docker run -d -p 6379:6379 redis:alpine
   ```

2. Initialize the Go module:
   ```bash
   cd watchdog
   go mod tidy
   ```

3. Build the demo application:
   ```bash
   go build -o bin/watchdog-demo ./cmd/demo
   ```

## Usage

### Basic Usage

```go
package main

import (
    "log"
    "time"
    "github.com/ksieras/Blind2Unblind/watchdog"
)

func main() {
    // Create configuration
    config := watchdog.DefaultConfig()
    config.ProcessName = "my-process"
    config.RefreshInterval = 30 * time.Second
    config.TimeoutDuration = 60 * time.Second
    
    // Create logger
    logger := log.New(os.Stdout, "[WATCHDOG] ", log.LstdFlags)
    
    // Create and start watchdog
    wd := watchdog.New(config, logger)
    
    if err := wd.Start(); err != nil {
        log.Fatalf("Failed to start watchdog: %v", err)
    }
    defer wd.Stop()
    
    // Your application logic here...
    
    log.Println("Application running...")
    time.Sleep(5 * time.Minute)
}
```

### Running the Demo

```bash
# Run the demonstration program
./bin/watchdog-demo
```

The demo will:
1. Start the watchdog with a 10-second refresh interval
2. Show periodic status updates and Redis key refreshes
3. Run until interrupted with Ctrl+C
4. Demonstrate graceful shutdown

### Configuration Options

```go
type Config struct {
    // Redis connection settings
    RedisAddr     string        // Redis server address (default: "localhost:6379")
    RedisPassword string        // Redis password (default: "")
    RedisDB       int           // Redis database number (default: 0)
    
    // Watchdog settings
    ProcessName     string        // Name of the process to monitor
    RefreshInterval time.Duration // How often to refresh the key (default: 30s)
    TimeoutDuration time.Duration // How long the key should live (default: 60s)
    
    // Retry settings
    MaxRetries    int           // Maximum number of connection retries (default: 5)
    RetryInterval time.Duration // Time between retry attempts (default: 5s)
}
```

## How It Works

1. **Key Generation**: Creates a unique Redis key for each process: `watchdog:<process-name>`
2. **Periodic Refresh**: Updates the key with a timestamp value every `RefreshInterval`
3. **TTL Management**: Sets a TTL (Time To Live) of `TimeoutDuration` on each key update
4. **Health Monitoring**: If a process stops refreshing, the key expires from Redis
5. **Error Recovery**: Automatically attempts to reconnect if Redis becomes unavailable

## Monitoring

You can monitor the watchdog status using Redis CLI:

```bash
# Check if a process is alive
redis-cli get "watchdog:my-process"

# Monitor key expiration
redis-cli ttl "watchdog:my-process"

# Watch for key updates in real-time
redis-cli monitor
```

## Error Handling

The watchdog includes comprehensive error handling:

- **Connection Failures**: Automatic retry with exponential backoff
- **Network Issues**: Detection and recovery from connection drops
- **Redis Unavailability**: Graceful degradation with continued retry attempts
- **Configuration Errors**: Validation and helpful error messages

## Testing Error Scenarios

1. **Redis Down During Startup**:
   ```bash
   sudo systemctl stop redis-server
   ./bin/watchdog-demo  # Will show retry attempts and failure
   ```

2. **Redis Down During Operation**:
   ```bash
   # Start the demo, then in another terminal:
   sudo systemctl stop redis-server
   # Watch the logs for reconnection attempts
   ```

3. **Recovery Testing**:
   ```bash
   # While watchdog is running and Redis is down:
   sudo systemctl start redis-server
   # Watch for successful reconnection
   ```

## Integration

To integrate the watchdog into your application:

1. Import the package
2. Create a configuration with your process name
3. Start the watchdog before your main application logic
4. Stop the watchdog during graceful shutdown

Example integration patterns:
- **Microservices**: Monitor service health across distributed systems
- **Background Jobs**: Ensure long-running tasks are alive
- **Data Processing**: Monitor ETL pipelines and data processors
- **Health Checks**: Provide external monitoring capabilities

## Dependencies

- `github.com/go-redis/redis/v8`: Redis client for Go
- Go 1.21 or later

## License

This implementation is part of the Blind2Unblind project.