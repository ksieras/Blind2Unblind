package watchdog

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
)

// Config holds the configuration for the Watchdog
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

// DefaultConfig returns a configuration with reasonable defaults
func DefaultConfig() *Config {
	return &Config{
		RedisAddr:       "localhost:6379",
		RedisPassword:   "",
		RedisDB:         0,
		ProcessName:     "watchdog-process",
		RefreshInterval: 30 * time.Second,
		TimeoutDuration: 60 * time.Second,
		MaxRetries:      5,
		RetryInterval:   5 * time.Second,
	}
}

// Watchdog represents a Redis-based process monitor
type Watchdog struct {
	config    *Config
	client    *redis.Client
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	isRunning bool
	mu        sync.RWMutex
	logger    *log.Logger
}

// New creates a new Watchdog instance with the given configuration
func New(config *Config, logger *log.Logger) *Watchdog {
	if config == nil {
		config = DefaultConfig()
	}
	if logger == nil {
		logger = log.Default()
	}

	ctx, cancel := context.WithCancel(context.Background())
	
	return &Watchdog{
		config: config,
		ctx:    ctx,
		cancel: cancel,
		logger: logger,
	}
}

// Start begins the watchdog monitoring process
func (w *Watchdog) Start() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.isRunning {
		return fmt.Errorf("watchdog is already running")
	}

	// Initialize Redis client
	w.client = redis.NewClient(&redis.Options{
		Addr:     w.config.RedisAddr,
		Password: w.config.RedisPassword,
		DB:       w.config.RedisDB,
	})

	// Test initial connection with retries
	if err := w.connectWithRetry(); err != nil {
		return fmt.Errorf("failed to connect to Redis after %d retries: %w", w.config.MaxRetries, err)
	}

	w.isRunning = true
	w.logger.Printf("Watchdog started for process '%s', refreshing every %v with timeout %v", 
		w.config.ProcessName, w.config.RefreshInterval, w.config.TimeoutDuration)

	// Start the refresh goroutine
	w.wg.Add(1)
	go w.refreshLoop()

	return nil
}

// Stop gracefully stops the watchdog
func (w *Watchdog) Stop() {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isRunning {
		return
	}

	w.logger.Printf("Stopping watchdog for process '%s'", w.config.ProcessName)
	
	// Cancel the context to stop the refresh loop
	w.cancel()
	
	// Wait for the refresh loop to finish
	w.wg.Wait()
	
	// Close Redis connection
	if w.client != nil {
		w.client.Close()
	}
	
	w.isRunning = false
	w.logger.Printf("Watchdog stopped for process '%s'", w.config.ProcessName)
}

// IsRunning returns whether the watchdog is currently running
func (w *Watchdog) IsRunning() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.isRunning
}

// GetKey returns the Redis key used for this process
func (w *Watchdog) GetKey() string {
	return fmt.Sprintf("watchdog:%s", w.config.ProcessName)
}

// connectWithRetry attempts to connect to Redis with retry logic
func (w *Watchdog) connectWithRetry() error {
	key := w.GetKey()
	
	for attempt := 0; attempt <= w.config.MaxRetries; attempt++ {
		// Test connection by trying to set a test key
		err := w.client.Set(w.ctx, key+":test", "connection_test", time.Second).Err()
		if err == nil {
			// Clean up test key
			w.client.Del(w.ctx, key+":test")
			w.logger.Printf("Successfully connected to Redis at %s", w.config.RedisAddr)
			return nil
		}

		if attempt < w.config.MaxRetries {
			w.logger.Printf("Failed to connect to Redis (attempt %d/%d): %v. Retrying in %v...", 
				attempt+1, w.config.MaxRetries+1, err, w.config.RetryInterval)
			time.Sleep(w.config.RetryInterval)
		} else {
			return fmt.Errorf("final connection attempt failed: %w", err)
		}
	}
	
	return fmt.Errorf("exceeded maximum retry attempts")
}

// refreshLoop is the main loop that periodically refreshes the Redis key
func (w *Watchdog) refreshLoop() {
	defer w.wg.Done()
	
	ticker := time.NewTicker(w.config.RefreshInterval)
	defer ticker.Stop()

	key := w.GetKey()
	
	// Initial refresh
	w.refreshKey(key)

	for {
		select {
		case <-w.ctx.Done():
			w.logger.Printf("Refresh loop stopped for process '%s'", w.config.ProcessName)
			return
		case <-ticker.C:
			w.refreshKey(key)
		}
	}
}

// refreshKey updates the Redis key with current timestamp and TTL
func (w *Watchdog) refreshKey(key string) {
	timestamp := time.Now().Unix()
	value := fmt.Sprintf("alive:%d", timestamp)
	
	err := w.client.Set(w.ctx, key, value, w.config.TimeoutDuration).Err()
	if err != nil {
		w.logger.Printf("ERROR: Failed to refresh key '%s': %v", key, err)
		
		// Try to reconnect if the error suggests connection issues
		if w.isConnectionError(err) {
			w.logger.Printf("Connection error detected, attempting to reconnect...")
			if reconnectErr := w.connectWithRetry(); reconnectErr != nil {
				w.logger.Printf("ERROR: Failed to reconnect to Redis: %v", reconnectErr)
			} else {
				w.logger.Printf("Successfully reconnected to Redis")
				// Retry the refresh after successful reconnection
				if retryErr := w.client.Set(w.ctx, key, value, w.config.TimeoutDuration).Err(); retryErr != nil {
					w.logger.Printf("ERROR: Failed to refresh key after reconnection: %v", retryErr)
				} else {
					w.logger.Printf("Successfully refreshed key '%s' after reconnection: %s", key, value)
				}
			}
		}
	} else {
		w.logger.Printf("Successfully refreshed key '%s': %s", key, value)
	}
}

// isConnectionError determines if an error is related to connection issues
func (w *Watchdog) isConnectionError(err error) bool {
	if err == nil {
		return false
	}
	
	// Common Redis connection error patterns
	errStr := err.Error()
	connectionErrors := []string{
		"connection refused",
		"connection reset",
		"network is unreachable",
		"no route to host",
		"timeout",
		"EOF",
		"broken pipe",
	}
	
	for _, connErr := range connectionErrors {
		if len(errStr) > 0 && (errStr == connErr || len(errStr) > len(connErr) && errStr[:len(connErr)] == connErr) {
			return true
		}
	}
	
	return false
}