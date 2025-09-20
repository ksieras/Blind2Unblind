package watchdog

import (
	"log"
	"os"
	"testing"
	"time"

	"github.com/go-redis/redis/v8"
)

func TestWatchdog_BasicFunctionality(t *testing.T) {
	// Create a test configuration
	config := DefaultConfig()
	config.ProcessName = "test-process"
	config.RefreshInterval = 2 * time.Second
	config.TimeoutDuration = 5 * time.Second
	config.MaxRetries = 2
	config.RetryInterval = 1 * time.Second

	// Create a test logger that discards output
	logger := log.New(os.Stdout, "[TEST] ", log.LstdFlags)

	// Create watchdog instance
	wd := New(config, logger)

	// Test initial state
	if wd.IsRunning() {
		t.Error("Watchdog should not be running initially")
	}

	// Start the watchdog
	if err := wd.Start(); err != nil {
		t.Skipf("Skipping test - Redis not available: %v", err)
	}

	// Verify it's running
	if !wd.IsRunning() {
		t.Error("Watchdog should be running after Start()")
	}

	// Wait for at least one refresh
	time.Sleep(3 * time.Second)

	// Check if the key exists in Redis
	client := redis.NewClient(&redis.Options{
		Addr: config.RedisAddr,
		DB:   config.RedisDB,
	})
	defer client.Close()

	key := wd.GetKey()
	val, err := client.Get(client.Context(), key).Result()
	if err != nil {
		t.Errorf("Failed to get key from Redis: %v", err)
	}

	if val == "" {
		t.Error("Key should exist in Redis")
	}

	t.Logf("Key value: %s", val)

	// Check TTL
	ttl, err := client.TTL(client.Context(), key).Result()
	if err != nil {
		t.Errorf("Failed to get TTL: %v", err)
	}

	if ttl <= 0 {
		t.Error("Key should have a positive TTL")
	}

	t.Logf("Key TTL: %v", ttl)

	// Stop the watchdog
	wd.Stop()

	// Verify it's stopped
	if wd.IsRunning() {
		t.Error("Watchdog should not be running after Stop()")
	}
}

func TestWatchdog_DoubleStart(t *testing.T) {
	config := DefaultConfig()
	config.ProcessName = "test-double-start"

	logger := log.New(os.Stdout, "[TEST] ", log.LstdFlags)
	wd := New(config, logger)

	// Start the watchdog
	if err := wd.Start(); err != nil {
		t.Skipf("Skipping test - Redis not available: %v", err)
	}
	defer wd.Stop()

	// Try to start again - should return error
	if err := wd.Start(); err == nil {
		t.Error("Starting an already running watchdog should return an error")
	}
}

func TestWatchdog_GetKey(t *testing.T) {
	config := DefaultConfig()
	config.ProcessName = "test-key-generation"

	wd := New(config, nil)

	expectedKey := "watchdog:test-key-generation"
	actualKey := wd.GetKey()

	if actualKey != expectedKey {
		t.Errorf("Expected key %s, got %s", expectedKey, actualKey)
	}
}

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.RedisAddr != "localhost:6379" {
		t.Errorf("Expected RedisAddr to be 'localhost:6379', got %s", config.RedisAddr)
	}

	if config.RefreshInterval != 30*time.Second {
		t.Errorf("Expected RefreshInterval to be 30s, got %v", config.RefreshInterval)
	}

	if config.TimeoutDuration != 60*time.Second {
		t.Errorf("Expected TimeoutDuration to be 60s, got %v", config.TimeoutDuration)
	}

	if config.MaxRetries != 5 {
		t.Errorf("Expected MaxRetries to be 5, got %d", config.MaxRetries)
	}
}