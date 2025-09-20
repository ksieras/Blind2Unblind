package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ksieras/Blind2Unblind/watchdog"
)

func main() {
	// Create a logger with timestamp and process info
	logger := log.New(os.Stdout, "[WATCHDOG] ", log.LstdFlags|log.Lshortfile)
	
	// Create configuration
	config := watchdog.DefaultConfig()
	config.ProcessName = "demo-process"
	config.RefreshInterval = 10 * time.Second  // Refresh every 10 seconds for demo
	config.TimeoutDuration = 20 * time.Second  // Key expires after 20 seconds
	config.MaxRetries = 3
	config.RetryInterval = 2 * time.Second

	logger.Printf("Starting Watchdog demonstration...")
	logger.Printf("Configuration:")
	logger.Printf("  Process Name: %s", config.ProcessName)
	logger.Printf("  Redis Address: %s", config.RedisAddr)
	logger.Printf("  Refresh Interval: %v", config.RefreshInterval)
	logger.Printf("  Timeout Duration: %v", config.TimeoutDuration)
	logger.Printf("  Max Retries: %d", config.MaxRetries)
	logger.Printf("  Retry Interval: %v", config.RetryInterval)

	// Create watchdog instance
	wd := watchdog.New(config, logger)

	// Set up signal handling for graceful shutdown
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	// Start the watchdog
	if err := wd.Start(); err != nil {
		logger.Fatalf("Failed to start watchdog: %v", err)
	}

	logger.Printf("Watchdog is running. Redis key: %s", wd.GetKey())
	logger.Printf("Press Ctrl+C to stop the watchdog gracefully...")

	// Simulate some work while the watchdog runs
	go func() {
		for i := 1; wd.IsRunning(); i++ {
			logger.Printf("Demo process working... iteration %d", i)
			time.Sleep(5 * time.Second)
		}
	}()

	// Wait for interrupt signal
	<-signalChan
	logger.Printf("Received shutdown signal...")

	// Gracefully stop the watchdog
	wd.Stop()
	
	logger.Printf("Watchdog demonstration completed.")
}