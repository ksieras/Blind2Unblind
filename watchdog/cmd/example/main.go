package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ksieras/Blind2Unblind/watchdog"
)

// Example demonstrates how to integrate the watchdog into an application
func main() {
	// Set up logging
	logger := log.New(os.Stdout, "[APP] ", log.LstdFlags)
	
	// Configure the watchdog
	config := watchdog.DefaultConfig()
	config.ProcessName = "example-app"
	config.RefreshInterval = 15 * time.Second
	config.TimeoutDuration = 30 * time.Second
	
	// Create and start the watchdog
	wd := watchdog.New(config, logger)
	
	if err := wd.Start(); err != nil {
		logger.Fatalf("Failed to start watchdog: %v", err)
	}
	defer wd.Stop()
	
	logger.Printf("Application started with watchdog monitoring")
	logger.Printf("Redis key: %s", wd.GetKey())
	
	// Set up graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	
	// Simulate application work
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	
	counter := 0
	for {
		select {
		case <-quit:
			logger.Printf("Shutting down application...")
			return
		case <-ticker.C:
			counter++
			logger.Printf("Application working... task %d completed", counter)
			
			// Simulate different workloads
			if counter%10 == 0 {
				logger.Printf("Performing heavy computation...")
				time.Sleep(2 * time.Second)
			}
		}
	}
}