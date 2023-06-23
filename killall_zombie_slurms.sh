#!/bin/bash

# Get the list of jobs with dependency-failed state
jobs=$(squeue -h -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" | grep "PD" | grep "DependencyNeverSatisfied" | awk '{print $1}')

# Iterate through the jobs
for job_id in $jobs; do
  # Get the job's reason
  reason=$(squeue -h -j $job_id -o "%R")

  # Check if the reason is not "dependency"
  if [ "$reason" != "Dependency" ]; then
    # Cancel the job
    scancel $job_id
    echo "Cancelled job: $job_id"
  fi
done