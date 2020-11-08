# Smart Queuing System  

[![](http://img.youtube.com/vi/a_MuDrHDavA/0.jpg)](http://www.youtube.com/watch?v=a_MuDrHDavA "Smart Queuing System")

## Objectives  
1. Propose a possible hardware solution  
2. Build out the application and test its performance on the DevCloud using multiple hardware types.  
3. Compare the performance to see which hardware performed best.  
4. Revise the proposal based on the test results.  

## Part 1: Hardware Proposal  
We've provided three different scenarios that depict real-world problems based on different sectors where edge devices are typically deployed.  

The three scenarios you'll be looking at are:  

Scenario 1: [Manufacturing Sector](./Manufacturing_Scenario.md)
Scenario 2: [Retail Sector](./Retail_Scenario.md)  
Scenario 3: [Transportation Sector](./Transportation_Scenario.md)

All of the scenarios involve people in queues, but each scenario will require different hardware. So your first task will be to determine which hardware might work for each scenario—and then explain your initial choice in a proposal document.  

## Part 2: Testing Your Hardware  
The following files contain the code to build out the smart queuing application and test its performance on all four different hardware types (CPU, IGPU, VPU, and FPGA) using the DevCloud.

### Files  
1. `Create_Python_Script.ipynb`: Generate a python script `person_detect.py` that contain Person Detection Model.
2. `Create_Job_Submission_Script.ipynb`: Generate a shell script `queue_job.sh` that will submit jobs to Intel's DevCloud to load and run inference on each type of hardware.  
3. `Manufacturing_Scenario.ipynb`: The smart queue monitoring system for manufacturing sector.  
4. `Retail_Scenario.ipynb`: The smart queue monitoring system for retail sector.  
5. `Transportation_Scenario.ipynb`: The smart queue monitoring system for transportation sector.  
6. `Choose the Right Hardware Proposal.pdf`: The final hardware proposal for three scenarios based on the performance across different edge devices.  
