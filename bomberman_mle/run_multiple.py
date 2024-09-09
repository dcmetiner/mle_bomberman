import os

# Number of times you want to run the code
num_runs = 400

for i in range(num_runs):
    print(f"Running iteration {i+1}")
    os.system("python main.py play --agents my_agent --train 1 --scenario coin-heaven --no-gui")