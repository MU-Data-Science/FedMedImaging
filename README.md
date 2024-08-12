# FedMedImaging


## INSTALLATION 
-----------------------------------
FOR EVERY NODE
-----------------------------------
	sudo apt update && sudo apt upgrade -y
	sudo apt install python3-pip -y
	pip3 install --upgrade pip
	pip3 install poetry
You need to disconnect and then connect again or adjust the root directory shown in the promt after this step. Otherwise, poetry cant be used

Clone this repository

 	poetry install
	poetry shell
	pip3 install pillow scipy
	certificate.conf: change IP.2 adress with local server adress

-----------------------------------
DATASET STRUCTURE
-----------------------------------
dataset
-  	test
	-		class1
	-		class2
-	train
	-		class1
	-		class2

Images shoud be reduced to 256x256 for default. In case of using different size, client.py should be changed accordingly
Dataset should be disturbuted into 10 nodes. 1 of the nodes should have all test set and should be treated as the main testing node


-----------------------------------
STARTING THE EXPERIMENT
-----------------------------------
- For server : models/server.py
		
		poetry run python3 server.py -i 'model name' -cl 'number of classes'
- For client : models/client.py

		poetry run python3 client.py -i 'model name' -cl 'number of classes'
- Optimization strategy, local epochs, batch size and num of rounds are defined in the server.py
- Dataset adress is defined in the client.py

-----------------------------------
USEFUL TOOLS
-----------------------------------
# SCREEN

Create screen
	
	screen -S 
List screens	

	screen -ls 
Open running screen

	screen -r "screen name" 
Create screen with logfile
 
	screen -L -Logfile log_filename 
Stop all screens	
	
	pkill screen  


