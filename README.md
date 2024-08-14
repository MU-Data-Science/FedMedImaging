# FedMedImaging

## Environment Setup
### Nodes Configuration:

*	The experiment environment consists of 10 machines, referred to as nodes.
*	One node is designated as the server.
*	The remaining nine nodes are designated as clients.
  
### Data Distribution:

*	Prepare the dataset to be used for the experiment.
*	Distribute the dataset across the client nodes. Each client should receive its own subset of data, which will be used for local training.
*	Verify that the data is correctly loaded and accessible.

### Network Configuration:

*	Ensure that all nodes (both the server and clients) are connected to the same network.
*	Configure the network to allow communication between all nodes. This may involve setting up IP addresses, firewalls, and ensuring that the necessary ports are open for communication.

### Software Installation:

*	Ensure that all nodes have the same version of the required libraries and dependencies.
*	Make sure that the libraries used across all nodes are compatible and do not clash.

### Experiment Execution

*	Set the federated learning strategy, number of epochs, and other relevant parameters manually in the server and client code. These settings will not be dynamically adjusted during the experiment and must be set correctly before starting.
*	Strategy templates are on Strategy.txt in the repository. It can be directly changed with the strategy in the server.py
*	Start the server process on the designated server node.
*	On each client node, start the client process.



## INSTALLATION 
-----------------------------------
FOR EVERY NODE
-----------------------------------
Codes are written between triple quotes (""" ... """) don't copy the quotes
Run the commands below from terminal

	"""	sudo apt update && sudo apt upgrade -y	"""
	"""	sudo apt install python3-pip -y		"""
	"""	pip3 install --upgrade pip		"""
	"""	pip3 install poetry			"""

At this point you may need to reset the node or update the root directory according to promt given to register the poetry. If no warnings are given proceed.

Clone this repository

 	"""	git clone https://github.com/MU-Data-Science/FedMedImaging.git	"""

Run the commands below from terminal

 	"""	cd FedMedImaging		"""
	"""	poetry shell			"""
	"""	pip install -r requirements.txt	"""

 	"""	pip3 install pillow scipy	"""
  	"""	poetry install			"""
 
certificate.conf: change IP.2 adress with local server ip adress for all client.py and server.py

Change the server_address="10.10.1.1:5000",with your server ip adress in client.py and server.py

You can find your servers ip adress with the command ifconfig run from your server terminal

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
Dataset should be disturbuted into 10 nodes in default. One of the nodes should have all test set and should be treated as the main testing node


-----------------------------------
STARTING THE EXPERIMENT
-----------------------------------
- Optimization strategy, local epochs, batch size and num of rounds are defined in the server.py
- Strategies hyperparameters may need to be adjusted according to model and dataset used.
- Dataset adress is defined in the client.py

- For server : server.py
		
		"""	poetry run python3 server.py -i 'model name' -cl 'number of classes'	"""
  *	for example:
  
    	""" poetry run python3 server.py -i EfficientNetB0 -cl 4 """
    
- For client : client.py

		"""	poetry run python3 client.py -i 'model name' -cl 'number of classes'	"""
   *	for example:   
  	
    		""" poetry run python3 client.py -i EfficientNetB0 -cl 4 """





