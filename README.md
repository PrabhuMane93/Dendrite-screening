# Dendrite-screening
Project for screening assessment for Dendrite.ai Data Scientist Internship

### How do I get set up? ###
* Following is written for python3.

* Setup Environment
```
curent working directory : Dendrite-screening
```

* Dependencies Set Up
> All the dependencies are in requirements.txt

* Install dependencies
```
pip3 install -r requirements.txt
```

* Deployment Instructions
> In the Dendrite-screening directory, start the project through main.py
```
python3 main.py <path to json file> <Path to csv file>
```
> In our case, that would be iris.csv and algoparams_from_ui.json in assets directory. Feel free to swap out any other dataset to experiment.
```
python3 main.py assets/algoparams_from_ui.json assets/iris.csv
```

* The code only works for usecases under regression.

* Python notebook screening_test.ipynb also contains the project in non-object oriented fashion. Please setup environment before launching the notebook.

* Launching Python Notebook
> In the Dendrite-screening directory, run the following command
```
python3 -m notebook
```
> This will launch a jupyter notebook server on localhost. Open screening_test.ipynb

* Output
> The output should look something like this:
<img width="902" alt="output" src="https://github.com/PrabhuMane93/Dendrite-screening/assets/86148520/4847f9bf-c160-423e-8859-c4e949b4fff9">
