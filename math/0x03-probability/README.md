# README.md 
# 0x03. Probability
## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General

* What is probability?
* Basic probability notation
* What is independence? What is disjoint?
* What is a union? intersection?
* What are the general addition and multiplication rules?
* What is a probability distribution?
* What is a probability distribution function? probability mass function?
* What is a cumulative distribution function?
* What is a percentile?
* What is mean, standard deviation, and variance?
* Common probability distributions

## Tasks
* 0. Initialize Poisson
Create a class Poisson that represents a poisson distribution:
	•	Class contructor def __init__(self, data=None, lambtha=1.):
	◦	data is a list of the data to be used to estimate the distribution
	◦	lambtha is the expected number of occurences in a given time frame
	◦	Sets the instance attribute lambtha
	▪	Saves lambtha as a float
	◦	If data is not given, (i.e. None (be careful: not data has not the same result as data is None)):
	▪	Use the given lambtha
	▪	If lambtha is not a positive value or equals to 0, raise a ValueError with the message lambtha must be a positive value
	◦	If data is given:
	▪	Calculate the lambtha of data
	▪	If data is not a list, raise a TypeError with the message data must be a list
	▪	If data does not contain at least two data points, raise a ValueError with the message data must contain multiple values
