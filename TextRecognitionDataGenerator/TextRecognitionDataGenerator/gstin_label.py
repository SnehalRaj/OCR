import random
def string():
	d = "0123456789"
	a = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	ai = random.choice("0123")+random.choice(d)+random.choice(a)+random.choice(a)+random.choice(a)+random.choice(a)+random.choice(a)+random.choice(d)+random.choice(d)+random.choice(d)+random.choice(d)+random.choice(a)+random.choice(a+d)+random.choice("Z")+random.choice(a+d)
	return ai
for i in range(1000000):
	print(string())
