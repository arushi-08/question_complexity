from urllib.request import urlopen
from bs4 import BeautifulSoup



questions = open("questions_final.csv")
questions.readline()

qSet = set()
for q in questions:
	qSet.add(q.split(",")[0])


count = 0
total = len(qSet)
for q in qSet:
	print(q,"\t",count+1," out of ",total,end="\t")
	count += 1
	num = q[:-1]
	letter = q[-1]
	try:
		qPage = urlopen("http://codeforces.com/problemset/problem/" + num +"/" + letter)	
	except:
		print("Error, skipping this question")
		continue

	soup = BeautifulSoup(qPage)
	

	try:
		problemStatement = soup.find_all('div',attrs={"class":"problem-statement"})[0]
		
		
		if(problemStatement.find_all('img')):
			print("Has images, skipping question")
			continue
		
		one = problemStatement.find_all('div',attrs={"class":"time-limit"})[0].get_text()
		two = problemStatement.find_all('div',attrs={"class":"memory-limit"})[0].get_text()
		three = problemStatement.find_all('div',attrs={"class":""})[0].get_text()
		four = problemStatement.find_all('div',attrs={"class":"input-specification"})[0].get_text()
		five = problemStatement.find_all('div',attrs={"class":"output-specification"})[0].get_text()

	except:
		print("Anomaly in soup structure, skipping this question")
		continue 
	f = open(q+".txt",'w')

	f.write("\n\n".join([one,two,three,four,five]))

	f.close()
	print()
	
questions.close()	
