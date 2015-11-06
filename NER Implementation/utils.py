# Script with all feature functions for MyMaxEnt class
'''
Tag Set:
ORGANIZATION, PERSON, LOCATION, DATE, TIME, MONEY, PERCENT, FACILITY, GPE, OTHER
'''
import os
import wikipedia
import re
import enchant

companies = []
products = []

def populate():
'''
	Populates a list of countries based on folder names in the given corpus.
	In future, we would like to scrape wikipedia pages for this.
'''
	global companies
	global products
	for root, dirs, files in os.walk('./NDTV_mobile_reviews_Classified/'):
		for d in dirs:
			x = d.split('-')
			companies.append(x[0].lower())
			products.append(x[1].lower())
	
	
def f1(h,t):
'''
	If word in pre-populated list of companies, tag as organization
'''
	if(h[2][h[3]].lower() in companies and t=="ORGANIZATION"):
		return 1
	else:
		return 0

def f2(h,t):
'''
	If word is followed by a product term, tag as organization
'''
	if((h[2].length-2 >= h[3]) and h[2][h[3]+1].lower() in products and t=="ORGANIZATION"):
		return 1
	else:
		return 0

def f3(h,t):
'''
	If abbreviation/symbol followed by numbers, tag MONEY
'''
	money_abbs = ["rs.","Rs.","$"]
	for i in money_abbs:
		if(i in h[2][h[3] and t=='MONEY'):
			return 1
	return 0

def f4(h,t):
'''
	If symbol numbers followed by common currency words, tag MONEY
'''
	money_words = ["rupees","pounds","euros","dollars","yen"]
	if((h[2].length-2 >= h[3]) and h[2][h[3]+1].lower() in money_words and t=="MONEY"):
		return 1
	else:
		return 0

def f5(h,t):
'''
	GPE is possibly after location in the format '<LOCATION> in/at <GPE>' as in 'Mount Everest at Nepal'
'''
	if(h[0]=="LOCATION" and h[1]=="OTHER" and t=="GPE"):
		return 1
	else:
		return 0

def f6(h,t):
'''
	Location is possibly after facility in the format '<FACILITY> in/at <LOCATION>' as in 'Institute at Roorkee'
'''
	if(h[0]=="FACILITY" and h[1]=="OTHER" and t=="LOCATION"):
		return 1
	else:
		return 0

def f7(h,t):
'''
	Tag person if a popular name or if not an english word
'''
	d = enchant.Dict("en_US")
	common_names = ["Bob","Phil","George","Jack","John","Paul","Frank"]
	if((h[2][h[3]] in common_names and t=="PERSON") or (d.check(h[2][h[3]])==False and t=="PERSON"):
		return 1
	else:
		return 0

def f8(h,t):
""" Feature function to identify whether the word is a TIME tag.
	Using the regex for time"""
	try:
		time.strptime(h[2][h[3]], '%H:%M')
		return 1
	except:
		return 0

def f9(h,t):
	""" Feature function to identify whether the word is a TIME tag.
	Using, the principle digits followed phrases like "am","pm","hrs".
	or if the subsequent word is one of the above and the current word is a set of digits"""

	time_words = ['am','pm','hrs','a.m','p.m']

	flag = False
	#check if the current word has any of the tags above
	for i in time_words:
		if i in h[2][h[3]]:
			flag = True
			break

	#check if current word has 4 digits and next word has any of the tags above
	p = re.compile('\d\d\d\d')
	m = p.match(h[2][h[3]])


	if m :
		try:
			if h[2][h[3]+1] == time_words[2]:
				flag=True
		except:
			pass

	if flag and t=='TIME':
		return 1
	else:
		return 0

def f10(h,t):
"""Feature function to identify whether the given word is DATE tag. 
	Returns true only if the date is in datetime format"""
	date_str = h[2][h[3]]

	if t!="DATE":
		return 0

	try:
        datetime.datetime.strptime(date_str, '%Y-%m-%d')
        return 1
    except ValueError:
        pass


    try:
        datetime.datetime.strptime(date_str, '%d-%m-%Y')
       	return 1
    except ValueError:
        pass


    try:
        datetime.datetime.strptime(date_str, '%Y/%m/%d')
        return 1
    except ValueError:
        pass


    try:
        datetime.datetime.strptime(date_str, '%d/%m/%Y')
        return 1
    except ValueError:
        pass

	return 0

def f11(h,t):
	"""Feature function to identify whether the given word is DATE tag. 
	Returns true only if the word contains any of the months, or the word is a set of digits followed by months, 
	or if the current word is a set of digits following a previous DATE/TIME tag"""

	if t!="DATE":
		return 0

	months = ["January","February","March","April","May","June","July","August","September","October","November","December"]

	if h2[h[3]] in months:
		return 1

	try:
		if h2[h[3]+1] in months:
			p = re.compile("\d{1-2}")
			m = p.match(h[2][h[3]])
			if m:
				return 1
	except:
		pass

	if t2 == "TIME" or t2 == "DATE":
		p = re.compile("\d{1-2}")
		m = p.match(h[2][h[3]])
		if m:
			return 1
def f12(h,t):
	""" Feature function to identify whether the word is a PERCENT tag.
	Using, the principle digits followed phrases like "%","percent","pct", "percentage".
	or if the subsequent word is one of the above and the current word is a set of digits"""

	p_words = ["%","percent","pct"]

	flag = False
	#check if the current word has any of the tags above
	for i in p_words:
		if i in h[2][h[3]]:
			flag = True
			break

	#check if current word has digits and next word has any of the tags above
	p = re.compile('(\d+\.)?\d+')
	m = p.match(h[2][h[3]])


	if m :
		try:
			if h[2][h[3]+1] in p_words:
				flag=True
		except:
			pass

	if flag and t=='PERCENT':
		return 1
	else:
		return 0

populate()
