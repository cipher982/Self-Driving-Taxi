import csv
import codecs
import fnmatch


f=codecs.open("model_output.csv","rb","utf-16")
csvread=csv.reader(f,delimiter='\n')
csvread.next()
lines_list = []
passes = 0
fails = 0
rewards = []
errors = []
total_rewards = []
#deadlines = []

for line in csvread:
	lines_list.append(line)

lines_iter = iter(lines_list)

for line in lines_iter:
    if line == ['Environment.act(): Primary agent has reached destination!']:
	    passes += 1
    elif line == ['Environment.step(): Primary agent ran out of time! Trial aborted.']:
	    fails += 1
    elif line == ['Reward is']:
	    rewards += lines_iter.next()

#deadlines = fnmatch.filter(lines_list, 'Environment.reset(): Trial set up with start =*')
		
rewards = map(float, rewards)
total_rewards = sum(rewards)

for reward in rewards:
    if reward < 0:
	    errors.append(reward)
		
errors = map(float, errors)

		
print "Your cab made %d succesful trips, and %d late." % (passes, fails) 
print "It also had a total rewards of %d and a total error amount of %d" % (total_rewards, sum(errors))