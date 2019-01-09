# coding:utf-8
#

# get longitude and latitude
def get_lati_longi(line):
    lati = float(line.split(' ')[2])
    longi = abs(float(line.split(' ')[3]))
    return lati, longi

# whether the key in the dictionary
def key_in_dic(key, dic):
    if key in dic:
        return True
    else:
        return False

import calendar
import time

import datetime

class Time:
    def __init__(self,check_time):
        self._check_time = check_time

    def getDateByTime(self):
        self.myDate = []
        t = str(time.strftime('%Y-%m-'))
        for i in range(1, 32):
            timeStr = t + str(i)
            try:
                # string to format
                tmp = time.strptime(timeStr, '%Y-%m-%d')
                # whether Saturday or Sunday
                if (tmp.tm_wday != 6) and (tmp.tm_wday != 5):
                    self.myDate.append(time.strftime('%Y-%m-%d', tmp))
            except:
                print('Date transboundary')
        if len(self.myDate) == 0:
            self.myDate.append(time.strftime('%Y-%m-%d'))
        return self.myDate

    # string to datetime
    def string_toDatetime(self,string):
        return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S")

    def getDateByDateTime(self):
        self.myDate = []
        self.myDate_dict = {}
        # get system time
        # now = datetime.datetime.now()
        # specified time
        now_1 = self._check_time
        now = self.string_toDatetime(now_1)
        tmp = now.strftime('%Y-%m-')
        # through calendar got first weekday in this month，and the number of days in this month
        t = calendar.monthrange(now.year, now.month)
        for i in range(1, t[1]):
            dateTmp = tmp + str(i)
            myDateTmp = datetime.datetime.strptime(dateTmp, '%Y-%m-%d')
            if myDateTmp.isoweekday() != 6 and myDateTmp.isoweekday() != 7:
                self.myDate.append(myDateTmp.strftime('%Y-%m-%d'))
        if len(self.myDate) == 0:
            self.myDate.append(now.strftime('%Y-%m-%d'))
        for date in self.myDate:
            self.myDate_dict[date] = ''
        return self.myDate_dict

# check-ins in weekday at 12:00-15:00 and 19:00-24:00
# check-ins in weekend at 8:00-12:00 and 15:00-19:00

# check-ins in afternoon or night
def time_in_weekday(line):
    hour = int(line.split(' ')[1].split('T')[1].split(':')[0])
    if 12 <= hour < 15 or 19 <= hour <= 23:
        return True
    else:
        return False

# check-ins in morning or afternoon
def time_in_weekend(line):
    hour = int(line.split(' ')[1].split('T')[1].split(':')[0])
    if 8 <= hour < 12 or 15 <= hour < 19:
        return True
    else:
        return False

# got nodes from edges
def get_nodes(line):
    new_str = line.split('\n')[0]
    new_list = new_str.split(' ')
    line_l = new_list[0]
    line_r = new_list[1]
    return line_l, line_r


# according to node degree, return the first three top nodes
def related_three_node(my_lst,save_lst,my_dict):
    son_dictionary = {}
    save_dictionary = {}
    three_node = []
    i = 0
    for element in my_lst:
        son_dictionary[element] = my_dict[element]
    for sl in save_lst:
        save_dictionary[sl] = ''
    for k,v in sorted(son_dictionary.iteritems(), key=lambda asd:asd[1], reverse=True):
        if key_in_dic(k,save_dictionary):
            pass
        else:
            i += 1
            three_node.append(k)
            if i == 3:
                break
    # one node in the first three top nodes may occur in save_node,if true,then got a new one
    return three_node

# random select edges function
import random
def random_nodes_pair(inputs, rate=0.7, seed=0):
    del_candidate = []
    total_length = len(inputs)
    random.seed(seed)
    # got the input dataset linenum into index
    index = [i for i in range(total_length)]
    # select size as input×rate data to delete
    del_index = random.sample(index, int(float(total_length) * rate))
    for del_i in del_index:
        del_candidate.append(inputs[del_i])
    print('del_candidate length:' + str(len(del_candidate)))
    return del_candidate

def flatten(sublist):
    my_str = ''
    count = 0
    for element in sublist:
        count += 1
        if count == len(sublist):
            my_str += str(element) + '\n'
        else:
            my_str += str(element) + ' '
    return my_str

# record user trajectory "poi-frequency" dictionary
def count_poi(line):
    list1 = line.split()
    set1 = set(list1)
    list2 = list(set1)
    # create a null dictionary
    dir1 = {}

    for x in range(len(list2)):
        dir1[list2[x]] = 0  # dicitionary initiallize as 0
        for y in range(len(list1)):
            if list2[x] == list1[y]:
                dir1[list2[x]] += 1
    return dir1
