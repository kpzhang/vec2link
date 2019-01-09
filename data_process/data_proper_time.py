# coding:utf-8
import utils
# You can choose the location range of check-ins
check_in = '../data/gowalla_totalCheckins.txt'
data_range = '../data/data_range.txt'
d_r = open(data_range,'w')
c_i = open(check_in)
for line in c_i.readlines():
    latitude,longitude = utils.get_lati_longi(line)
    if 40.0 <= latitude <= 45.0 and 84.0 <= longitude <= 91.0:
        d_r.write(line)
    else:
        pass
d_r.close()
c_i.close()

# You can choose the check_time of check-ins
# 2010/5-2010/10(GWL)
year_lst = [2010]
month_lst = [i for i in range(1,13)]
year_month_dict = {}
for y in year_lst:
    for m in month_lst:
        # if y == 2010 and m > 4 or y == 2010 and m < 11:
        if 4 < m < 11:
            if len(str(m)) == 1:
                year_month_dict[str(y) + '-0' + str(m)] = ''
            else:
                year_month_dict[str(y)+'-'+str(m)] = ''
        else:
            pass

# 2009/2-2009/7(BK)
# year_lst = [2009]
# month_lst = [i for i in range(1,13)]
# year_month_dict = {}
# for y in year_lst:
#     for m in month_lst:
#         # if y == 2010 and m > 4 or y == 2010 and m < 11:
#         if 1 < m < 8:
#             if len(str(m)) == 1:
#                 year_month_dict[str(y) + '-0' + str(m)] = ''
#             else:
#                 year_month_dict[str(y)+'-'+str(m)] = ''
#         else:
#             pass

# # 2012/4-2012/9(NYC&&TKY)
# year_lst = [2012]
# month_lst = [i for i in range(1,13)]
# year_month_dict = {}
# for y in year_lst:
#     for m in month_lst:
#         # if y == 2010 and m > 4 or y == 2010 and m < 11:
#         if 3 < m < 10:
#             if len(str(m)) == 1:
#                 year_month_dict[str(y) + '-0' + str(m)] = ''
#             else:
#                 year_month_dict[str(y)+'-'+str(m)] = ''
#         else:
#             pass

#
user_poi = '../data/data_range.txt'
poi_filter1 = '../data/poi_filter1.txt'
u_p = open(user_poi)
p_f = open(poi_filter1,'w')
for line in u_p.readlines():
    if utils.key_in_dic(line.split()[1].split('T')[0][:7],year_month_dict):
        p_f.write(line)
    else:
        pass
u_p.close()
p_f.close()

# check-ins in weekday at 12:00-15:00 and 19:00-24:00
# check-ins in weekend at 8:00-12:00 and 15:00-19:00
user_poi = '../data/poi_filter1.txt'
weekday_poi = '../data/weekday_poi.txt'
weekend_poi = '../data/weekend_poi.txt'
wd_p = open(weekday_poi,'w')
we_p = open(weekend_poi,'w')
time_dict = {}
weekday_dict = {}
u_p = open(user_poi)
for line in u_p.readlines():
    if utils.key_in_dic(line.split()[1].split('T')[0][:7],time_dict):
        if utils.key_in_dic(line.split()[1].split('T')[0],weekday_dict):
            wd_p.write(line)
        else:
            we_p.write(line)
    else:
        time_dict[line.split()[1].split('T')[0][:7]]=''
        my_date = line.split()[1].split('T')[0] + ' ' + line.split()[1].split('T')[1].split('Z')[0]

        my_time = utils.Time(my_date)
        # obtain all weekdays
        weekday_dict = my_time.getDateByDateTime()
        # record the weekdays down
        if utils.key_in_dic(line.split()[1].split('T')[0],weekday_dict):
            wd_p.write(line)
        else:
            we_p.write(line)
u_p.close()
wd_p.close()
we_p.close()

poi_filter = '../data/poi_filter2.txt'
p_f = open(poi_filter,'w')

wd_p = open(weekday_poi)
for line in wd_p.readlines():
    if utils.time_in_weekday(line):
        p_f.write(line)
    else:
        pass
wd_p.close()

we_p = open(weekend_poi)
for line in we_p.readlines():
    if utils.time_in_weekend(line):
        p_f.write(line)
    else:
        pass
we_p.close()

p_f.close()

user_dict = {}
p_f = open(poi_filter)
for line in p_f.readlines():
    user_dict[line.split()[0]] = ''
p_f.close()
