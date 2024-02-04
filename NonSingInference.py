from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np

#pip install scipy, matplotlib and numpy if not installed on testing device. 
#Simply run the code and provide inputs
#After the diagram is closed by user, it would ask for additional inputs.
#Might take a while to start

#This function is used for generating membership functions. First argument is the range of it.
def membership_gen(arr, func_type, is_elbowed, is_thresh, thresh, scaler, params): #scaler shall be used for scaling. thresh for clipping
    calcs = []
    if func_type == "trap":
        if(x < arr[0] or x > arr[3]):
            return 0
        if(x>=arr[0] and x <= arr[1]):
            if(not arr[0] == 0):
                calc = (x - arr[0]) / (arr[1] - arr[0])
            else:
                calc = 1
        elif(x > arr[1] and x <= arr[2]):
            calc = 1
        else:
            if(not arr[3] == 100):
                calc = 1 - ((x - arr[2]) / (arr[3] - arr[2]))
            else:
                calc = 1
    elif func_type == "norm":
        norms = [norm.pdf(i, loc=(arr[0] + arr[1])/2, scale=(arr[1]-arr[0])/6) for i in list(round(val,2) for val in np.arange(arr[0], arr[1],0.1))]
        for x in list(round(val,1) for val in np.arange(arr[0], arr[1], 0.1)):
          calc = norm.pdf(x, loc=(arr[0] + arr[1])/2, scale=(arr[1]-arr[0])/6)
          calc = (calc - (min(norms))) / (max(norms) - min(norms))

          if(is_elbowed == 'l' and x < (arr[1] + arr[0])/2):
              calc = 1
          elif(is_elbowed == 'r' and x > (arr[1] + arr[0])/2):
            calc = 1
          if(is_thresh and calc >= thresh):
              calc = thresh
          calcs.append(calc)
    elif func_type == "stupa":  #our custom function based on sigmoids
        def sigmoid(x, base):
          x = x-base
          return 1 / (1 + np.exp(params[0]*-x))
        x = np.arange(arr[0], arr[1], 0.1)
        calcs = [min(a,b) for a,b in zip(sigmoid(x, arr[0]), (sigmoid(x, arr[0]))[::-1])]
        calcs = [((a - min(calcs)) / (max(calcs) - min(calcs)))*scaler for a in calcs] #mix-max normalization to bring all values between 0 and 1, and then scaling
        if(is_elbowed == 'l'): #elbowing
          calcs = [(scaler if i < (arr[1] + arr[0])/2 else a) for a,i in zip(calcs, x)]
        elif(is_elbowed == 'r'):
          calcs = [(scaler if i > (arr[1] + arr[0])/2 else a) for a,i in zip(calcs, x)]
        if(is_thresh):
          calcs = [(1 if a > thresh else a) for a in calcs]
    return calcs #returns list of y values


#This function is used for generating linguistic variables. In the parameters we specify number of linguistic terms, names of linguistic terms, their ranges, etc.
def ling_var_gen(num_terms, labels, terms_ranges, terms_func_types, terms_thresh, scalers = [], elbow_mode = False, thresh_mode=False, params = []):
  membership_arrs = []
  if len(scalers) == 0:   #Scalers are used for scaling consequent membership funcs based on firing strengths of rules. Useless for antecendats so just set all to 1
    scalers = [1]*num_terms
  for i in range(0,num_terms): #loop to generate mem func for each term
    thresh_val = terms_thresh[i] if thresh_mode else -1
    ebm = 'n'
    if(elbow_mode and i == 0): #left sided elbowing for left-most term's mem function, if elbowing mode is on
      ebm = 'l'
    elif(elbow_mode and i == num_terms-1):  #right sided elbowing for right-most term's mem function, if elbowing mode is on
      ebm = 'r'
    y = membership_gen(terms_ranges[i],  terms_func_types[i], ebm, thresh_mode, thresh_val, scalers[i], params)
    membership_arrs.append([list(round(val,1) for val in np.arange(terms_ranges[i][0], terms_ranges[i][-1],0.1)),y, labels[i], ebm])
  return membership_arrs # [[x's, y's, LingTerm, ElbowMode], [x's, y's, LingTerm, ElbowMode], ...]

def get_mem_for_crisp(crisp_val, ling_var):  #used to fuzzify a crisp value
  memberships = {}
  for l in ling_var:
    if(crisp_val < l[0][0]):
      if(l[3] == 'l'):  #if crisp less than (meaning more left than) the range of leftmost ling var, and elbow mode on, then mem val would be 1
        memberships[l[2]] = 1
      else:
        memberships[l[2]] = 0
    elif(crisp_val > l[0][-1]): #if crisp greater than (meaning more right than) the range of rightmost ling var, and elbow mode on, then mem val would be 1
      if(l[3] == 'r'):
        memberships[l[2]] = 1
      else:
        memberships[l[2]] = 0
    else:
      index = l[0].index(crisp_val)
      memberships[l[2]] = l[1][index]
  return memberships # {LingTermA: mem ,LingTermB: mem...}

#function to find intersection of input range function and membership functions
def find_intersected_input_memfuncs(range_of_input, range_norm, num_mem_funcs, mem_funcs):
  ranges_of_intersected_mem_funcs = []
  #for each membership function of ling var, I would maintain an array indicating the start point of its intersection with range function, and end point
  for i in range(0, num_mem_funcs):
    ranges_of_intersected_mem_funcs.append([False, False]) #first bool indicates whether start point has been found. 2nd indicates for end point
  intersected_ys_for_all_mems = []
  for x in list(np.arange(range_of_input[0],range_of_input[1],0.1)):
    for j, mem_func in zip(range(0,num_mem_funcs), mem_funcs):
      if(x >= mem_func[0][0] and ranges_of_intersected_mem_funcs[j][0] == False):
        ranges_of_intersected_mem_funcs[j][0] = True # start point of intersection found
        ranges_of_intersected_mem_funcs[j].append(x)
      if(ranges_of_intersected_mem_funcs[j][0] == True and ranges_of_intersected_mem_funcs[j][1] == False and x >= mem_func[0][-1]):
        ranges_of_intersected_mem_funcs[j][1] = True #end point of intersection found
        ranges_of_intersected_mem_funcs[j].append(x)
  for m in ranges_of_intersected_mem_funcs:
    if(m[0] == True and m[1] == False): #if start point of intersection found but end point not found, end point becomes end point of input range
      m.append(range_of_input[1])

  intersected_mem_funcs = []
  for i in range(0, len(ranges_of_intersected_mem_funcs)):
    intersected_mem_funcs.append([[], []])
  for r,j in zip(ranges_of_intersected_mem_funcs, range(0,len(ranges_of_intersected_mem_funcs))):
    if r[0] == True:
      for x in list(np.arange(r[2], r[3], 0.1)):
        intersected_mem_funcs[j][0].append(x)
        i1 = range_norm[0].index(round(x,1))
        i2 = mem_funcs[j][0].index(round(x,1))
        intersected_mem_funcs[j][1].append(min(range_norm[1][i1], mem_funcs[j][1][i2]))
  return intersected_mem_funcs #[x's, intersected y's] for each mem func


#Function to find jacard similarity between input range function and membership functions, when using similarity based composition in NSFLS
def find_jaccardian_of_intersected_input_memfuncs(range_of_input, range_norm, num_mem_funcs, mem_funcs):
  ranges_of_intersected_mem_funcs = []
  #for each membership function of ling var, I would maintain an array indicating the start point of its intersection with range function, and end point
  for i in range(0, num_mem_funcs):
    ranges_of_intersected_mem_funcs.append([False, False]) #first bool indicates whether start point has been found. 2nd indicates for end point
  intersected_ys_for_all_mems = []
  for x in list(np.arange(range_of_input[0],range_of_input[1],0.1)):
    for j, mem_func in zip(range(0,num_mem_funcs), mem_funcs):
      if(x >= mem_func[0][0] and ranges_of_intersected_mem_funcs[j][0] == False):
        ranges_of_intersected_mem_funcs[j][0] = True # start point of intersection found
        ranges_of_intersected_mem_funcs[j].append(x)
      if(ranges_of_intersected_mem_funcs[j][0] == True and ranges_of_intersected_mem_funcs[j][1] == False and x >= mem_func[0][-1]):
        ranges_of_intersected_mem_funcs[j][1] = True #end point of intersection found
        ranges_of_intersected_mem_funcs[j].append(x)
  for m in ranges_of_intersected_mem_funcs:
    if(m[0] == True and m[1] == False): #if start point of intersection found but end point not found, end point becomes end point of input range
      m.append(range_of_input[1])

  intersected_mem_funcs = []
  unioned_mem_funcs = []
  #Jaccardian is intersection / union
  for i in range(0, len(ranges_of_intersected_mem_funcs)):
    intersected_mem_funcs.append([])
    unioned_mem_funcs.append([])
  for r,j in zip(ranges_of_intersected_mem_funcs, range(0,len(ranges_of_intersected_mem_funcs))):
    if r[0] == True:
      for x in list(np.arange(r[2], r[3], 0.1)):
        i1 = range_norm[0].index(round(x,1))
        i2 = mem_funcs[j][0].index(round(x,1))
        intersected_mem_funcs[j].append(min(range_norm[1][i1], mem_funcs[j][1][i2]))
        unioned_mem_funcs[j].append(max(range_norm[1][i1], mem_funcs[j][1][i2]))

  sigma_union = 0
  sigma_intersection = 0
  jaccardian_similarities = []
  for union, intersection in zip(unioned_mem_funcs, intersected_mem_funcs):
    sigma_union = sum(union)
    sigma_intersection = sum(intersection)
    if(not (sigma_union == 0)):
      jaccard = sigma_intersection / sigma_union
      jaccardian_similarities.append(jaccard)
    else:
      jaccardian_similarities.append(0)


  return jaccardian_similarities

#to find y value corresponding to centroid of intersection, when using centroid based composition in NSFLS
def firing_stren_from_centroid(x_arr,y_arr):
  x_arr = list(map(lambda x: round(x,1),x_arr))
  if(len(y_arr) == 0):
    return 0
  numerator = 0
  denom = 0
  for x,y in zip(x_arr, y_arr):
    numerator += x*y
    denom += y
  centroid = numerator/denom
  y_corresponding_to_centroid = y_arr[x_arr.index(round(centroid,1))]
  return y_corresponding_to_centroid

use_jaccardian = True
if(input("Please enter c if you'd like to use centroid based NSFLS instead of jaccard similarity based. Else enter any other letter : ") == 'c'):
  use_jaccardian = False

#BodyTemperature
colors = [(49/255,154/255,240/255), (113/255,215/255,244/255), (53/255, 236/255, 158/255), (248/255,175/255, 41/255), (235/255,54/255,100/255)]
temp_input_range = (input("Enter range of body temperature (separated by space): ")).split()
BodyTemperature = ling_var_gen(5,['SevereHypothermia', 'Hypothermia', 'NormalTemp', 'LowFever', 'HighFever'],[[31,35.5],[35,36.5], [36,37.5], [37,39], [38,40]], ['stupa', 'stupa', 'stupa', 'stupa', 'stupa'], [], [], True, False, params = [6])
for s,i in zip(BodyTemperature, range(0,len(BodyTemperature))):
  plt.plot(s[0],s[1], color=colors[i], label = s[2])
plt.plot(list(np.arange(int(temp_input_range[0]), int(temp_input_range[1]),0.1)), membership_gen([int(temp_input_range[0]), int(temp_input_range[1])], 'norm', False, False, -1, 1, []), color = (0,0,0), label = "Input Range Gauss")
plt.legend()
plt.title("BodyTemperature: membership functions and input range function")
plt.show()
temp_membership_vals = {}
input_membership_bodytemp = [list(map(lambda a: round(a,1) , list(np.arange(int(temp_input_range[0]), int(temp_input_range[1]), 0.1)))), membership_gen([int(temp_input_range[0]), int(temp_input_range[1])], 'norm', False, False, -1, 1, [])]

if not use_jaccardian:
  for i, intersection_func in zip(list(range(0,5)), find_intersected_input_memfuncs([int(temp_input_range[0]), int(temp_input_range[1])], input_membership_bodytemp, 5, BodyTemperature)):
    plt.plot(intersection_func[0], intersection_func[1], color = colors[i])
    temp_membership_vals[BodyTemperature[i][2]] = firing_stren_from_centroid(intersection_func[0], intersection_func[1])
  plt.title("BodyTemperature: Input range function intersected with each membership function")
  plt.show()
else:
  jaccardians = find_jaccardian_of_intersected_input_memfuncs([int(temp_input_range[0]), int(temp_input_range[1])], input_membership_bodytemp, 5, BodyTemperature)
  for i in range(0,5):
   temp_membership_vals[BodyTemperature[i][2]] = jaccardians[i]

#HeadacheLevel
colors = [(53/255, 236/255, 158/255), (248/255,175/255, 41/255), (235/255,54/255,100/255)]
headache_input_range = (input("Enter range of headache level between 0 and 10: ")).split()
HeadacheLevel = ling_var_gen(3,['LowHeadache', 'MediumHeadache', 'HighHeadache'],[[0,5], [4,8], [6.5,10]], ['stupa', 'stupa', 'stupa'], [], [], True, False, params = [3])
for s,i in zip(HeadacheLevel, range(0,len(HeadacheLevel))):
  plt.plot(s[0],s[1], color=colors[i], label = s[2])
plt.plot(list(np.arange(int(headache_input_range[0]), int(headache_input_range[1]),0.1)), membership_gen([int(headache_input_range[0]), int(headache_input_range[1])], 'norm', False, False, -1, 1, []), color = (0,0,0), label = "Input Rage Gauss")
plt.legend()
plt.title("HeadacheLevel: membership functions and input range function")
plt.show()

input_membership_headache = [list(map(lambda a: round(a,1) , list(np.arange(int(headache_input_range[0]), int(headache_input_range[1]), 0.1)))), membership_gen([int(headache_input_range[0]), int(headache_input_range[1])], 'norm', False, False, -1, 1, [])]
headache_membership_vals = {}
if not use_jaccardian:
  for i, intersection_func in zip(list(range(0,3)), find_intersected_input_memfuncs([int(headache_input_range[0]), int(headache_input_range[1])], input_membership_headache, 3, HeadacheLevel)):
    plt.plot(intersection_func[0], intersection_func[1], color = colors[i])
    headache_membership_vals[HeadacheLevel[i][2]] = firing_stren_from_centroid(intersection_func[0], intersection_func[1])
  plt.title("HeadacheLevel: Input range function intersected with each membership function")
  plt.show()
else:
  jaccardians = find_jaccardian_of_intersected_input_memfuncs([int(headache_input_range[0]), int(headache_input_range[1])], input_membership_headache, 3, HeadacheLevel)
  for i in range(0,3):
    headache_membership_vals[HeadacheLevel[i][2]] = jaccardians[i]

#AgeGroup
colors = [(49/255,154/255,240/255), (113/255,215/255,244/255), (248/255,175/255, 41/255), (235/255,54/255,100/255)]
age_input_range = (input("Enter range of age between 0 and 130: ")).split()
AgeGroup = ling_var_gen(4,['Infant', 'Young', 'MiddleAged', 'Old'], [[0,9],[6,35], [27,65], [55,130]], ['stupa', 'stupa', 'stupa', 'stupa'], [], [], True, False,  params = [0.4])
for s,i in zip(AgeGroup, range(0,len(AgeGroup))):
  plt.plot(s[0],s[1], color=colors[i], label = s[2])
plt.plot(list(np.arange(int(age_input_range[0]), int(age_input_range[1]),0.1)), membership_gen([int(age_input_range[0]), int(age_input_range[1])], 'norm', False, False, -1, 1, []), color = (0,0,0), label = "Input Rage Gauss")
plt.legend()
plt.title("AgeGroup: membership functions and input range function")
plt.show()

input_membership_age = [list(map(lambda a: round(a,1) , list(np.arange(int(age_input_range[0]), int(age_input_range[1]), 0.1)))), membership_gen([int(age_input_range[0]), int(age_input_range[1])], 'norm', False, False, -1, 1, [])]
age_membership_vals = {}
if not use_jaccardian:
  for i, intersection_func in zip(list(range(0,4)), find_intersected_input_memfuncs([int(age_input_range[0]), int(age_input_range[1])], input_membership_age, 4, AgeGroup)):
    plt.plot(intersection_func[0], intersection_func[1], color = colors[i])
    age_membership_vals[AgeGroup[i][2]] = firing_stren_from_centroid(intersection_func[0], intersection_func[1])
  plt.title("AgeGroup: Input range function intersected with each membership function")
  plt.show()
else:
  jaccardians = find_jaccardian_of_intersected_input_memfuncs([int(age_input_range[0]), int(age_input_range[1])], input_membership_age, 4, AgeGroup)
  for i in range(0,4):
    age_membership_vals[AgeGroup[i][2]] = jaccardians[i]

print(temp_membership_vals)
print(headache_membership_vals)
print(age_membership_vals)


rules = [
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "LowHeadache", "AgeGroup": "Infant", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "LowHeadache", "AgeGroup": "Young", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "LowHeadache", "AgeGroup": "MiddleAged", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "LowHeadache", "AgeGroup": "Old", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Infant", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Young", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "MediumHeadache", "AgeGroup": "MiddleAged", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Old", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "HighHeadache", "AgeGroup": "Infant", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "HighHeadache", "AgeGroup": "Young", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "HighHeadache", "AgeGroup": "MiddleAged", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "SevereHypothermia", "HeadacheLevel": "HighHeadache", "AgeGroup": "Old", "Severity": "Urgent", 'Value': -1},

    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "LowHeadache", "AgeGroup": "Infant", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "LowHeadache", "AgeGroup": "Young", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "LowHeadache", "AgeGroup": "MiddleAged", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "LowHeadache", "AgeGroup": "Old", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Infant", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Young", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "MediumHeadache", "AgeGroup": "MiddleAged", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Old", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "HighHeadache", "AgeGroup": "Infant", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "HighHeadache", "AgeGroup": "Young", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "HighHeadache", "AgeGroup": "MiddleAged", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "Hypothermia", "HeadacheLevel": "HighHeadache", "AgeGroup": "Old", "Severity": "Urgent", 'Value': -1},

    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "LowHeadache", "AgeGroup": "Infant", "Severity": "NotSevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "LowHeadache", "AgeGroup": "Young", "Severity": "NotSevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "LowHeadache", "AgeGroup": "MiddleAged", "Severity": "NotSevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "LowHeadache", "AgeGroup": "Old", "Severity": "NotSevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Infant", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Young", "Severity": "NotSevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "MediumHeadache", "AgeGroup": "MiddleAged", "Severity": "NotSevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Old", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "HighHeadache", "AgeGroup": "Infant", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "HighHeadache", "AgeGroup": "Young", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "HighHeadache", "AgeGroup": "MiddleAged", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "NormalTemp", "HeadacheLevel": "HighHeadache", "AgeGroup": "Old", "Severity": "Severe", 'Value': -1},

    {"BodyTemperature": "LowFever", "HeadacheLevel": "LowHeadache", "AgeGroup": "Infant", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "LowHeadache", "AgeGroup": "Young", "Severity": "NotSevere", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "LowHeadache", "AgeGroup": "MiddleAged", "Severity": "NotSevere", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "LowHeadache", "AgeGroup": "Old", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Infant", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Young", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "MediumHeadache", "AgeGroup": "MiddleAged", "Severity": "MildlySevere", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Old", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "HighHeadache", "AgeGroup": "Infant", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "HighHeadache", "AgeGroup": "Young", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "HighHeadache", "AgeGroup": "MiddleAged", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "LowFever", "HeadacheLevel": "HighHeadache", "AgeGroup": "Old", "Severity": "Urgent", 'Value': -1},

    {"BodyTemperature": "HighFever", "HeadacheLevel": "LowHeadache", "AgeGroup": "Infant", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "LowHeadache", "AgeGroup": "Young", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "LowHeadache", "AgeGroup": "MiddleAged", "Severity": "Severe", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "LowHeadache", "AgeGroup": "Old", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Infant", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Young", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "MediumHeadache", "AgeGroup": "MiddleAged", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "MediumHeadache", "AgeGroup": "Old", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "HighHeadache", "AgeGroup": "Infant", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "HighHeadache", "AgeGroup": "Young", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "HighHeadache", "AgeGroup": "MiddleAged", "Severity": "Urgent", 'Value': -1},
    {"BodyTemperature": "HighFever", "HeadacheLevel": "HighHeadache", "AgeGroup": "Old", "Severity": "Urgent", 'Value': -1},
]


#calculate firing strength of each rule
for rule in rules:
  rule["Value"] =  ((min(temp_membership_vals[rule["BodyTemperature"]], headache_membership_vals[rule["HeadacheLevel"]], age_membership_vals[rule["AgeGroup"]])) if (min(temp_membership_vals[rule["BodyTemperature"]], headache_membership_vals[rule["HeadacheLevel"]], age_membership_vals[rule["AgeGroup"]])) <= 0.4 else max(temp_membership_vals[rule["BodyTemperature"]], headache_membership_vals[rule["HeadacheLevel"]], age_membership_vals[rule["AgeGroup"]])) \
  if rule["Severity"] == 'Urgent' \
  else \
  (min(temp_membership_vals[rule["BodyTemperature"]], headache_membership_vals[rule["HeadacheLevel"]], age_membership_vals[rule["AgeGroup"]]))


consequent_membership_vals = {"NotSevere": -1, "MildlySevere": -1, "Severe": -1, "Urgent": -1}


#find max firing strength of rules with same severity level. These would become the 4 firing strengths of our consequent ling terms
for severityLevel in consequent_membership_vals:
  consequent_membership_vals[severityLevel] = max([rule["Value"] for rule in rules if rule['Severity'] == severityLevel])

print(consequent_membership_vals)

#scale the 4 consequent mem funcs based on respective firing strengths. An alternative approach would be clipping.
colors = [(53/255, 236/255, 158/255), (248/255,175/255, 41/255), (235/255,54/255,100/255), (150/255,75/255,0/255)]
Severity = ling_var_gen(4,['NotSevere', 'MildlySevere', 'Severe', 'Urgent'], [[0,30],[20,60], [50,80], [75,100]], ['stupa', 'stupa', 'stupa', 'stupa'], [], [consequent_membership_vals['NotSevere'], consequent_membership_vals['MildlySevere'], consequent_membership_vals['Severe'], consequent_membership_vals['Urgent']], True, False, [0.4])
for s,i in zip(Severity, list(range(0, len(Severity)))):
  plt.plot(s[0],s[1], color=colors[i], label = s[2])
plt.legend()
plt.title("Scaled Consequent Functions")
plt.show()

#aggregation
#combine all consequent mem funcs
aggregated_output = []

for i in range(0,101):
  temporary_arr = []
  for s in Severity:
    if(i < s[0][0] or i > s[0][-1]):
      temporary_arr.append(0)
    else:
      index = s[0].index(i)
      temporary_arr.append(s[1][index])
  aggregated_output.append(max(temporary_arr))

plt.plot(range(0,101), aggregated_output)
plt.title("Aggregated Output")
plt.show()


#centroid defuzz
numerator = 0
denom = 0
for a,i in zip(aggregated_output, range(0,101)):
    numerator += a * i
    denom += a
centroid_output = numerator/denom
print("\nCentroid difuzz: ", centroid_output)

#bisector defuzz
half_area = denom/2
area_so_far = 0
for a,i in zip(aggregated_output, range(0,101)):
  area_so_far += a
  if(area_so_far >= half_area):
    break
print("Bisector difuzz: ", i)
