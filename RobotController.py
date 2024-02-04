from scipy.stats import norm

left = int(input("ENTER LEFT SENSOR READING: "))
right = int(input("ENTER RIGHT SENSOR READING: "))

if(left > 50 or right > 50 or left < 0 or right < 0):
    raise Exception('Enter values between 0 and 50')
go_straight = 0
go_right = 0
go_left = 0
go_hard = 0

close_ranges = [0,10, 30, 50]
far_ranges = [0,20, 40, 50]

def close_membership(x):
    if(x >= close_ranges[0] and x <= close_ranges[1]):
        return 1
    elif(x > close_ranges[1] and x <= close_ranges[2]):
        return 1-((x- close_ranges[1]) / (close_ranges[2] - close_ranges[1]))
    else:
        return 0 

def far_membership(x):
    if(x >= far_ranges[0] and x <= far_ranges[1]):
        return 0
    elif(x > far_ranges[1] and x <= far_ranges[2]):
        return ((x- far_ranges[1]) / (far_ranges[2] - far_ranges[1]))
    else:
        return 1


closes = []
fars = []

for i in range(0,51):
    closes.append(close_membership(i))
    fars.append(far_membership(i))

from matplotlib import pyplot as plt

plt.plot(closes, label = "Close")
plt.scatter([left], [close_membership(left)])
plt.annotate(f"({left} , {close_membership(left)})", (left, close_membership(left)), textcoords = "offset points", xytext = (0,10), ha = 'center')
plt.scatter([left], [far_membership(left)])
plt.annotate(f"({left} , {far_membership(left)})", (left, far_membership(left)), textcoords = "offset points", xytext = (0,10), ha = 'center')
plt.plot(fars, label = "Far")
plt.title('Left Sensor')
plt.xlabel('Distance')
plt.legend()
plt.show()

plt.plot(closes, label = "Close")
plt.scatter([right], [close_membership(right)])
plt.annotate(f"({right} , {close_membership(right)})", (right, close_membership(right)), textcoords = "offset points", xytext = (0,10), ha = 'center')
plt.scatter([right], [far_membership(right)])
plt.annotate(f"({right} , {far_membership(right)})", (right, far_membership(right)), textcoords = "offset points", xytext = (0,10), ha = 'center')
plt.plot(fars, label = "Far")
plt.title('Right Sensor')
plt.xlabel('Distance')
plt.legend()
plt.show()

left_close = close_membership(left)
left_far = far_membership(left)
right_close = close_membership(right)
right_far = far_membership(right)

#L FAR AND R FAR - STRAIGHT
go_straight = min(left_far, right_far)

#L FAR AND R CLOSE - RIGHT
go_left= min(left_far, right_close)

#L CLOSE AND R FAR - LEFT
go_right = min(left_close, right_far)


#L CLOSE AND R CLOSE - HARD
go_hard= min(left_close, right_close)


print("STRAIGHT {}  RIGHT {}  LEFT  {}   HARD  {}".format(go_straight, go_right, go_left, go_hard))

steer_hard_ranges = [0, 0, 5, 15]
steer_left_ranges = [5, 20, 30, 45]  
steer_straight_ranges = [30, 45, 55, 70]  
steer_right_ranges = [60, 70, 80, 100]

def output_membership(x, arr, is_thresh, thresh, is_trap = False):
    calc = 0
    if is_trap:
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
    else:
        norms = [norm.pdf(i, loc=(arr[0] + arr[3])/2, scale=(arr[3]-arr[0])/6) for i in range(arr[0], arr[3]+1)]
        calc = norm.pdf(x, loc=(arr[0] + arr[3])/2, scale=(arr[3]-arr[0])/6)
        calc = (calc - (min(norms))) / (max(norms) - min(norms))
    if(is_thresh and calc >= thresh):
        calc = thresh
    return calc
   
hard_steer_arr = []
for x in range(steer_hard_ranges[0], steer_hard_ranges[3]+1):
    hard_steer_arr.append(output_membership(x, steer_hard_ranges, False, 0, True))

left_steer_arr = []
for x in range(steer_left_ranges[0], steer_left_ranges[3]+1):
    left_steer_arr.append(output_membership(x, steer_left_ranges, False, 0))

straight_steer_arr = []
for x in range(steer_straight_ranges[0], steer_straight_ranges[3]+1):
    straight_steer_arr.append(output_membership(x, steer_straight_ranges, False, 0))

right_steer_arr = []
for x in range(steer_right_ranges[0], steer_right_ranges[3]+1):
    right_steer_arr.append(output_membership(x, steer_right_ranges, False, 0))

plt.plot(range(0,16,1),hard_steer_arr, color='red', label = 'Hard Turn')
plt.plot(range(5,46,1),left_steer_arr, color='blue', label = 'Left')
plt.plot(range(30,71,1),straight_steer_arr, color='green', label = 'Straight')
plt.plot(range(55,101,1),right_steer_arr, color='orange', label = 'Right')
plt.legend()
plt.title('Consequent Memberships of Steering')
plt.xlabel('Steering Output')
#plt.plot(range(0,41,1),[go_left] * 41)
#plt.plot(range(30,61,1),[go_straight] * 31)
#plt.plot(range(50,101,1),[go_right] * 51)

hard_steer_arr_thresh = []
hard_thresh_area = 0
for x in range(steer_hard_ranges[0], steer_hard_ranges[3]+1):
    calc = output_membership(x, steer_hard_ranges, True, go_hard, True)
    hard_steer_arr_thresh.append(calc)
    hard_thresh_area += calc

left_steer_arr_thresh = []
left_thresh_area = 0
for x in range(steer_left_ranges[0], steer_left_ranges[3]+1):
    calc = output_membership(x, steer_left_ranges, True, go_left)
    left_steer_arr_thresh.append(calc)
    left_thresh_area += calc

straight_steer_arr_thresh = []
straight_thresh_area = 0
for x in range(steer_straight_ranges[0], steer_straight_ranges[3]+1):
    calc = output_membership(x, steer_straight_ranges, True, go_straight)
    straight_steer_arr_thresh.append(calc)
    straight_thresh_area += calc

right_steer_arr_thresh = []
right_thresh_area = 0
for x in range(steer_right_ranges[0], steer_right_ranges[3]+1):
    calc = output_membership(x, steer_right_ranges, True, go_right)
    right_steer_arr_thresh.append(calc)
    right_thresh_area += calc

plt.fill_between(range(0,16,1),hard_steer_arr_thresh, alpha=0.5, color='red')
plt.fill_between(range(5,46,1),left_steer_arr_thresh, alpha=0.5, color='blue')
plt.fill_between(range(30,71,1),straight_steer_arr_thresh, alpha=0.5, color='green')
plt.fill_between(range(55,101,1),right_steer_arr_thresh, alpha=0.5, color='orange')
plt.show()

plt.fill_between(range(0,16,1),hard_steer_arr_thresh, alpha=0.5, color='red')
plt.fill_between(range(5,46,1),left_steer_arr_thresh, alpha=0.5, color='blue')
plt.fill_between(range(30,71,1),straight_steer_arr_thresh, alpha=0.5, color='green')
plt.fill_between(range(55,101,1),right_steer_arr_thresh, alpha=0.5, color='orange')

area = left_thresh_area + straight_thresh_area + right_thresh_area


#INTERSECTION FINDING CODE
# triangle_a_x = [steer_straight_ranges[0], (steer_straight_ranges[0]+steer_left_ranges[3])/2 , steer_left_ranges[3]]
# triangle_a_y = [0, output_membership(triangle_a_x[1], steer_straight_ranges, False, 0) , 0]
# intersection_a_thresh = min(go_left, go_straight, output_membership(triangle_a_x[1], steer_straight_ranges, False, 0))
# calc = 0
# for x in range(triangle_a_x[0], triangle_a_x[2]+1):
#     val = min(output_membership(x, steer_left_ranges,True,intersection_a_thresh), output_membership(x, steer_straight_ranges,True,intersection_a_thresh))
#     calc += val
#     plt.plot((x,x),(0,val), color='black')

# print(calc)
# print(1/2 * ((triangle_a_x[2] - triangle_a_x[0]) * triangle_a_y[1]))
# area -= calc

# triangle_b_x = [steer_right_ranges[0], (steer_right_ranges[0]+steer_straight_ranges[3])/2 , steer_straight_ranges[3]]
# triangle_b_y = [0, output_membership(triangle_b_x[1], steer_right_ranges, False, 0) , 0]
# intersection_b_thresh = min(go_straight, go_right, output_membership(triangle_b_x[1], steer_right_ranges, False, 0))
# calc = 0
# for x in range(triangle_b_x[0], triangle_b_x[2]+1):
#     val = min(output_membership(x, steer_straight_ranges,True,intersection_b_thresh), output_membership(x, steer_right_ranges,True,intersection_b_thresh))
#     calc += val
#     plt.plot((x,x),(0,val), color='black')

# print(calc)
# print(1/2 * ((triangle_b_x[2] - triangle_b_x[0]) * triangle_b_y[1]))
# area -= calc
# print("Area is {}".format(area))
# plt.fill_between(range(0,41,1),left_steer_arr_thresh, color='red', alpha=0.5)
# plt.fill_between(range(30,71,1),straight_steer_arr_thresh, color='red', alpha=0.5)
# plt.fill_between(range(60,101,1),right_steer_arr_thresh, color='red', alpha=0.5)

# plt.fill_between(range(0,41,1),left_steer_arr_thresh, alpha=0.5)
# plt.fill_between(range(30,71,1),straight_steer_arr_thresh, alpha=0.5)
# plt.fill_between(range(60,101,1),right_steer_arr_thresh, alpha=0.5)


#plt.fill(triangle_a_x, triangle_a_y, alpha=0.5)
#plt.fill(triangle_b_x, triangle_b_y, alpha=0.5)

# upper_triangle_a_height = triangle_a_y[1] - go_straight
# upper_triangle_a_base = (triangle_a_x[2] - triangle_a_x[0]) * (upper_triangle_a_height / triangle_a_y[1])
# lower_triangle_a_area = (1/2 * (triangle_a_x[2] - triangle_a_x[0]) * triangle_a_y[1]) - (1/2 * upper_triangle_a_base * upper_triangle_a_height)

# upper_triangle_b_height = triangle_b_y[1] - go_straight
# upper_triangle_a_base = (triangle_a_x[2] - triangle_a_x[0]) * (upper_triangle_a_height / triangle_a_y[1])
# lower_triangle_a_area = (1/2 * (triangle_a_x[2] - triangle_a_x[0]) * triangle_a_y[1]) - (1/2 * upper_triangle_a_base * upper_triangle_a_height)
import numpy as np

numerator = 0
memberships = []
area = 0
for i in np.arange(0,101, 1):
    mem = max(output_membership(i, steer_hard_ranges,True,go_hard, true), output_membership(i, steer_left_ranges,True,go_left), output_membership(i, steer_straight_ranges,True,go_straight), output_membership(i, steer_right_ranges,True,go_right))
    memberships.append(mem)
    numerator += mem*i
    area += mem

defuzzified = numerator / area
plt.plot(memberships, color='black', lw=2)
print("Area: {} ".format(area))
print('Defuzzified crisp output of steering: {}'.format(defuzzified))
plt.title('Steering Controller Graph')
plt.ylim(0,1)
plt.show()
output_membership_values = [output_membership(defuzzified, steer_hard_ranges,False,go_hard, True), output_membership(defuzzified, steer_left_ranges,False,go_left), output_membership(defuzzified, steer_straight_ranges,False,go_straight), output_membership(defuzzified, steer_right_ranges,False,go_right)]


# import skfuzzy as skf
# skf_cdefuzzified = skf.defuzz(np.array(range(0,101)), np.array(memberships), 'centroid')
# print(skf_cdefuzzified)