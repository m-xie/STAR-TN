import numpy as np
import alphashape
import cv2
from skimage.segmentation import flood_fill

def bounding_box(points):

    x_coordinates, y_coordinates = points

    width = max(x_coordinates) - min(x_coordinates)
    height = max(y_coordinates) - min(y_coordinates)

    x_center = (width/2) + min(x_coordinates)
    y_center = (height/2) + min(y_coordinates)
    box_center = [x_center, y_center]
    return width, height, box_center

def get_clusters(X,y):
    s = np.argsort(y)
    return np.split(X[s], np.unique(y[s], return_index=True)[1][1:])

def hull_and_kmeans(data_object):
    arr2_binary = data_object.astype('uint8')

    # find contours
    image_points = np.where(arr2_binary > 120)
    shape_pts = []

    for idx in range(image_points[0].shape[0]):
        shape_pts.append((image_points[1][idx], image_points[0][idx]))

    alpha_shape_total = alphashape.alphashape(shape_pts, 1/6)
    points = []
    if alpha_shape_total.geom_type != 'Polygon':
        for polygon in list(alpha_shape_total.geoms):
            points.extend(polygon.exterior.coords[:-1])
    else:
        points.extend(alpha_shape_total.exterior.coords[:-1])

    hull_arr2 = []

    if len(points) < 1:
        return arr2_binary

    arr = [[np.asarray(points[0], dtype="int32")]]
    for point in points:
        arr = np.append(arr, [[np.asarray(point,dtype="int32")]], axis=0)

    hull_arr2.append(arr)
    hull_arr2 = np.asarray(hull_arr2)

    cv2.drawContours(arr2_binary, hull_arr2, -1, (254, 0, 0), 1)

    return arr2_binary

def loop(arr2, area_prev, arr_prev_flooded, arr_prev_plane, area_percent=3, crop=41):
    upper_limit = 120
    box_points_prev = np.where(arr_prev_plane == 1)
    width_prev, height_prev, box_center_prev = bounding_box(box_points_prev)

    arr2_flooded = np.zeros_like(arr2) == 1
    #loop through all x and y spots and compare back to the previous z to see where they overlay
    flag = False
    for x in range(np.size(arr2, 0)):
        for z in range(np.size(arr2, 1)):
            tol = upper_limit - np.ceil(arr2[x, z])
            if arr_prev_flooded[x, z] == 1 and arr2_flooded[x, z] == 0 and (arr2[x,z]<upper_limit):
                arr2_forflood = arr2.copy()
                arr2_forflood[arr2_forflood < arr2[x,z]] = arr2[x,z]

                arr_flooded_temp = flood_fill(arr2_forflood, (x, z), 1, connectivity = 1, tolerance = tol)
                arr_flooded_temp = arr_flooded_temp == 1

                #pass the new floodfill back to the arr2 fill
                check_area = arr2_flooded | arr_flooded_temp

                check_area_prev = check_area | arr_prev_plane
                #find contours for check_area
                box_points = np.where(check_area_prev == 1)
                width, height, box_center = bounding_box(box_points)

                area_truth = (sum(sum(check_area))) <(area_percent*area_prev) or sum(sum(check_area)) < 50

                acceptable_box_change = (width < crop*.6 and height < crop*.6) and (((((width-width_prev)/width_prev) < 1.5 and (height-height_prev)/height_prev<0.66)) or (((width-width_prev)/width_prev <1.5 and (height-height_prev)/height_prev) < 0.66))

                if area_truth:
                    arr2_flooded = arr2_flooded | arr_flooded_temp
                elif not area_truth:
                    flag = True
                else:
                    print(f"Issue is the box change")

    arr2_area = sum(sum(arr2_flooded.astype(int)))

    if arr2_area > 0:
        width_curr, height_curr, _ = bounding_box(np.where(arr2_flooded == 1))
        if (width_curr < width_prev * .1 and height_curr < height_prev * .1):
            arr2_flooded = np.zeros_like(arr2) == 1
            print("Box became too small")
            arr2_area = sum(sum(arr2_flooded.astype(int)))

    print(f"Calculated area for next loop is {arr2_area}")
    return arr2_flooded, flag

def remove_background(arr_binary, curr_flood=None):
    connectivity = 4
    output = cv2.connectedComponentsWithStats(arr_binary, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    mask = np.zeros(arr_binary.shape, dtype="uint8")

    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        keepWidth = w>5 and w<50
        keepHeight = h>5 and h <65
        keepArea = area>30 and area < 1500
        keepX = (x+(w/2)) >5 and (x+(w/2))<35
        keepY = (y+(h/2)) > 5 and (y+(h/2))<35
        if curr_flood is None:
            keepCloseToSeg = True
        else:
            keepCloseToSeg = np.sum((cv2.dilate(np.asarray(curr_flood*255, dtype="uint8"), np.ones((4,4))) > 0) * (labels == i)) > 0

        if all((keepWidth, keepHeight, keepArea, keepX, keepY, keepCloseToSeg)):
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

    return mask

#function to move forward
def press_forward_y(arr_prev_flooded, y, area_prev, crop_scan, segmentation_y, area_percent=3, crop=41):


    arr2_og = crop_scan[:, y+1, :]
    #pull out the  coronal plane
    arr2_og = np.array(arr2_og, dtype=np.uint16)
    #convert data types

    arr2_blurred = cv2.GaussianBlur(arr2_og, (3,3), 0)

    dat = np.array(arr2_blurred, dtype = np.uint8) #can switch to arr2_blurred or arr2_og
    #scale to uint8

    #loop through all x and y spots and compare back to the previous z to see hwere they overlay
    arr2_flooded = np.zeros_like(arr2_og) == 1

    arr2_binary = cv2.adaptiveThreshold(dat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    centerPix = dat[int(np.round(np.sum(np.sum(arr_prev_flooded,axis=1) * np.arange(arr_prev_flooded.shape[0])) / np.sum(arr_prev_flooded))), \
        int(np.round(np.sum(np.sum(arr_prev_flooded,axis=0) * np.arange(arr_prev_flooded.shape[1])) / np.sum(arr_prev_flooded)))]
    th = (np.percentile(dat.flatten(),95) - centerPix)/3 + centerPix
    _, arr2_binary_thresh = cv2.threshold(dat, th, 255, cv2.THRESH_BINARY)
    arr2_binary = remove_background(arr2_binary | arr2_binary_thresh, arr_prev_flooded)

    out, flag = loop(arr2_binary, area_prev, arr_prev_flooded, arr_prev_flooded)

    arr2_flooded = arr2_flooded | out
    arr2_area = sum(sum(arr2_flooded.astype(int)))

    arr4_flooded = arr2_flooded.copy()
    arr4_area = sum(sum(arr4_flooded))

    if flag == True:
        print(f"Original area size is {arr2_area}")
        arr2_flooded = np.zeros_like(arr2_og) == 1

        arr2 = hull_and_kmeans(arr2_binary)

        out, flag = loop(arr2, area_prev, arr_prev_flooded, arr_prev_flooded)
        arr2_flooded = out | arr2_flooded

        arr2_old = arr2_flooded.copy();
        tmp = np.zeros((crop, crop));

        seg_mean = np.mean(arr2_og[arr2_flooded])
        seg_sd = np.std(arr2_og[arr2_flooded])

        for i in range(arr2_flooded.shape[0]):
            for j in range(arr2_flooded.shape[1]):
                if arr2_old[i,j] == True:
                    if i > 0 and arr2[i-1,j] == 254 and arr2_og[i-1,j] - seg_mean < seg_sd:
                        tmp[i-1,j] = tmp[i-1,j] + 1
                    if j > 0 and arr2[i,j-1] == 254 and arr2_og[i,j-1] - seg_mean < seg_sd:
                        tmp[i,j-1] = tmp[i,j-1] + 1
                    if i < arr2_flooded.shape[0]-1 and arr2[i+1,j] == 254 and arr2_og[i+1,j] - seg_mean < seg_sd:
                        tmp[i+1,j] = tmp[i+1,j] + 1
                    if j < arr2_flooded.shape[1]-1 and arr2[i,j+1] == 254 and arr2_og[i,j+1] - seg_mean < seg_sd:
                        tmp[i,j+1] = tmp[i,j+1] + 1
                    if i > 0 and j > 0 and arr2[i-1,j-1] == 254 and arr2_og[i-1,j-1] - seg_mean < seg_sd:
                        tmp[i-1,j-1] = tmp[i-1,j-1] + 1
                    if i > 0 and j < arr2_flooded.shape[1]-1 and arr2[i-1,j+1] == 254 and arr2_og[i-1,j+1] - seg_mean < seg_sd:
                        tmp[i-1,j+1] = tmp[i-1,j+1] + 1
                    if i < arr2_flooded.shape[0]-1 and j > 0 and arr2[i+1,j-1] == 254 and arr2_og[i+1,j-1] - seg_mean < seg_sd:
                        tmp[i+1,j-1] = tmp[i+1,j-1] + 1
                    if  i < arr2_flooded.shape[0]-1 and j < arr2_flooded.shape[1]-1 and arr2[i+1,j+1] == 254 and arr2_og[i+1,j+1] - seg_mean < seg_sd:
                        tmp[i+1,j+1] = tmp[i+1,j+1] + 1
        arr2_flooded = arr2_flooded | (tmp > 3)

        arr3_binary_all = cv2.adaptiveThreshold(dat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
        _, arr3_binary_thresh = cv2.threshold(dat, 130, 255, cv2.THRESH_BINARY)
        arr3_binary_all = arr3_binary_all | arr3_binary_thresh

        #create the new binary
        mask = remove_background(arr3_binary_all, arr_prev_flooded)
        arr3 = mask.astype('uint8')
        arr3[arr2_flooded] = 254

        arr2_dilated = cv2.dilate(np.asarray(arr2_flooded*255, dtype="uint8"), np.ones((3,3)))

        out, flag = loop(arr3, area_prev, arr2_dilated == 255, arr_prev_flooded)
        check_arr2_flooded = out | arr2_flooded

        if sum(sum(check_arr2_flooded)) < area_percent*area_prev:
            print(f"Filled in the gaps with area {sum(sum(check_arr2_flooded))} and allowable change is {area_percent*area_prev} at {y+1}")
            arr2_flooded = out | arr2_flooded
        else:
            print(f"Did not fill in the gaps with area {sum(sum(check_arr2_flooded))} and allowable change is {area_percent*area_prev} at {y+1}")


    arr2_area = sum(sum(arr2_flooded.astype(int)))
    print(f"[FORWARD] Calculated area for next forwards step is {arr2_area}")
    print(f"Previous area was {area_prev}")

    if (arr2_area < area_percent * area_prev or arr2_area < 100) and arr2_area < 200 and y + 2 < crop and arr2_area > 2:
        segmentation_y[:, y+1, :] = arr2_flooded.astype(int)
        press_forward_y(arr2_flooded, y+1, arr2_area, crop_scan, segmentation_y)
    else:
        print(f"[FORWARD] Stopping forward movement at {y+1} and area is {arr2_area}")

def press_backward_y(arr_prev_flooded, y, area_prev, crop_scan, segmentation_y, area_percent=3, crop=41):
    #pull out the next array

    arr2_og = crop_scan[:, y-1, :]
    #pull out the  coronal plane
    arr2_og = np.array(arr2_og, dtype=np.uint16)
    #convert data types

    arr2_blurred = cv2.GaussianBlur(arr2_og, (3,3), 0)

    dat = np.array(arr2_blurred, dtype = np.uint8) #can switch to arr2_blurred or arr2_og
    #scale to uint8

    #loop through all x and y spots and compare back to the previous z to see hwere they overlay
    arr2_flooded = np.zeros_like(arr2_og) == 1

    arr2_binary = cv2.adaptiveThreshold(dat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    centerPix = dat[int(np.round(np.sum(np.sum(arr_prev_flooded,axis=1) * np.arange(arr_prev_flooded.shape[0])) / np.sum(arr_prev_flooded))), \
        int(np.round(np.sum(np.sum(arr_prev_flooded,axis=0) * np.arange(arr_prev_flooded.shape[1])) / np.sum(arr_prev_flooded)))]
    th = (np.percentile(dat.flatten(),95) - centerPix)/3 + centerPix
    _, arr2_binary_thresh = cv2.threshold(dat, th, 255, cv2.THRESH_BINARY)
    arr2_binary = remove_background(arr2_binary | arr2_binary_thresh, arr_prev_flooded)

    out, flag = loop(arr2_binary, area_prev, arr_prev_flooded, arr_prev_flooded)

    arr2_flooded = arr2_flooded | out
    arr2_area = sum(sum(arr2_flooded.astype(int)))

    arr4_flooded = arr2_flooded.copy()
    arr4_area = sum(sum(arr4_flooded))

    if flag == True:
        print(f"Original area size is {arr2_area}")
        arr2_flooded = np.zeros_like(arr2_og) == 1

        arr2 = hull_and_kmeans(arr2_binary)

        out, flag = loop(arr2, area_prev, arr_prev_flooded, arr_prev_flooded)
        arr2_flooded = out | arr2_flooded

        arr2_old = arr2_flooded.copy()
        tmp = np.zeros((crop,crop))

        seg_mean = np.mean(arr2_og[arr2_flooded])
        seg_sd = np.std(arr2_og[arr2_flooded])

        for i in range(arr2_flooded.shape[0]):
            for j in range(arr2_flooded.shape[1]):
                if arr2_old[i,j] == True:
                    if i > 0 and arr2[i-1,j] == 254 and arr2_og[i-1,j] - seg_mean < seg_sd:
                        tmp[i-1,j] = tmp[i-1,j] + 1
                    if j > 0 and arr2[i,j-1] == 254 and arr2_og[i,j-1] - seg_mean < seg_sd:
                        tmp[i,j-1] = tmp[i,j-1] + 1
                    if i < arr2_flooded.shape[0]-1 and arr2[i+1,j] == 254 and arr2_og[i+1,j] - seg_mean < seg_sd:
                        tmp[i+1,j] = tmp[i+1,j] + 1
                    if j < arr2_flooded.shape[1]-1 and arr2[i,j+1] == 254 and arr2_og[i,j+1] - seg_mean < seg_sd:
                        tmp[i,j+1] = tmp[i,j+1] + 1
                    if i > 0 and j > 0 and arr2[i-1,j-1] == 254 and arr2_og[i-1,j-1] - seg_mean < seg_sd:
                        tmp[i-1,j-1] = tmp[i-1,j-1] + 1
                    if i > 0 and j < arr2_flooded.shape[1]-1 and arr2[i-1,j+1] == 254 and arr2_og[i-1,j+1] - seg_mean < seg_sd:
                        tmp[i-1,j+1] = tmp[i-1,j+1] + 1
                    if i < arr2_flooded.shape[0]-1 and j > 0 and arr2[i+1,j-1] == 254 and arr2_og[i+1,j-1] - seg_mean < seg_sd:
                        tmp[i+1,j-1] = tmp[i+1,j-1] + 1
                    if  i < arr2_flooded.shape[0]-1 and j < arr2_flooded.shape[1]-1 and arr2[i+1,j+1] == 254 and arr2_og[i+1,j+1] - seg_mean < seg_sd:
                        tmp[i+1,j+1] = tmp[i+1,j+1] + 1
        arr2_flooded = arr2_flooded | (tmp > 3)

        arr3_binary_all = cv2.adaptiveThreshold(dat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
        _, arr3_binary_thresh = cv2.threshold(dat, 130, 255, cv2.THRESH_BINARY)
        arr3_binary_all = arr3_binary_all | arr3_binary_thresh

        #create the new binary
        mask = remove_background(arr3_binary_all, arr_prev_flooded)
        arr3 = mask.astype('uint8')
        arr3[arr2_flooded] = 254

        arr2_dilated = cv2.dilate(np.asarray(arr2_flooded*255, dtype="uint8"), np.ones((3,3)))

        out, flag = loop(arr3, area_prev, arr2_dilated == 255, arr_prev_flooded)
        check_arr2_flooded = out | arr2_flooded

        if sum(sum(check_arr2_flooded)) < area_percent*area_prev:
            print(f"Filled in the gaps with area {sum(sum(check_arr2_flooded))} and allowable change is {area_percent*area_prev} at {y-1}")
            arr2_flooded = out | arr2_flooded
        else:
            print(f"Did not fill in the gaps with area {sum(sum(check_arr2_flooded))} and allowable change is {area_percent*area_prev} at {y-1}")


    arr2_area = sum(sum(arr2_flooded.astype(int)))
    print(f"[BACK] Calculated area for next backwards step is {arr2_area}")
    print(f"Previous area was {area_prev}")

    if (arr2_area <area_percent*area_prev or arr2_area < 100) and arr2_area < 200 and y - 2 >= 0 and arr2_area > 2:
        segmentation_y[:, y-1, :] = arr2_flooded.astype(int)
        press_backward_y(arr2_flooded, y-1, arr2_area, crop_scan, segmentation_y)
    else:
        print(f"[BACK] Stopping backward movement at {y} and area is {arr2_area}")