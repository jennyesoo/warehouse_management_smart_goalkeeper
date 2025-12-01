
def get_file_name(computer_no):
    """
    Get the output file name based on current time stamp
    """
    from datetime import datetime
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    name = 'SkyEyes_{}_{}.jpg'.format(computer_no, current_time)
    return name


def detect_direction(direction, frame_size, ratio):
    """
    Get the limit according to the direction that is wanted
    """
    if direction == 'right':
        limit = int(frame_size - frame_size * 1/ratio)
    elif direction == 'left':
        limit = int(frame_size - frame_size * (ratio - 1)/ratio)
    elif direction == 'up':
        limit = int(frame_size - frame_size * 1/ratio)
    elif direction == 'down':
        limit = int(frame_size - frame_size * (ratio - 1)/ratio)
    return limit


def bbox2points(bbox):
    """
    From bounding box yolo format
    """
    x, y, w, h = bbox
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax


def center_record(track_res, record_dict):
    """
    Recoring centroid of each detected and tracked item.
    """
    for i in range(len(track_res)):
        ids = track_res[i][-1]
        x1, y1, x2, y2 = track_res[i][0], track_res[i][1], track_res[i][2], track_res[i][3]
        center = [[x1 + (x2 - x1)/2, y1 + (y2 - y1)/2]]
        # If the key already existed in the dict, then just add new centroid
        if ids in record_dict.keys():
            record_dict[ids].extend(center)
        # If not, add new ids into the key of dict
        else:
            record_dict[ids] = center
    return record_dict


def check_direction(track_res, detect_direction, alarm_limit, record_dict, duration=5, min_dist=3):
    """
    Check is the centroid of detected item moved across limit line.
    If yes, check the direction of the item in dict of centroid(pts).
    """
    import numpy as np
    to_alarm_table = []
    to_alarm_ids = []
    for i in range(len(track_res)):
        ids = track_res[i][-1]
        x1, y1, x2, y2 = track_res[i][0], track_res[i][1], track_res[i][2], track_res[i][3]
        center = np.array([x1 + (x2 - x1)/2, y1 + (y2 - y1)/2])
        if len(record_dict[ids]) >= duration:
            mean_point = np.mean(record_dict[ids][-2 * duration:], axis=0)
            diff = center - mean_point
            if (detect_direction == 'right') & (center[0] >= alarm_limit):
                if (np.sign(diff[0]) == 1) & (np.abs(diff[0]) >= min_dist):
                    to_alarm_table.append(track_res[i])
                    to_alarm_ids.append(ids)
            elif (detect_direction == 'left') & (center[0] <= alarm_limit):
                if (np.sign(diff[0]) == -1) & (np.abs(diff[0]) >= min_dist):
                    to_alarm_table.append(track_res[i])
                    to_alarm_ids.append(ids)
            elif (detect_direction == 'up') & (center[1] >= alarm_limit):
                if (np.sign(diff[1]) == 1) & (np.abs(diff[1]) >= min_dist):
                    to_alarm_table.append(track_res[i])
                    to_alarm_ids.append(ids)
            elif (detect_direction == 'down') & (center[1] <= alarm_limit):
                if (np.sign(diff[1]) == -1) & (np.abs(diff[1]) >= min_dist):
                    to_alarm_table.append(track_res[i])
                    to_alarm_ids.append(ids)
    return to_alarm_table, to_alarm_ids

def draw_boxes(alarm_table, image, colors, items_score):
    import cv2
    for i in range(len(alarm_table)):
        x1, y1, x2, y2 = int(alarm_table[i][0]), int(alarm_table[i][1]), int(alarm_table[i][2]), int(alarm_table[i][3])
        label = items_score[i][0]
        confidence = items_score[i][1]
        item_id = int(alarm_table[i][-1])
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], 2)
        cv2.putText(image, "ID: {} {} [{:.2f}]".format(item_id, label, float(confidence)),
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def yolo2sort(detection):
    '''
    Rearrange yolo detection result to sort format
    '''
    dets = []
    items = []
    for i in range(len(detection)):
        (detect_item, score, bbox) = detection[i]
        xmin, ymin, xmax, ymax = bbox2points(bbox)
        dets.append([xmin, ymin, xmax, ymax, score])
        items.append([detect_item, score])
    return dets, items