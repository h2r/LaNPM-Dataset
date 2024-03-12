from ai2thor.controller import Controller
import curses
from PIL import Image
import numpy as np
import transformations as tft
from math import degrees
import json
import zipfile
from datetime import datetime
import logging



data = [] #each point is a step
scene = ''
command = ""

def convert_to_euler(rot_quat):

    # Extract quaternion from the dictionary
    quaternion = [rot_quat['x'], rot_quat['y'], rot_quat['z'], rot_quat['w']]

    # Convert quaternion to Euler angles
    roll, pitch, yaw = tft.euler_from_quaternion(quaternion)

    roll = degrees(roll)
    pitch = degrees(pitch)
    yaw = degrees(yaw)

    return (roll, pitch, yaw)

def get_objs_pos(held_objs, all_objs):
    pos_rot_dic = {}
    for held_obj_id in held_objs: #usually 1 held object
        for obj_dic in all_objs:
            if obj_dic['objectId'] == held_obj_id:
                pos_rot_dic[held_obj_id] = {'position': obj_dic['position'],  'rotation':obj_dic['rotation']}
    return pos_rot_dic

prev_state_body = []
prev_state_ee = []
prev_hand_sphere_center = 10000000
prev_held_objs = []
prev_held_objs_state = {}

def gather_data(event):
    global scene, prev_state_body, prev_state_ee, prev_hand_sphere_center, prev_held_objs, prev_held_objs_state
    scene = event.metadata['sceneName']

    dic = {}
    dic['sim_time'] = event.metadata['currentTime']
    dic['wall-clock_time'] = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    dic['rgb'] = event.frame
    dic['depth'] = event.depth_frame
    dic['seg'] = event.instance_segmentation_frame
    global_coord_agent = event.metadata['agent']['position']
    yaw_agent =  event.metadata['agent']['rotation']['y'] #degrees
    dic['state_body'] = [global_coord_agent['x'], global_coord_agent['y'], global_coord_agent['z'], yaw_agent]
    global_coord_ee = event.metadata["arm"]["joints"][3]['position']
    rot_ee = convert_to_euler(event.metadata["arm"]["joints"][3]['rotation'])
    dic['state_ee'] = [global_coord_ee['x'], global_coord_ee['y'],global_coord_ee['z'], rot_ee[0], rot_ee[1], rot_ee[2]]
    dic['hand_sphere_center'] = event.metadata['arm']['handSphereRadius']
    dic['held_objs'] = event.metadata['arm']['heldObjects']
    pos_rot_dic = get_objs_pos(event.metadata['arm']['heldObjects'], event.metadata["objects"])
    dic['held_objs_state'] =  pos_rot_dic
    dic['action'] = event.metadata['lastAction']

    if prev_state_body == dic['state_body'] and prev_state_ee == dic['state_ee'] and prev_hand_sphere_center == dic['hand_sphere_center'] and prev_held_objs == dic['held_objs'] and prev_held_objs_state == dic['held_objs_state']:
        return
    
    prev_state_body = dic['state_body']
    prev_state_ee = dic['state_ee']
    prev_hand_sphere_center = dic['hand_sphere_center']
    prev_held_objs = dic['held_objs']
    prev_held_objs_state = dic['held_objs_state']

    data.append(dic)


# def main(stdscr):
controller = Controller(
    agentMode="arm",
    massThreshold=1,
    #examples:
    # scene="FloorPlan_Train1_4",
    # scene="FloorPlan_Train4_5",
    # scene="FloorPlan_Val2_3",

    #used for the Aryan and George commands
    # scene='FloorPlan_Val2_5',
    # scene='FloorPlan_Val2_1',
    # scene = 'FloorPlan_Train9_4',
    # scene = "FloorPlan_Train5_2",
    scene = "FloorPlan_Val1_1",
    snapToGrid=False,
    visibilityDistance=1.5,
    gridSize=0.25,
    renderDepthImage=True,
    renderInstanceSegmentation=True,
    width= 1280,
    height= 720,
    fieldOfView=60
)

#collect the intial data before moving
gather_data(controller.last_event)

stdscr = curses.initscr()
curses.echo()
stdscr.nodelay(True)  # Do not wait for the user input
# stdscr.timeout(50)  # Wait 100 ms for input
stdscr.keypad(True)  # Enable keypad mode to capture special keys
i = 0
incr = 0.025
x = 0
y = 0
z = 0
fixedDeltaTime = 0.02
move = 0.15

event = controller.step(
    action="MoveArmBase",
    y=i,
    speed=1,
    returnToStart=False,
    fixedDeltaTime=fixedDeltaTime
)
i+=incr


while True:
    user_input = stdscr.getch()
    # stdscr.refresh()
    
    # event = controller.interact()

    # if user_input != -1:  # -1 means no input
    # if user_input == ord('q'):
    #     break
    if user_input == ord('d'):
        event = controller.step(
            action="MoveAgent",
            right = move,
            returnToStart=False,
            speed=1,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord("w"):
        event = controller.step(
            action="MoveAgent",
            ahead = move,
            returnToStart=False,
            speed=1,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord("a"):
        event = controller.step(
            action="MoveAgent",
            right= -move,
            returnToStart=False,
            speed=1,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord("s"):
        event = controller.step(
            action="MoveAgent",
            ahead= -move,
            returnToStart=False,
            speed=1,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord("l"):
        event = controller.step(
            action="RotateAgent",
            degrees=20,
            returnToStart=False,
            speed=1,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord("j"):
        event = controller.step(
            action="RotateAgent",
            degrees=-20,
            returnToStart=False,
            speed=1,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord('i'):
        controller.step("LookUp")
        gather_data(event)
    elif user_input == ord('k'):
        controller.step("LookDown")
        gather_data(event)
    elif user_input == curses.KEY_UP:
        i+=incr
        event = controller.step(
            action="MoveArmBase",
            y=i,
            speed=1,
            returnToStart=False,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == curses.KEY_DOWN:
        i-=incr
        event = controller.step(
            action="MoveArmBase",
            y=i,
            speed=1,
            returnToStart=False,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord('7'):
        x += incr
        event = controller.step(
            action="MoveArm",
            position=dict(x=x, y=y, z=z),
            coordinateSpace="wrist",
            restrictMovement=False,
            speed=1,
            returnToStart=False,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord('4'):
        x -= incr
        event = controller.step(
            action="MoveArm",
            position=dict(x=x, y=y, z=z),
            coordinateSpace="wrist",
            restrictMovement=False,
            speed=1,
            returnToStart=False,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord('8'):
        y += incr
        event = controller.step(
            action="MoveArm",
            position=dict(x=x, y=y, z=z),
            coordinateSpace="wrist",
            restrictMovement=False,
            speed=1,
            returnToStart=False,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord('5'):
        y -= incr
        event = controller.step(
            action="MoveArm",
            position=dict(x=x, y=y, z=z),
            coordinateSpace="wrist",
            restrictMovement=False,
            speed=1,
            returnToStart=False,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord('9'):
        z += incr
        event = controller.step(
            action="MoveArm",
            position=dict(x=x, y=y, z=z),
            coordinateSpace="wrist",
            restrictMovement=False,
            speed=1,
            returnToStart=False,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord('6'):
        z -= incr
        event = controller.step(
            action="MoveArm",
            position=dict(x=x, y=y, z=z),
            coordinateSpace="wrist",
            restrictMovement=False,
            speed=1,
            returnToStart=False,
            fixedDeltaTime=fixedDeltaTime
        )
        gather_data(event)
    elif user_input == ord('g'):
        controller.step(
            action="SetHandSphereRadius",
            radius=0.1
        )
        event = controller.step(action="PickupObject")
        gather_data(event)
    elif user_input == ord('r'):
        controller.step(
            action="SetHandSphereRadius",
            radius=0.06
        )
        event = controller.step(action="ReleaseObject")
        gather_data(event)
    elif user_input == 27: #esc key
        break



# curses.wrapper(main)


###########################################################################################
'''data collection'''
###########################################################################################

final_dic = {command, "scene":scene, "steps":data}

def ndarray_to_list(obj):
    logging.info("Processing an object of type %s", type(obj))
    if isinstance(obj, dict):
        return {key: ndarray_to_list(value) if key in ['seg', 'depth', 'rgb'] or isinstance(value, (dict, list)) else value
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ndarray_to_list(element) for element in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


final_dic = ndarray_to_list(final_dic)

# import sys
# print(sys.getsizeof(final_dic)/(1024 * 1024 * 1024))
# exit()

# saving the data as json
# with open('data.json', 'w') as f:
#     json.dump(final_dic, f, indent=4)


# zipfile.ZipFile('data.zip',mode='w').write(filename='data.json', arcname=None, compress_type=zipfile.ZIP_DEFLATED)

# from io import BytesIO

# buffer = BytesIO()

# with zipfile.Zipfile(buffer, 'a', zipfile.ZIP_DEFLATED, False) as zipf:
#     json_data = json.dump(final_dic).encode('utf-8')
#     zipf.writestr('data.json', json_data)

# zipped_content = buffer.getvalue()

# with open('data.zip', 'wb') as f:
#     f.write(zipped_content)




def chunk_dict(data, chunk_size):
    """Yield successive chunk_size chunks from the dictionary."""
    
    # Construct the base dictionary without the 'hi' key and its associated list
    base_dict = {k: v for k, v in data.items() if k != 'steps'}
    list_data = data['steps']

    # Iterate over the list in chunks
    for i in range(0, len(list_data), chunk_size):
        chunked_dict = base_dict.copy()  # Copy the base data (without the large list)
        chunked_dict['steps'] = list_data[i:i+chunk_size]
        yield chunked_dict

CHUNK_SIZE=1
# Write the JSON data chunks directly into a zip file
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
with zipfile.ZipFile(f'data_{current_time}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for idx, chunk in enumerate(chunk_dict(final_dic, CHUNK_SIZE)):
        with zipf.open(f'data_chunk_{idx}.json', 'w') as json_file:
            # Convert the chunk to a JSON string and encode it to bytes
            json_data = json.dumps(chunk).encode('utf-8')
            json_file.write(json_data)









'''displaying the depth frame'''

# depth_frame = final_dic["steps"][-1]["d"]

# depth_frame_normalized = ((depth_frame - depth_frame.min()) * (1/(depth_frame.max() - depth_frame.min()) * 255)).astype(np.uint8)

# Convert to PIL Image and show
# image = Image.fromarray(depth_frame_normalized)
# image = Image.fromarray(final_dic["steps"][-1]["seg"])

# image.show()