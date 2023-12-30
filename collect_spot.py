import bosdyn.client
import json
import time
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from get_image import capture


previous_gripper_open_percentage = None

def is_moving(robot_state, threshold=0.05):
    """Determine if the robot is moving in any direction or rotating."""
    # velocity = robot_state.kinematic_state.velocity.vel
    linear_velocity = robot_state.kinematic_state.velocity_of_body_in_vision.linear
    angular_velocity = robot_state.kinematic_state.velocity_of_body_in_vision.angular

    return (abs(linear_velocity.x) > threshold or
            abs(linear_velocity.y) > threshold or
            abs(linear_velocity.z) > threshold or
            abs(angular_velocity.x) > threshold or
            abs(angular_velocity.y) > threshold or
            abs(angular_velocity.z) > threshold)

def is_arm_moving(manipulator_state, linear_threshold=0.1, angular_threshold=0.1):
    """Determine if the robot's arm is moving."""
    # Choose either 'velocity_of_hand_in_vision' or 'velocity_of_hand_in_odom' based on your requirement
    linear_velocity = manipulator_state.velocity_of_hand_in_vision.linear
    angular_velocity = manipulator_state.velocity_of_hand_in_vision.angular
    # print(angular_velocity)
    # print(linear_velocity)

    # Check if the linear or angular velocity exceeds the thresholds
    linear_moving = abs(linear_velocity.x) > linear_threshold or abs(linear_velocity.y) > linear_threshold or abs(linear_velocity.z) > linear_threshold
    angular_moving = abs(angular_velocity.x) > angular_threshold or abs(angular_velocity.y) > angular_threshold or abs(angular_velocity.z) > angular_threshold

    return linear_moving or angular_moving


def is_gripper_moving(manipulator_state, threshold=0.01):
    global previous_gripper_open_percentage
    current_percentage = manipulator_state.gripper_open_percentage

    if previous_gripper_open_percentage is None:
        previous_gripper_open_percentage = current_percentage
        return False

    if abs(current_percentage - previous_gripper_open_percentage) > threshold:
        previous_gripper_open_percentage = current_percentage
        return True

    previous_gripper_open_percentage = current_percentage
    return False



def collect_data(image_client, manipulation_client):
    """Collect and return required data."""
    # Placeholder for collecting image data from front and gripper cameras
    front_image_data = "front_camera_image_data"  # Replace with actual image capture logic
    gripper_image_data = "gripper_camera_image_data"  # Replace with actual image capture logic

    # Placeholder for collecting arm and gripper state
    arm_state_data = "arm_state_data"  # Replace with actual arm state capture logic
    gripper_state_data = "gripper_state_data"  # Replace with actual gripper state capture logic

    return {
        "images": {
            "front_camera": front_image_data,
            "gripper_camera": gripper_image_data
        },
        "arm_state": arm_state_data,
        "gripper_state": gripper_state_data
    }

def is_robot_sitting(robot_state):
    """Determine if the robot is in a sitting position."""
    # Placeholder for demonstration
    # Replace with actual logic to determine if the robot is sitting
    # odom_position = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map['odom'].parent_tform_child.position
    vision_position = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map['vision'].parent_tform_child.position
    if vision_position.z >= 0.15:
        return True
    return False


def get_image(sdk, robot):
    import cv2
    import numpy as np
    import bosdyn.client
    from bosdyn.client.image import ImageClient
    from bosdyn.api import image_pb2
    from bosdyn.client.image import ImageClient, build_image_request


    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(options.image_service)
    requests = [
        build_image_request(source, quality_percent=options.jpeg_quality_percent,
                            resize_ratio=options.resize_ratio) for source in options.image_sources
    ]

    for image_source in options.image_sources:
        cv2.namedWindow(image_source, cv2.WINDOW_NORMAL)
        if len(options.image_sources) > 1 or options.disable_full_screen:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        else:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # # Create image client
    # image_client = robot.ensure_client(ImageClient.default_service_name)

    # # Define camera name
    # camera_name = 'frontright_fisheye_image'

    # image_request = [image_pb2.ImageRequest(image_source_name=camera_name, 
    #                                     pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8)]
    # response = image_client.get_image(image_request)

    # if response:
    #     image_data = response[0].shot.image.data
    #     nparr = np.frombuffer(image_data, np.uint8)
    #     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #     # Correct the orientation by rotating the image
    #     # rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    #     cv2.imwrite('front_right_camera_image.jpg', image)

def main():
    # Create robot object and authenticate
    sdk = bosdyn.client.create_standard_sdk('SpotRobotClient')
    robot = sdk.create_robot('138.16.161.24')
    robot.authenticate('user', 'bigbubbabigbubba')

    # image_client = robot.ensure_client(ImageClient.default_service_name)
    # camera_sources = image_client.list_image_sources()
    # print("Available camera sources:", camera_sources)


    # Create state, image, and manipulation clients
    state_client = robot.ensure_client(RobotStateClient.default_service_name)
    # image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    data_sequence = []

    while True:
        robot_state = state_client.get_robot_state()
        manipulator_state = robot_state.manipulator_state
       
        # arm_state = manipulation_client.get_arm_state()
        # print(manipulation_client.manipulation_api_feedback_command)

        # if is_robot_sitting(robot_state):
        #     with open('spot_data.json', 'w') as file:
        #         json.dump(data_sequence, file, indent=4)
        #     break

        # get_image(sdk, robot)
        # capture(robot, "PIXEL_FORMAT_RGB_U8")
        capture(robot, "PIXEL_FORMAT_DEPTH_U16")
        print('done')
        exit()
        if is_moving(robot_state) or is_arm_moving(manipulator_state) or is_gripper_moving(manipulator_state):
            print('moving')
            # collected_data = collect_data(image_client, manipulation_client)
            # data_sequence.append(collected_data)
        else:
            print('not moving')

        time.sleep(0.1)

if __name__ == '__main__':
    main()