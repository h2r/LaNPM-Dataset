import bosdyn.client
import json
import time
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from get_image import capture


IP = '138.16.161.24'
USER = 'user'
PASS = "bigbubbabigbubba"

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



def collect_data(robot, robot_state):
    """Collect and return required data."""
    front_image_data = capture(robot, "PIXEL_FORMAT_RGB_U8")
    front_depth_data = capture(robot, "PIXEL_FORMAT_DEPTH_U16")
    front_depth_color_data = capture(robot)
    gripper_image_data = capture(robot, mode="arm")

    body_state_odom = robot_state.kinematic_state.velocity_of_body_in_odom
    body_state_vision = robot_state.kinematic_state.velocity_of_body_in_vision
    arm_state_odom = robot_state.manipulator_state.velocity_of_hand_in_odom
    arm_state_vision = robot_state.manipulator_state.velocity_of_hand_in_vision
    gripper_open_percent = robot_state.manipulator_state.gripper_open_percentage
    stow_state = robot_state.manipulator_state.stow_state

    return {
        "images": {
            # "front_rgb_camera": list(front_image_data),
            # "front_depth_camera": list(front_depth_data),
            # "front_depth_color_camera": list(front_depth_color_data),
            # "gripper_camera": list(gripper_image_data)
        },
        "body_state_odom": {
            "linear": {
                "x": body_state_odom.linear.x,
                "y": body_state_odom.linear.y,
                "z": body_state_odom.linear.z
            },
            "angular": {
                "x": body_state_odom.angular.x,
                "y": body_state_odom.angular.y,
                "z": body_state_odom.angular.z
            }
        },
        "body_state_vision": {
            "linear": {
                "x": body_state_vision.linear.x,
                "y": body_state_vision.linear.y,
                "z": body_state_vision.linear.z
            },
            "angular": {
                "x": body_state_vision.angular.x,
                "y": body_state_vision.angular.y,
                "z": body_state_vision.angular.z
            }
        },
        "arm_state_odom": {
            "linear": {
                "x": arm_state_odom.linear.x,
                "y": arm_state_odom.linear.y,
                "z": arm_state_odom.linear.z
            },
            "angular": {
                "x": arm_state_odom.angular.x,
                "y": arm_state_odom.angular.y,
                "z": arm_state_odom.angular.z
            }
        },
        "arm_state_vision": {
            "linear": {
                "x": arm_state_vision.linear.x,
                "y": arm_state_vision.linear.y,
                "z": arm_state_vision.linear.z
            },
            "angular": {
                "x": arm_state_vision.angular.x,
                "y": arm_state_vision.angular.y,
                "z": arm_state_vision.angular.z
            }
        },
        "gripper_open_percent": gripper_open_percent,
        "stow_state": stow_state
    }

def is_robot_sitting(robot_state):
    """Determine if the robot is in a sitting position."""
    # Placeholder for demonstration
    # Replace with actual logic to determine if the robot is sitting
    vision_position = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map['vision'].parent_tform_child.position
    if vision_position.z >= 0.15:
        return True
    return False


def main():
    # Create robot object and authenticate
    sdk = bosdyn.client.create_standard_sdk('SpotRobotClient')
    robot = sdk.create_robot(IP)
    robot.authenticate(USER, PASS)


    # Create state, image, and manipulation clients
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    data_sequence = []

    while True:
        robot_state = state_client.get_robot_state()
        # collected_data = collect_data(robot, robot_state)
        
        if is_robot_sitting(robot_state):
            with open('spot_data2.json', 'w') as file:
                json.dump(data_sequence, file, indent=4)
                print('Data saved!')
            break
       
        if is_moving(robot_state) or is_arm_moving(robot_state.manipulator_state) or is_gripper_moving(robot_state.manipulator_state):
            print('moving')
            collected_data = collect_data(robot, robot_state)
            data_sequence.append(collected_data)
        else:
            print('not moving')

        time.sleep(0.1)

if __name__ == '__main__':
    main()