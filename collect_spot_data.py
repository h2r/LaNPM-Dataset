# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface for graph nav with options to download/upload a map and to navigate a map. """

import argparse
import logging
import math
import os
import sys
from datetime import datetime
import time

import google.protobuf.timestamp_pb2
import graph_nav_util
import grpc

import cv2 

import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.api import geometry_pb2, power_pb2, robot_state_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on, safe_power_off
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME


class GraphNavInterface(object):
    """GraphNav service command line interface."""

    def __init__(self, robot, upload_path, task_name):
        self._robot = robot

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.previous_robot_state = None
        self.previous_gripper_percentage = None
        self.previous_robot_position = [0,0,0]
        self.previous_robot_orientation = [None, None, None]

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

        self._command_dictionary = {
            '1': self._get_localization_state,
            '2': self._set_initial_localization_fiducial,
            '3': self._set_initial_localization_waypoint,
            '4': self._list_graph_waypoint_and_edge_ids,
            '5': self._upload_graph_and_snapshots,
            '8': self._navigate_to_anchor,
            '9': self._clear_graph,
            '0': self._toggle_data_collection
        }

        #additional data for toggling continuous data collection
        self.collect_data = False
        self.pic_hz = 3

        #store the current task name in natural language
        self.task_name = task_name
        self.save_base_path = os.path.join('./Trajectories/{}'.format(self.task_name))
        if not os.path.exists(self.save_base_path):
            os.mkdir(self.save_base_path)

    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        print('Got localization: \n%s' % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print('Got robot state in kinematic odometry frame: \n%s' % str(odom_tform_body))

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)


    def _set_initial_localization_waypoint(self, args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            print("No waypoint specified to initialize to.")
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id)

    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print("Loading the graph from disk into local storage...")
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print("Loaded graph has {} waypoints and {} edges".format(
                len(self._current_graph.waypoints), len(self._current_graph.edges)))
        for waypoint in self._current_gr ts from disk.
            with open(self._upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                      "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(self._upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id),
                      "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print("Uploaded {}".format(waypoint_snapshot.id))
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print("Uploaded {}".format(edge_snapshot.id))

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print("\n")
            print("Upload complete! The robot is currently not localized to the map; please localize", \
                   "the robot using commands (2) or (3) before attempting a navigation command.")

    def _navigate_to_anchor(self, *args):
        """Navigate to a pose in seed frame, using anchors."""
        # The following options are accepted for arguments: [x, y], [x, y, yaw], [x, y, z, yaw],
        # [x, y, z, qw, qx, qy, qz].
        # When a value for z is not specified, we use the current z height.
        # When only yaw is specified, the quaternion is constructed from the yaw.
        # When yaw is not specified, an identity quaternion is used.

        if len(args) < 1 or len(args[0]) not in [2, 3, 4, 7]:
            print("Invalid arguments supplied.")
            return

        seed_T_goal = SE3Pose(float(args[0][0]), float(args[0][1]), 0.0, Quat())

        if len(args[0]) in [4, 7]:
            seed_T_goal.z = float(args[0][2])
        else:
            localization_state = self._graph_nav_client.get_localization_state()
            if not localization_state.localization.waypoint_id:
                print("Robot not localized")
                return
            seed_T_goal.z = localization_state.localization.seed_tform_body.position.z

        if len(args[0]) == 3:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][2]))
        elif len(args[0]) == 4:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][3]))
        elif len(args[0]) == 7:
            seed_T_goal.rot = Quat(w=float(args[0][3]), x=float(args[0][4]), y=float(args[0][5]),
                                   z=float(args[0][6]))

        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return False

        nav_to_cmd_id = None
        # Navigate to the destination.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(
                    seed_T_goal.to_proto(), 1.0, command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                return False
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)
        
        return is_finished

    def _navigate_to(self, args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return False

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return False
       
        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        return is_finished

    def _navigate_route(self, *args):
        """Navigate through a specific route of waypoints."""
        if len(args) < 1 or len(args[0]) < 1:
            # If no waypoint ids are given as input, then return without requesting navigation.
            print("No waypoints provided for navigate route.")
            return
        waypoint_ids = args[0]
        for i in range(len(waypoint_ids)):
            waypoint_ids[i] = graph_nav_util.find_unique_waypoint_id(
                waypoint_ids[i], self._current_graph, self._current_annotation_name_to_wp_id)
            if not waypoint_ids[i]:
                # Failed to find the unique waypoint id.
                return

        edge_ids_list = []
        all_edges_found = True
        # Attempt to find edges in the current graph that match the ordered waypoint pairs.
        # These are necessary to create a valid route.
        for i in range(len(waypoint_ids) - 1):
            start_wp = waypoint_ids[i]
            end_wp = waypoint_ids[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                print("Failed to find an edge between waypoints: ", start_wp, " and ", end_wp)
                print(
                    "List the graph's waypoints and edges to ensure pairs of waypoints has an edge."
                )
                break

        if all_edges_found:
            if not self.toggle_power(should_power_on=True):
                print("Failed to power on the robot, and cannot complete navigate route request.")
                return

            # Navigate a specific route.
            route = self._graph_nav_client.build_route(waypoint_ids, edge_ids_list)
            is_finished = False
            while not is_finished:
                # Issue the route command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route, cmd_duration=1.0)
                time.sleep(.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the route is complete. Then sit
                # the robot down once it is finished.
                is_finished = self._check_success(nav_route_command_id)

            # Power off the robot if appropriate.
            if self._powered_on and not self._started_powered_on:
                # Sit the robot down + power off after the navigation command is complete.
                self.toggle_power(should_power_on=False)

    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def _toggle_data_collection(self, *args):
        """Toggle the data collection boolean"""
        self.collect_data = True if self.collect_data is False else False

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(
                    timeout=10)  # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print("Robot is impaired.")
            return True
        else:
            # Navigation command is not complete yet.
            return False

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None

    def _on_quit(self):
        """Cleanup on quit from the command line interface."""
        # Sit the robot down + power off after the navigation command is complete.
        if self._powered_on and not self._started_powered_on:
            self._robot_command_client.robot_command(RobotCommandBuilder.safe_power_off_command(),
                                                     end_time_secs=time.time())
    
    def is_arm_moving(self, curr_robot_state, linear_threshold=0.1, angular_threshold=0.1):
        """Determine if the robot's arm is moving."""
        # Choose either 'velocity_of_hand_in_vision' or 'velocity_of_hand_in_odom' based on your requirement
        linear_velocity = curr_robot_state.manipulator_state.velocity_of_hand_in_vision.linear
        angular_velocity = curr_robot_state.manipulator_state.velocity_of_hand_in_vision.angular
        

        # Check if the linear or angular velocity exceeds the thresholds
        linear_moving = abs(linear_velocity.x) > linear_threshold or abs(linear_velocity.y) > linear_threshold or abs(linear_velocity.z) > linear_threshold
        angular_moving = abs(angular_velocity.x) > angular_threshold or abs(angular_velocity.y) > angular_threshold or abs(angular_velocity.z) > angular_threshold

        return linear_moving or angular_moving


    def is_gripper_moving(self, curr_robot_state, threshold=0.01):
        
        current_gripper_percentage = curr_robot_state.manipulator_state.gripper_open_percentage

        output = False

        if previous_gripper_open_percentage is None or abs(current_gripper_percentage - self.previous_gripper_percentage) >= threshold:
            
            output = True

        return output

    def is_robot_sitting(self, robot_state):
        """Determine if the robot is in a sitting position."""
        # Placeholder for demonstration
        # Replace with actual logic to determine if the robot is sitting
        vision_position = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map['vision'].parent_tform_child.position
        if vision_position.z >= 0.15:
            return True
        return False

    def is_body_moving(self, robot_curr_position, robot_curr_orientation, distance_threshold=0.1, angle_threshold=0.1):

        moving = False; rotating = False

        if self.previous_robot_position is None or abs(robot_curr_position[0]-self.previous_robot_position[0])>=distance_threshold or abs(robot_curr_position[1]-self.previous_robot_position[1])>=distance_threshold or abs(robot_curr_position[2]-self.previous_robot_position[2])>=distance_threshold:
            moving = True
        if None in self.previous_robot_orientation or abs(robot_curr_orientation[0]-self.previous_robot_orientation[0])>=angle_threshold or abs(robot_curr_orientation[1]-self.previous_robot_orientation[1])>=angle_threshold or abs(robot_curr_orientation[2]-self.previous_robot_orientation[2])>=angle_threshold:
            rotating = True
        
        return moving or rotating

    def collect_images(self):
        camera_sources = ['hand_depth_in_hand_color_frame', 'hand_color_image', 'frontleft_depth', 'frontleft_fisheye_image','frontleft_depth', 'frontright_fisheye_image']

        image_responses = self.image_client.get_image_from_sources(camera_sources)


        #hand depth image: extract and convert from mm to meters (div by 1000)
        hand_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
        hand_depth = hand_depth.reshape(image_responses[0].shot.image.rows,
                                    image_responses[0].shot.image.cols)
        hand_depth = hand_depth / 1000.0

        #if the min and max depth on image are < 12cm (length of gripper) then object is there
        object_held = np.min(hand_depth) <= 0.12 and np.max(hand_depth) <= 0.12
       

        #hand color image
        hand_color_image = cv2.imdecode(np.frombuffer(image_responses[1].show.image.data, dtype=np.uint8), -1)
        hand_color_image = hand_color_image if len(hand_color_image.shape)==3 else cv2.cvtColor(hand_color_image, cv2.COLOR_GRAY2RGB)

        #left fisheye depth
        left_fisheye_depth = np.frombuffer(image_reponses[2].shot.image.data, dtype=np.uint16)
        left_fisheye_depth = left_fisheye_depth.reshape(image_reponses[2].shot.image.rows, image_reponses[2].shot.image.cols)
        left_fisheye_depth = left_fisheye_depth / 1000.0

        #left fisheye image
        left_fisheye_image = cv2.imdecode(np.frombuffer(image_reponses[3].show.image.data, dtype=np.uint8), -1)
        left_fisheye_image = left_fisheye_image if len(left_fisheye_image.shape)==3 else cv2.cvtColor(left_fisheye_image, cv2.COLOR_GRAY2RGB) 

        #right fisheye depth
        right_fisheye_depth = np.frombuffer(image_reponses[4].shot.image.data, dtype=np.uint16)
        right_fisheye_depth = right_fisheye_depth.reshape(image_reponses[4].shot.image.rows, image_reponses[4].shot.image.cols)
        right_fisheye_depth = right_fisheye_depth / 1000.0

        #right fisheye image
        right_fisheye_image = cv2.imdecode(np.frombuffer(image_reponses[5].show.image.data, dtype=np.uint8), -1)
        right_fisheye_image = right_fisheye_image if len(right_fisheye_image.shape)==3 else cv2.cvtColor(right_fisheye_image, cv2.COLOR_GRAY2RGB) 

        return hand_depth, hand_color_image, left_fisheye_depth, left_fisheye_image, right_fisheye_depth, right_fisheye_image, object_held


    def collect_images_and_metadata(self):
        
        curr_robot_state = self._robot_state_client.get_robot_state()

        frame_tree_snapshot = self.robot.get_frame_tree_snapshot()
        body_tform_hand = get_a_tform_b(frame_tree_snapshot, "body", "hand")


        #get all the graphnav related localizations, positions and orientations relative to the seed frame
        graphnav_localization_state = self._graph_nav_client.get_localization_state()


        seed_tform_body = graphnav_localization_state.localization.seed_tform_body
        seed_tform_hand = seed_tform_body * body_tform_hand

        seed_tform_body = bosdyn.client.math_helpers.SE3Pose(seed_tform_body.position.x, seed_tform_body.position.y, seed_tform_body.position.z, seed_tform_body.rotation)

        robot_curr_position = np.array([seed_tform_body.position.x,seed_tform_body.position.y, seed_tform_body.position.z])
        robot_curr_quaternion = np.array([seed_tform_body.rotation.w, seed_tform_body.rotation.x, seed_tform_body.rotation.y, seed_tform_body.rotation.z])
        robot_curr_orientation = np.array([seed_tform_hand.rotation.to_roll(), seed_tform_hand.rotation.to_pitch(), seed_tform_hand.rotation.to_yaw()])

        arm_curr_position_rel_body = np.array([body_tform_hand.position.x, body_tform_hand.position.y, body_tform_hand.position.z])
        arm_curr_quaternion_rel_body = np.array([body_tform_hand.rotation.w, body_tform_hand.rotation.x, body_tform_hand.rotation.y, body_tform_hand.rotation.z])
        arm_curr_orientation_rel_body = np.array([body_tform_hand.rotation.to_roll(), body_tform_hand.rotation.to_pitch(), body_tform_hand.rotation.to_yaw()])

        arm_curr_position_rel_seed = np.array([seed_tform_hand.position.x, body_tform_hand.y, body_tform_hand.z])
        arm_curr_quaternion_rel_seed = np.array([seed_tform_hand.rotation.w, seed_tform_hand.rotation.x, seed_tform_rotation.y, seed_tform_rotation.z])
        arm_curr_orientation_rel_seed = np.array([seed_tform_hand.rotation.to_roll(), seed_tform_hand.rotation.to_pitch(), seed_tform_hand.rotation.to_yaw()])

        #additional state variables e.g. stowing and grasping
        current_gripper_percentage = curr_robot_state.manipulator_state.gripper_open_percentage
        current_stow_state = curr_robot_state.manipulator_state.stow_state

        #TODO: figure out how to make this in the body frame
        joint_states = {x.name: x.position for x in curr_robot_state.kinematic_state.joint_states}
        curr_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]


        #collect and save all the necessary images + store the image paths
        hand_depth, hand_color_image, left_fisheye_depth, left_fisheye_image, right_fisheye_depth, right_fisheye_image, object_held = self.collect_images()
        
        curr_save_path = os.path.join(self.save_base_path, curr_time)
        os.mkdir(curr_save_path)
        
        pickle.dump(hand_depth, open(os.path.join(curr_save_path, 'hand_depth'), "wb"))
        pickle.dump(left_fisheye_depth, open(os.path.join(curr_save_path, 'left_fisheye_depth'), "wb"))
        pickle.dump(right_fisheye_depth, open(os.path.join(curr_save_path, 'right_fisheye_depth'), "wb"))
        cv2.imwrite(os.path.join(curr_save_path, 'hand_color_image.jpeg'), hand_color_image)
        cv2.imwrite(os.path.join(curr_save_path, 'left_fisheye_image.jpeg'), left_fisheye_image)
        cv2.imwrite(os.path.join(curr_save_path, 'right_fisheye_image.jpeg'), right_fisheye_image)

        #Step 1: check robot is not sitting down and that it's gripper/arm/body are moving
        if self.is_arm_moving(curr_robot_state) or self.is_gripper_moving(curr_robot_state) or self.is_robot_sitting(curr_robot_state) or self.is_body_moving(robot_curr_position, robot_curr_orientation):

            #Step 2: if all conditions are met, capture the current data including
            '''
            Language command (given)
            Front RGB images
            Front Depth images
            Gripper RGB images
            Wall clock time
            Gripper RGB images
            Body state (x, y, z, roll, pitch, yaw)
            Arm state (x, y, z, roll, pitch, yaw)
            Body quaternion
            Arm quaternion
            Arm stow state (True/False)
            Gripper open percentage
            Current held object(s)
            Front instance segmentation
            Gripper instance segmentation
            '''

            current_data = {'language_command': self.task_name,
                            'left_fisheye_rgb': os.path.join(curr_save_path, 'left_fisheye_image.jpeg'),
                            'left_fisheye_depth': os.path.join(curr_save_path, 'left_fisheye_depth'),
                            'right_fisheye_rgb': os.path.join(curr_save_path, 'right_fisheye_image.jpeg'),
                            'right_fisheye_depth': os.path.join(curr_save_path, 'right_fisheye_depth'),
                            'gripper_rgb': os.path.join(curr_save_path, 'hand_color_image.jpeg'),
                            'gripper_depth': os.path.join(curr_save_path, 'hand_depth'),
                            'wall_clock_time': curr_time,
                            'body_state': {'x': robot_curr_position[0], 'y':robot_curr_position[1], 'z': robot_curr_position[2]},
                            'body_quaternion': {'w': robot_curr_quaternion[0], 'x': robot_curr_quaternion[1], 'y':robot_curr_quaternion[2], 'z':robot_curr_quaternion[3]},
                            'body_orientation': {'r': robot_curr_orientation[0], 'p': robot_curr_orientation[1], 'y': robot_curr_orientation[2]},
                            'arm_state_rel_body': {'x': arm_curr_position_rel_body[0], 'y': arm_curr_position_rel_body[1], 'z': arm_curr_position_rel_body[2]},
                            'arm_quaternion_rel_body': {'w': arm_curr_quaternion_rel_body[0], 'x': arm_curr_quaternion_rel_body[1], 'y': arm_curr_quaternion_rel_body[2], 'z': arm_curr_quaternion_rel_body[2]},
                            'arm_orientation_rel_body': {'x': arm_curr_orientation_rel_body[0], 'y': arm_curr_orientation_rel_body[1], 'z': arm_curr_orientation_rel_body[2]},
                            'arm_state_global': {'x': arm_curr_orientation_rel_seed[0], 'y': arm_curr_orientation_rel_seed[1], 'z': arm_curr_orientation_rel_seed[2]},
                            'arm_quaternion_global': {'w': arm_curr_quaternion_rel_seed[0], 'x':arm_curr_orientation_rel_seed[1], 'y': arm_curr_orientation_rel_seed[2], 'z':arm_curr_orientation_rel_seed[3]},
                            'arm_orientation_global': {'x':arm_curr_quaternion_rel_seed[0], 'y':arm_curr_quaternion_rel_seed[1], 'z':arm_curr_quaternion_rel_seed[2]},
                            'joint_positions': joint_states,
                            'arm_stowed': current_stow_state,
                            'gripper_open_percentage': current_gripper_percentage,
                            'object_held': object_held,
                            'left_fisheye_instance_seg': os.path.join(curr_save_path, 'left_fisheye_instance_seg.jpeg'),
                            'right_fisheye_instance_seg': os.path.join(curr_save_path, 'right_fisheye_instance_seg.jpeg'),
                            'gripper_fisheye_instance_seg': os.path.join(curr_save_path, 'hand_color_image_instance_seg.jpeg'),
                            }

        self.previous_gripper_percentage = current_gripper_percentage
        self.previous_robot_state = curr_robot_state
        self.previous_robot_position = robot_curr_position
        self.previous_robot_orientation = robot_curr_orientation
        

    def run(self):
        """Main loop for the command line interface."""

        self._upload_graph_and_snapshots()

        #localize graphnav with initial fiducial and list out waypoints with edges
        self._set_initial_localization_fiducial()
        self._list_graph_waypoint_and_edge_ids()

        

        while True:
            print("""
            Options:
            (1) Get localization state.
            (2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
            (3) Initialize localization to a specific waypoint (must be exactly at the waypoint.
            (4) List the waypoint ids and edge ids of the map on the robot.
            (5) Upload the graph and its snapshots.
            (6) Navigate to. The destination waypoint id is the second argument.
            (7) Navigate route. The (in-order) waypoint ids of the route are the arguments.
            (8) Navigate to in seed frame. The following options are accepted for arguments: [x, y],
                [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz]. (Don't type the braces).
                When a value for z is not specified, we use the current z height.
                When only yaw is specified, the quaternion is constructed from the yaw.
                When yaw is not specified, an identity quaternion is used.
            (9) Clear the current graph.
            (0) Toggle data collection mode.
            (q) Exit.
            """)
            try:
                inputs = input('>')
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == 'q':
                self._on_quit()
                break

            elif self.collect_data is True:
                self.collect_images_and_metadata()

            if req_type not in self._command_dictionary:
                print("Request not in the known command dictionary.")
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)

            # 3Hz frequency for data collection: capture sample every 1/3 seconds
            time.sleep(1/float(self.pic_hz))


def main(argv):

    HOSTNAME = ''
    USER = 'user'
    PASS = 'bigbubbabigbubba'

    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-u', '--upload-filepath',
                        help='Full filepath to graph and snapshots to be uploaded.', required=True)
    parser.add_argument('-t', '--task', help='Task name in language', required=True)
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    # Setup and authenticate the robot.
    sdk = bosdyn.client.create_standard_sdk('GraphNavClient')
    robot = sdk.create_robot(HOSTNAME)
    robot.authenticate(USER, PASS)

    graph_nav_command_line = GraphNavInterface(robot, options.upload_filepath, options.task)
    
    try:
        graph_nav_command_line.run()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(exc)
        print("Graph nav command line client threw an error.")
        return False



if __name__ == '__main__':
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.
