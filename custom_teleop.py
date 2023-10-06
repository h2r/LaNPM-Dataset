
def main():
    stdscr.timeout(100)  # Wait 100 ms for input
    stdscr.keypad(True)  # Enable keypad mode to capture special keys


    i = 0
    x = 0
    y = 0
    z = 0
    incr = 0.05

    while True:
        # user_input = stdscr.getch()
        
        # if user_input != -1:  # -1 means no input
        #     if user_input == ord('q'):
        #         break
        #     elif user_input == ord('d'):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("w"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             ahead = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("a"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right=-0.25,
        #             returnToStart=False,
        #         controller.step(
        #             action="SetHandSphereRadius",
        #             radius=0.1
        #         )
        #         event = controller.step(action="ReleaseObject")
        #         gather_data(event)
        #     elif user_input == 27:
        #         break
#             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("s"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             ahead=-0.25,
        #             returnToStart=False,
        #             speed=1,.0.
        #         event = controller.step(
        #             action="RotateAgent",
        #             degrees=-30,
        #             returnToStart=True,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #        #             action="SetHandSphereRadius",
        #             radius=0.1
        #         )
        #         event = controller.step(action="ReleaseObject")
        #         gather_data(event)
        #     elif user_input == 27:
        #         break
       action="SetHandSphereRadius",
        #             radius=0.1
        #         )while True:
        # user_input = stdscr.getch()
        
        event = controller.interact()
ookDown")
        #         gather_data(event)
        #     elif user_input == curses.KEY_UP:
        #         i+=incr
        #         event = controller.step(
        #             action="MoveArmBase",
        #             y=i,
        #             speed=1,
        #             returnToStart=False,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == curses.KEY_DOWN:
        #         i-=incr
        #         event = controller.step(
        #             action="MoveArmBase",
        #             y=i,
        #             speed=1,
        #             returnToStart=Fwhile True:
        # user_input = stdscr.getch()
        
        event = controller.interact# user_input = stdscr.getch()
        
        # if user_input != -1:  # -1 means no input
        #     if user_input == ord('q'):
        #         break
        #     elif user_input == ord('d'):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("w"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             ahead = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("a"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right=-0.25,
        #             returnToStart=False,
        #         controller.step(
        #             action="SetHandSphereRadius",
        #             radius=0.1
        #         )
        #         event = controller.step(action="ReleaseObject")
        #         gather_data(event)
        #     elif user_input == m",
        #             position=dict(x=x, y=y, z=z),
        #             coordinateSpace="wrist",
        #  #             action="SetHandSphereRadius",
        #             radius=0.1
        #         )
        #         event = controller.step(action="ReleaseObject")
        #         gather_data(event)
        #     elif user_input == 27:
        #         break
     elif user_input == ord('8'):
        #         y += incr
        #         event = controller.step(
        #             action="MoveArm",
        #             position=dict(x=x, y=y, z=z),
        #             coordinateSpace="wrist",
        #             restrictMovement=False,
        #             speed=1,
        #             returnToStart=True,
        #             fixedDeltaTime# user_input = stdscr.getch()
        
        # if user_input != -1:  # -1 means no input
        #     if user_input == ord('q'):
        #         break
        #     elif user_input == ord('d'):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("w"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             ahead = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("a"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right=-0.25,
        #             returnToStart=# user_input = stdscr.getch()
        
        # if user_input != -1:  # -1 means no input
        #     if user_input == ord('q'):
        #         break
        #     elif user_input == ord('d'):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("w"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             ahead = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("a"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right=-0.25,
        #             returnToStart=False,
        #         controller.step(
        #             action="SetHandSphereRadius",
        #             radius=0.1
        #         )
        #         event = controller.step(action="ReleaseObject")
        #         gather_data(event)
        #     elif user_input == 
        #             action="SetHandSphereRadius",
        #             radius=0.1
        #         )
        #         event = controller.step(action="ReleaseObject")
        #         gather_data(event)
        #     elif user_input == 27:
        #         break
#             position=dict(x=x, y=y, z=z),
        #             coordinateSpace="wrist",
        #             action="SetHandSphereRadius",
        #             radius=0.1
        #         )
        #         event = controller.step(action="ReleaseObject")
        #         gather_data(event)
        #     elif user_input == 27:
        #         break
  #             restrictMovement=False,
        #             speed=1,
        #             returnToStart=True,
        #             fixedDeltaTime=0.02
        #         )# user_input = stdscr.getch()
        
        # if user_input != -1:  # -1 means no input
        #     if user_input == ord('q'):
        #         break
        #     elif user_input == ord('d'):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("w"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             ahead = 0.25,
        #             returnToStart=False,
        #             speed=1,
        #             fixedDeltaTime=0.02
        #         )
        #         gather_data(event)
        #     elif user_input == ord("a"):
        #         event = controller.step(
        #             action="MoveAgent",
        #             right=-0.25,
        #             returnToStart=False,
        #         controller.step(
        #             action="SetHandSphereRadius",
        #             radius=0.1
        #         )
        #         event = controller.step(action="ReleaseObject")
        #         gather_data(event)
        #     elif user_input == 
        #         event = controller.step(action="PickupObject")
        #         gather_data(event)
        #     elif user_input == ord('r'):
        #         controller.step(
        #             action="SetHandSphereRadius",
        #             radius=0.1
        #         )
        #         event = controller.step(action="ReleaseObject")
        #         gather_data(event)
        #     elif user_input == 27:
        #         break


curses.wrapper(main)
