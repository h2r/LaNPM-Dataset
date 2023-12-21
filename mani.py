from ai2thor.controller import Controller

if __name__ == "__main__":
    controller = Controller(
        massThreshold = 1,
        agentMode="arm",
        #used for the Aryan and George commands
        # scene='FloorPlan_Val2_5',
        # scene='FloorPlan_Val2_1',
        # scene = 'FloorPlan_Train9_4',
        # scene = "FloorPlan_Train5_2",
        # scene = "FloorPlan_Val1_1",
        scene = "FloorPlan_Val3_4",
        snapToGrid=False,
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width= 1280,
        height= 720,
        fieldOfView=60
    )

    while True:            
        controller.interact()#metadata=True, color_frame=True, depth_frame=True, semantic_segmentation_frame=True, instance_segmentation_frame=True)