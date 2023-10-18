from ai2thor.controller import Controller

if __name__ == "__main__":
    controller = Controller(
        massThreshold = 1,
        agentMode="arm",
        # agentMode="default",
        # scene="FloorPlan_Train3_1",
        scene = "FloorPlan_Train1_4",
        snapToGrid=False,
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width= 1280,
        height= 720,
        fieldOfView=60
    )

    while True:            
        controller.interact(metadata=True, color_frame=True, depth_frame=True, semantic_segmentation_frame=True, instance_segmentation_frame=True)