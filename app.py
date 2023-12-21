from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulator')
def simulator():
    # Launch the AI2-THOR simulator locally here
    # You may need to start the simulator as a subprocess or in a separate thread

    # For demonstration purposes, this route will just return a placeholder

    from ai2thor.controller import Controller

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

    # while True:            
    #     controller.interact()#metadata=True, color_frame=True, depth_frame=True, semantic_segmentation_frame=True, instance_segmentation_frame=True)

    return controller.interact()

if __name__ == '__main__':
    app.run(debug=True)
