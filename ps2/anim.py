import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Optional
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from environment import ManipulatorEnv, State

def animate_plan(
    env: ManipulatorEnv, 
    plan: List[State], 
    video_output_file: Optional[str] = "solve_4R.mp4",
    fps: int = 10,  # Added fps parameter with a default value
    frame_duplication: int = 1  # Optional: Duplicate frames to slow down
):
    """
    Visualizes the plan with pyplot and, optionally, saves it to the video file.

    :param env: Manipulator environment
    :param plan: Plan - sequence of states
    :param video_output_file: If not None, saves animation to this file. Suggested extension is .mp4.
    :param fps: Frames per second for the output video. Lower values slow down the animation.
    :param frame_duplication: Number of times each frame is duplicated to slow down the animation.
    """
    print("Starting animation...")
    # Use the current working directory instead of __file__
    script_dir = Path.cwd()  # Alternatively, use os.getcwd()
    video_output_file = script_dir / video_output_file

    # Setup matplotlib figure
    fig, ax = plt.subplots()
    ax.set_xlim([-4, 4])
    ax.set_ylim([-2.5, 2.5])
    fig.set_size_inches(8, 6)

    # Initialize video writer
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
        canvas.get_width_height()[::-1] + (3,)
    )
    frame_height, frame_width, _ = mat.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_output_file), fourcc, fps, (frame_width, frame_height))

    for i, state in enumerate(plan):
        env.state = state
        env.render(plt_show=False)

        # Render the frame
        canvas.draw()
        mat = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
            canvas.get_width_height()[::-1] + (3,)
        )
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

        # Duplicate frames if needed
        for _ in range(frame_duplication):
            video_writer.write(mat)

        ax.clear()  # Clear only the axes, keep figure
        ax.set_xlim([-4, 4])
        ax.set_ylim([-2.5, 2.5])

    # Release resources
    video_writer.release()
    plt.close(fig)
    print(f"Animation saved to {video_output_file}")
