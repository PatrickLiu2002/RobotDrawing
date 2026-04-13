import math
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from svgpathtools import svg2paths, Line
from scipy.spatial.transform import Rotation
from franky import (
    Robot, 
    Affine, 
    CartesianMotion, 
    CartesianWaypointMotion, 
    CartesianWaypoint, 
    ReferenceType,
    ControllerMode,
    JointMotion
)

class RobotLogger:
    def __init__(self, robot, log_rate=0.02):
        self.robot = robot
        self.log_rate = log_rate # 50Hz logging by default
        self.is_logging = False
        self.timestamps = []
        self.joint_torques = [] 
        self.z_forces = []      
        self.x_positions = []
        self.y_positions = []
        self.thread = None

    def _log_loop(self):
        start_time = time.time()
        while self.is_logging:
            try:
                state = self.robot.state
                self.timestamps.append(time.time() - start_time)
                self.joint_torques.append(list(state.tau_J))
                self.z_forces.append(state.O_F_ext_hat_K[2]) 
                
                # Capture actual Cartesian X/Y position
                actual_pose = state.O_T_EE 
                self.x_positions.append(actual_pose.translation[0])
                self.y_positions.append(actual_pose.translation[1])
            except Exception as e:
                pass # Fail silently so we don't spam the console during a move
            time.sleep(self.log_rate)

    def start(self):
        print("Starting data logger...")
        self.is_logging = True
        self.thread = threading.Thread(target=self._log_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_logging = False
        if self.thread:
            self.thread.join()
        print(f"Logging stopped. Captured {len(self.timestamps)} data points.")

def save_telemetry_video(logger, filename="robot_telemetry.mp4"):
    if not logger.timestamps:
        print("No data logged, skipping video creation.")
        return

    print("Generating telemetry video. This might take a minute...")
    
    t = np.array(logger.timestamps)
    torques = np.array(logger.joint_torques) 
    forces = np.array(logger.z_forces)       
    x_pos = np.array(logger.x_positions)
    y_pos = np.array(logger.y_positions)

    # Create 3 subplots: Torques, Forces, X/Y Path
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Setup Torques Axis
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(np.min(torques) - 5, np.max(torques) + 5)
    ax1.set_title("Joint Torques over Time")
    ax1.set_ylabel("Torque (Nm)")
    
    # Setup Z-Force Axis
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(np.min(forces) - 2, np.max(forces) + 2)
    ax2.set_title("End Effector Z-Force (Pen Pressure)")
    ax2.set_ylabel("Force (N)")
    ax2.set_xlabel("Time (s)")

    # Setup X/Y Path Axis
    ax3.set_xlim(np.min(x_pos) - 0.05, np.max(x_pos) + 0.05)
    ax3.set_ylim(np.min(y_pos) - 0.05, np.max(y_pos) + 0.05)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title("End Effector Cartesian Path (Top-Down)")
    ax3.set_xlabel("X Position (m)")
    ax3.set_ylabel("Y Position (m)")
    ax3.invert_yaxis() # Match standard SVG/drawing orientation

    # Initialize lines
    torque_lines = [ax1.plot([], [], label=f'Joint {i+1}')[0] for i in range(7)]
    force_line, = ax2.plot([], [], color='red', label='Z-Force')
    path_line, = ax3.plot([], [], color='blue', linewidth=1.5, label='Trace')
    pen_head, = ax3.plot([], [], marker='o', color='red', markersize=6) # Current position dot
    
    ax1.legend(loc='upper right', fontsize='small')
    ax2.legend(loc='upper right', fontsize='small')
    fig.tight_layout()

    def update(frame):
        current_t = t[:frame]
        
        for i in range(7):
            torque_lines[i].set_data(current_t, torques[:frame, i])
            
        force_line.set_data(current_t, forces[:frame])
        
        # Draw path up to current frame, and dot at exact current frame
        path_line.set_data(x_pos[:frame], y_pos[:frame])
        if frame > 0:
            pen_head.set_data([x_pos[frame-1]], [y_pos[frame-1]])
            
        return torque_lines + [force_line, path_line, pen_head]

    frames = len(t)
    ani = animation.FuncAnimation(fig, update, frames=range(0, frames, 2), blit=True)

    writer = animation.FFMpegWriter(fps=int(1/(logger.log_rate * 2)), metadata=dict(artist='FrankyPlotter'), bitrate=2000)
    ani.save(filename, writer=writer)
    plt.close(fig)
    print(f"Video saved successfully to {filename}!")


def generate_2d_strokes(svg_file, center_x=0.5, center_y=0.0, side_length=0.3, step_size=0.004):
    """Parses an SVG and groups the X/Y coordinates into discrete, unbroken strokes."""
    paths, attributes = svg2paths(svg_file)
    
    xmin, xmax, ymin, ymax = float('inf'), float('-inf'), float('inf'), float('-inf')
    for path in paths:
        for seg in path:
            x0, x1, y0, y1 = seg.bbox()
            xmin, xmax = min(xmin, x0), max(xmax, x1)
            ymin, ymax = min(ymin, y0), max(ymax, y1)
            
    svg_width = xmax - xmin
    svg_height = ymax - ymin
    scale = side_length / max(svg_width, svg_height) if max(svg_width, svg_height) > 0 else 1.0
    
    offset_x = center_x - (svg_width * scale) / 2.0
    offset_y = center_y - (svg_height * scale) / 2.0

    strokes = []
    current_stroke = []
    last_point = None

    for path in paths:
        for seg in path:
            nx_start = (seg.start.real - xmin) * scale
            ny_start = (seg.start.imag - ymin) * scale
            start_x = offset_x + nx_start
            start_y = offset_y - ny_start + (svg_height * scale)

            if last_point is None or (abs(seg.start - last_point) * scale) > 0.005:
                if current_stroke:
                    strokes.append(current_stroke)
                    current_stroke = []
                current_stroke.append((start_x, start_y))

            if isinstance(seg, Line):
                nx_end = (seg.end.real - xmin) * scale
                ny_end = (seg.end.imag - ymin) * scale
                end_x = offset_x + nx_end
                end_y = offset_y - ny_end + (svg_height * scale)
                
                if math.dist(current_stroke[-1], (end_x, end_y)) > 0.003:
                    current_stroke.append((end_x, end_y))
            else:
                seg_length = seg.length() * scale 
                num_steps = max(1, int(math.ceil(seg_length / step_size)))
                
                for i in range(1, num_steps + 1):
                    t = i / num_steps
                    point_complex = seg.point(t)
                    
                    nx = (point_complex.real - xmin) * scale
                    ny = (point_complex.imag - ymin) * scale
                    pt_x = offset_x + nx
                    pt_y = offset_y - ny + (svg_height * scale)
                    
                    if math.dist(current_stroke[-1], (pt_x, pt_y)) >= 0.003:
                        current_stroke.append((pt_x, pt_y))
            
            last_point = seg.end

    if current_stroke:
        strokes.append(current_stroke)

    return strokes


def execute_strokes(robot_ip, strokes, z_draw, z_transit, dynamics_factor=0.05):
    print(f"Connecting to Franka robot at {robot_ip}...")
    robot = Robot(robot_ip)
    robot.recover_from_errors()
    
    robot.relative_dynamics_factor = dynamics_factor 
    robot.set_cartesian_impedance([3000.0, 3000.0, 50.0, 200.0, 200.0, 200.0]) 
    
    t_limit = [100.0] * 7
    f_limit = [100.0] * 6
    robot.set_collision_behavior(t_limit, t_limit, f_limit, f_limit)
    
    quat = Rotation.from_euler("xyz", [math.pi, 0.0, 0.0]).as_quat()
    robot.controller_mode = ControllerMode.CartesianImpedance
    
    if not strokes:
        return
        
    # --- START LOGGING HERE ---
    logger = RobotLogger(robot, log_rate=0.02)
    logger.start()

    print(f"Executing {len(strokes)} separate strokes...")

    idx = 0
    while idx < len(strokes):
        # We work on a mutable copy of the stroke so we can slice it upon failure
        stroke_remainder = strokes[idx].copy()
        shift_x, shift_y = 0.0, 0.0 # Track our singularity-avoidance shift
        
        retries = 0
        success = False
        
        while retries < 3 and not success:
            try:
                # Apply current shifts to the start point
                start_x = stroke_remainder[0][0] + shift_x
                start_y = stroke_remainder[0][1] + shift_y
                
                # 1. Hover above the start of the stroke
                hover_pose = Affine([start_x, start_y, z_transit], quat)
                robot.move(CartesianMotion(hover_pose, ReferenceType.Absolute))
                time.sleep(0.05)
                
                # 2. Plunge down to the paper
                drop_pose = Affine([start_x, start_y, z_draw], quat)
                robot.move(CartesianMotion(drop_pose, ReferenceType.Absolute))
                
                # 3. Wait for contact, then stabilize
                t0 = time.time()
                while True:
                    current_force = abs(robot.state.O_F_ext_hat_K[2])
                    if current_force >= 0.5: 
                        time.sleep(0.1) 
                        break
                    if time.time() - t0 > 2.0:
                        print(f"Stroke {idx+1}: Contact timeout, drawing anyway!")
                        break
                    time.sleep(0.01)
                    
                # 4. Waypoint Drawing Path 
                if len(stroke_remainder) > 1:
                    stroke_waypoints = []
                    for x, y in stroke_remainder[1:]: 
                        pose = Affine([x + shift_x, y + shift_y, z_draw], quat)
                        stroke_waypoints.append(CartesianWaypoint(pose, ReferenceType.Absolute))
                        
                    robot.move(CartesianWaypointMotion(stroke_waypoints))
                
                # 5. Lift straight back up
                end_x = stroke_remainder[-1][0] + shift_x
                end_y = stroke_remainder[-1][1] + shift_y
                lift_pose = Affine([end_x, end_y, z_transit], quat)
                robot.move(CartesianMotion(lift_pose, ReferenceType.Absolute))
                
                # 6. Spring Settling Time
                time.sleep(0.3) 
                
                success = True
                idx += 1
                
            except Exception as e:
                # CRITICAL: Capture failure coordinates BEFORE recovering/lifting
                try:
                    curr_pose = robot.state.O_T_EE
                    fail_x, fail_y = curr_pose.translation[0], curr_pose.translation[1]
                except:
                    fail_x, fail_y = None, None

                retries += 1
                print(f"⚠️ Stroke {idx+1} failed (Attempt {retries}): {e}")
                robot.recover_from_errors() 
                
                # 1. Safety Lift
                try:
                    curr_pose = robot.state.O_T_EE_d
                    curr_trans = curr_pose.translation
                    robot.move(CartesianMotion(Affine([curr_trans[0], curr_trans[1], z_transit], curr_pose.quaternion), ReferenceType.Absolute))
                except:
                    pass 
                
                # 2. Re-home to reset joint configurations
                try:
                    print("Re-homing to default joint posture...")
                    robot.relative_dynamics_factor = 0.05 
                    default_joints = [0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4]
                    robot.move(JointMotion(default_joints))
                except Exception as ex:
                    print(f"Failed to re-home: {ex}")
                    robot.recover_from_errors()
                    
                time.sleep(1.0)

                # 3. Resume from failure point and apply micro-shift
                if retries < 3 and fail_x is not None:
                    # Find closest waypoint to where we failed
                    min_dist = float('inf')
                    closest_idx = 0
                    for i, (px, py) in enumerate(stroke_remainder):
                        dist = math.dist((fail_x, fail_y), (px + shift_x, py + shift_y))
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i
                            
                    # Backtrack by 1 waypoint to overlap slightly and avoid gaps
                    resume_idx = max(0, closest_idx - 1)
                    stroke_remainder = stroke_remainder[resume_idx:]
                    
                    # Add a random shift between -3mm and +3mm
                    shift_x += np.random.uniform(-0.003, 0.003)
                    shift_y += np.random.uniform(-0.003, 0.003)
                    
                    print(f"   ➔ Resuming from waypoint {resume_idx} with shift offset ({shift_x*1000:.1f}mm, {shift_y*1000:.1f}mm)")
        
        if not success:
            print(f"❌ Skipping stroke {idx+1} after repeated failures.")
            idx += 1

    # --- STOP LOGGING AND RENDER VIDEO ---
    logger.stop()
    save_telemetry_video(logger, "drawing_telemetry.mp4")

    print("Masterpiece complete!")

if __name__ == "__main__":
    ROBOT_IP = "192.168.0.2"  
    SVG_FILE = 'superbuff.svg' 
    
    strokes = generate_2d_strokes(
        svg_file=SVG_FILE,
        center_x=0.5,
        center_y=0.3,
        side_length=0.3,
        step_size=0.004   
    )
    
    execute_strokes(
        robot_ip=ROBOT_IP, 
        strokes=strokes, 
        z_draw=0.1, 
        z_transit=0.3, 
        dynamics_factor=0.03 
    )
