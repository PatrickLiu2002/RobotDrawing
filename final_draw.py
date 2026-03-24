import warp as wp
import math
import newton
import newton.examples
import newton.solvers
import newton.ik as ik
import newton.utils
from svgpathtools import svg2paths

@wp.kernel
def compute_tip_world_pos(
    body_q: wp.array(dtype=wp.transform), 
    ee_index: int, 
    offset: wp.vec3, 
    out_pos: wp.array(dtype=wp.vec3)
):
    tf = body_q[ee_index]
    out_pos[0] = wp.transform_point(tf, offset)

class Example:
    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        
        # ==========================================
        # 1. BUILD MODEL AND SCENE
        # ==========================================
        builder = newton.ModelBuilder()
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )
        builder.add_ground_plane()
        
        # Add a tilted table to draw on
        plane_pos = wp.vec3(0.5, 0.0, 0.15) 
        tilt_angle = -0.05 
        plane_rot = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), tilt_angle)
        
        builder.add_shape_box(
            body=-1, 
            hx=0.4, hy=0.4, hz=0.2,
            xform=wp.transform(
                wp.vec3(plane_pos.x, plane_pos.y, plane_pos.z - 0.19), 
                plane_rot
            )
        )

        # End Effector & Tool Setup
        self.ee_index = 9 
        
        # Cylinder housing
        builder.add_shape_cylinder(
            self.ee_index,
            radius=0.015, half_height=0.055, 
            xform=wp.transform(wp.vec3(0, 0, 0.11), wp.quat_identity())
        )
        # Spherical marker tip
        builder.add_shape_sphere(
            self.ee_index,
            radius=0.015,
            xform=wp.transform(wp.vec3(0, 0, 0.165), wp.quat_identity())
        )

        self.model = builder.finalize()
        self.viewer.set_model(self.model)
        
        # ==========================================
        # 2. PHYSICS & CONTROL SETUP
        # ==========================================
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state = self.state_0  
        self.control = self.model.control()
        
        # Tune joint stiffness for a compliant wrist
        ke_np = self.model.joint_target_ke.numpy()
        kd_np = self.model.joint_target_kd.numpy()
        
        ke_np[0:4] = 30000.0  # Stiff Base/Shoulder
        kd_np[0:4] = 400.0
        ke_np[4:7] = 5000.0  # Compliant Wrist
        kd_np[4:7] = 200.0
        
        ke_np[7:9] = 20000.0  # Very stiff fingers
        kd_np[7:9] = 200.0
        
        self.model.joint_target_ke.assign(ke_np)
        self.model.joint_target_kd.assign(kd_np)

        self.solver_physics = newton.solvers.SolverMuJoCo(self.model) 
        
        # ==========================================
        # 3. SENSOR & ADMITTANCE PARAMETERS
        # ==========================================
        self.target_force = 2.0      
        self.kp_force = 0.0001       
        self.filtered_force = 0.0    
        self.filter_alpha = 0.02
        
        # Adaptive tracking state
        self.center = wp.vec3(0.5, 0.0, 0.25) # High starting Z
        self.current_target_z = self.center.z 
        self.last_contact_z = self.center.z
        
        # Probe Phase state
        self.is_probed = False
        self.probe_speed = 0.02 # meters per second
        self.draw_start_time = 0.0
        
        # ==========================================
        # 4. SVG PARSING & EXECUTION PLAN
        # ==========================================
        self.side_length = 0.3
        self.draw_speed = 10  
        
        paths, attributes = svg2paths('house.svg')
        xmin, xmax, ymin, ymax = float('inf'), float('-inf'), float('inf'), float('-inf')
        for path in paths:
            for seg in path:
                x0, x1, y0, y1 = seg.bbox()
                xmin, xmax = min(xmin, x0), max(xmax, x1)
                ymin, ymax = min(ymin, y0), max(ymax, y1)
                
        self.svg_width = xmax - xmin
        self.svg_height = ymax - ymin
        self.svg_xmin = xmin
        self.svg_ymin = ymin

        self.execution_plan = []
        self.total_path_length = 0.0 
        last_point = None
        
        for path in paths:
            for seg in path:
                if last_point is not None:
                    gap_distance = abs(seg.start - last_point)
                    if gap_distance > 1e-3:
                        self.execution_plan.append({
                            'type': 'TRANSIT', 
                            'start': last_point, 
                            'end': seg.start, 
                            'length': gap_distance
                        })
                        self.total_path_length += gap_distance
                
                length = seg.length() 
                self.execution_plan.append({'type': 'DRAW', 'seg': seg, 'length': length})
                self.total_path_length += length
                last_point = seg.end
                    
        self.num_steps = len(self.execution_plan)
        self.period = self.total_path_length / self.draw_speed if self.total_path_length > 0 else 5.0
        
        # --- NEW CODE: Calculate exact start point to avoid teleporting ---
        first_point_complex = paths[0][0].start
        scale = self.side_length / max(self.svg_width, self.svg_height)
        offset_x = self.center.x - (self.svg_width * scale) / 2.0
        offset_y = self.center.y - (self.svg_height * scale) / 2.0
        
        nx = (first_point_complex.real - self.svg_xmin) * scale
        ny = (first_point_complex.imag - self.svg_ymin) * scale
        
        start_x = offset_x + nx
        start_y = offset_y - ny + (self.svg_height * scale)
        
        self.probe_pos = wp.vec3(start_x, start_y, self.center.z)
        # ------------------------------------------------------------------

        # ==========================================
        # 5. INVERSE KINEMATICS SETUP
        # ==========================================
        down_quat = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.18), # Traces the absolute tip of the sphere
            target_positions=wp.array([self.probe_pos], dtype=wp.vec3), # <-- CHANGED HERE
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*down_quat)], dtype=wp.vec4),
        )
        # Extract limits to numpy
        lower_limits = self.model.joint_limit_lower.numpy()
        upper_limits = self.model.joint_limit_upper.numpy()
        
        # Force the IK solver to ignore the fingers by locking their mathematical limits
        lower_limits[7:9] = 0.04
        upper_limits[7:9] = 0.04
        
        self.limit_obj = ik.IKObjectiveJointLimit(
            wp.array(lower_limits, dtype=wp.float32),
            wp.array(upper_limits, dtype=wp.float32)
        )
        
        # Changed the final two values to 0.04 to fully open the claws
        q_home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
        self.joint_q = wp.array([q_home[:self.model.joint_coord_count]], dtype=wp.float32)

        q_1d = self.joint_q.numpy()[0]
        self.control.joint_target_pos.assign(q_1d)
        self.state_0.joint_q.assign(q_1d)
        self.state_1.joint_q.assign(q_1d)

        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.limit_obj]
        )
        
        # Trail rendering data
        self.trail_points = []
        self.actual_trail_points = []
        self.tip_pos_out = wp.zeros(1, dtype=wp.vec3)
        self.contacts = None 

    def _get_svg_state(self, draw_time):
        """Calculates X/Y coordinate based on drawing progress."""
        if getattr(self, 'num_steps', 0) == 0 or self.total_path_length == 0:
            return self.center, True, 0.0
            
        # Freeze at end of drawing
        if draw_time >= self.period:
            draw_time = self.period - 0.001
            
        norm_t = (draw_time % self.period) / self.period 
        target_distance = norm_t * self.total_path_length
        
        current_dist = 0.0
        step_data = self.execution_plan[-1] 
        t_step = 1.0 
        
        for step in self.execution_plan:
            if current_dist + step['length'] >= target_distance:
                step_data = step
                if step['length'] > 1e-6:
                    t_step = (target_distance - current_dist) / step['length']
                else:
                    t_step = 0.0
                break
            current_dist += step['length']
            
        if step_data['type'] == 'DRAW':
            point_complex = step_data['seg'].point(t_step)
            is_drawing = True
            hop_offset = 0.0
        else:
            start_c = step_data['start']
            end_c = step_data['end']
            point_complex = start_c + (end_c - start_c) * t_step
            is_drawing = False
            # Lifts smoothly in the first half of the transit, then holds the height
            hop_offset = 0.03 * math.sin(min(t_step * 2.0, 1.0) * math.pi / 2)

        scale = self.side_length / max(self.svg_width, self.svg_height)
        offset_x = self.center.x - (self.svg_width * scale) / 2.0
        offset_y = self.center.y - (self.svg_height * scale) / 2.0
        
        nx = (point_complex.real - self.svg_xmin) * scale
        ny = (point_complex.imag - self.svg_ymin) * scale
        
        final_x = offset_x + nx
        final_y = offset_y - ny + (self.svg_height * scale)
        
        return wp.vec3(final_x, final_y, 0.0), is_drawing, hop_offset
        
    def step(self):
        # 1. Update Actual Tip Position
        wp.launch(
            kernel=compute_tip_world_pos,
            dim=1,
            inputs=[self.state_0.body_q, self.ee_index, wp.vec3(0.0, 0.0, 0.18)],
            outputs=[self.tip_pos_out]
        )
        wp.synchronize()
        actual_pos = self.tip_pos_out.numpy()[0]

        # --- FIX 1: Evaluate SVG State and detect stroke transitions ---
        draw_time = self.sim_time - self.draw_start_time
        
        # Initialize edge detection
        if not hasattr(self, 'was_drawing'):
            self.was_drawing = True 

        if self.is_probed:
            t_xy, is_drawing, hop_offset = self._get_svg_state(draw_time)
            target_xy = t_xy
            
            # Detect transition from TRANSIT to a NEW STROKE
            if is_drawing and not self.was_drawing:
                self.is_probed = False
                self.probe_pos = t_xy  # Lock X/Y here for the probe phase
                print("Transit complete. Probing for surface...")
                
            self.was_drawing = is_drawing
        else:
            # We are probing. Keep target_xy locked and pause the drawing timer.
            target_xy = self.probe_pos
            is_drawing = False 
            hop_offset = 0.0
            self.draw_start_time += self.frame_dt # Pauses SVG progress mathematically

        # 2. Purely Topological Collision Detection
        force_mag = 0.0
        is_touching = False  

        if self.contacts is not None and hasattr(self.contacts, "rigid_contact_count"):
            count = self.contacts.rigid_contact_count.numpy()[0] if hasattr(self.contacts.rigid_contact_count, "numpy") else self.contacts.rigid_contact_count
            if count > 0:
                shape0_np = self.contacts.rigid_contact_shape0.numpy()
                shape1_np = self.contacts.rigid_contact_shape1.numpy()
                shape_body_mapping = self.model.shape_body.numpy()
                
                safe_count = min(int(count), shape0_np.shape[0])
                for i in range(safe_count):
                    b0 = shape_body_mapping[shape0_np[i]]
                    b1 = shape_body_mapping[shape1_np[i]]
                    
                    # Touching the world (-1)
                    if (b0 == self.ee_index and b1 == -1) or (b1 == self.ee_index and b0 == -1):
                        is_touching = True
                        break

        # 3. Compute Deflection Force
        if is_touching:
            deflection = actual_pos[2] - self.current_target_z
            if deflection > 0.0: 
                force_mag = deflection * 1500.0 
                        
        self.filtered_force = (self.filter_alpha * force_mag) + ((1.0 - self.filter_alpha) * self.filtered_force)
        
        # Track physical contacts
        # FIX: Require actual physical pressure to ignore the collision margin
        if is_touching and is_drawing and self.is_probed and force_mag > 0.5: 
            self.actual_trail_points.append((actual_pos[0], actual_pos[1], actual_pos[2] + 0.02))
        else:
            if len(self.actual_trail_points) > 0 and self.actual_trail_points[-1] is not None:
                self.actual_trail_points.append(None)
                        
        self.filtered_force = (self.filter_alpha * force_mag) + ((1.0 - self.filter_alpha) * self.filtered_force)
        
        # ==========================================
        # 4. BEHAVIOR STATE MACHINE
        # ==========================================
        if not self.is_probed:
            # PHASE 1: Probing the surface
            if not is_touching:
                self.current_target_z -= self.probe_speed * self.frame_dt
            else:
                force_error = self.target_force - self.filtered_force
                self.current_target_z -= self.kp_force * force_error
                
                # Requires actual pressure equilibrium to finish probing
                if abs(force_error) < 0.2:
                    self.is_probed = True
                    self.last_contact_z = self.current_target_z  # Save detected table height
                    print("Surface detected. Starting/Resuming drawing sequence.")
        else:
            # PHASE 2: Drawing Sequence
            if is_drawing:
                # Normal admittance drawing
                force_error = self.target_force - self.filtered_force
                self.current_target_z -= self.kp_force * force_error
                self.current_target_z = max(0.05, min(0.3, self.current_target_z))
                
                self.last_contact_z = self.current_target_z  # Continuously track table height
                self.trail_points.append((target_xy.x, target_xy.y, self.current_target_z + 0.04))
                
            else:
                # Sensor-Driven Transit
                self.filtered_force = 0.0
                
                # Base the hop off the actual table height, not the high starting center!
                self.current_target_z = self.last_contact_z + hop_offset
                
                if len(self.trail_points) > 0 and self.trail_points[-1] is not None:
                    self.trail_points.append(None)

        # 5. Update IK Target
        final_target = wp.vec3(target_xy.x, target_xy.y, self.current_target_z)
        self.pos_obj.set_target_position(0, final_target)
        self.solver.step(self.joint_q, self.joint_q, iterations=64) 
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        
        # Render Kinematic Target Trail (Red)
        starts, ends = [], []
        for i in range(1, len(self.trail_points)):
            p1, p2 = self.trail_points[i - 1], self.trail_points[i]
            if p1 is not None and p2 is not None:
                starts.append(wp.vec3(*p1))
                ends.append(wp.vec3(*p2))
        
        if len(starts) > 0:
            self.viewer.log_lines(
                "target_trail", 
                wp.array(starts, dtype=wp.vec3), 
                wp.array(ends, dtype=wp.vec3), 
                (1.0, 0.0, 0.0)
            )
            
        # Render Physical Contact Trail (Blue)
        act_starts, act_ends = [], []
        for i in range(1, len(self.actual_trail_points)):
            p1, p2 = self.actual_trail_points[i - 1], self.actual_trail_points[i]
            if p1 is not None and p2 is not None:
                act_starts.append(wp.vec3(*p1))
                act_ends.append(wp.vec3(*p2))
                
        if len(act_starts) > 0:
            self.viewer.log_lines(
                "actual_trail", 
                wp.array(act_starts, dtype=wp.vec3), 
                wp.array(act_ends, dtype=wp.vec3), 
                (0.0, 1.0, 0.0)
            )
            
        # Assign IK output and step physics
        self.control.joint_target_pos.assign(self.joint_q.numpy()[0])
        self.contacts = self.model.collide(self.state_0)
        
        self.solver_physics.step(
            self.state_0, self.state_1, self.control, self.contacts, self.frame_dt
        )
        
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.state = self.state_0  
        
        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        wp.synchronize()
        
if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)
