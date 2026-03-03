#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Manual Control — FULL ADAS FUSION
Virtual Vahana 2026 — KCT Competition Entry

AEB  | FCW | ACC | LKA | LDW | TSR | Sign Detection | BSM

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Cycle Sensor Views (RGB/Depth/Sem/Lidar)
    . (Period)   : Toggle AEB Safety System
    K            : Cycle LKA/LDW states: BOTH ON -> LKA OFF -> BOTH OFF -> BOTH ON

    --- ADAS (ACC / FCW) ---
    J            : Toggle ADAS master switch (ACC/FCW)
    Y            : Toggle Adaptive Cruise Control (ACC)
    U            : Increase cruise speed (+5 km/h)
    O            : Decrease cruise speed (-5 km/h)

    F1           : toggle HUD info panel
    F2           : toggle ADAS LKA panel
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/carla')
except IndexError:
    pass

# Optional: LKA agent (f.py pattern)
try:
    from agents.navigation.roaming_agent import RoamingAgent
    from agents.navigation.local_planner import RoadOption
    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False

import carla
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def get_speed(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)


def get_traffic_light_state(vehicle, world):
    # PRIMARY: CARLA's built-in traffic light state (most reliable when at stop line)
    if vehicle.is_at_traffic_light():
        light = vehicle.get_traffic_light()
        if light:
            state = str(light.get_state()).split('.')[-1]
            # Compute real distance to the light — returning 0.0 was causing 'RED 0m' ghost
            # triggers. At 0m, tl_safe_dist = max(0.0, d_partial+1) which is identical to
            # a false close-range WARN. Use actual stop-line distance as floor of 3m.
            # Ref: J2945/1 V2X SPaT messages include distance-to-stop-line metric.
            try:
                stop_wps = light.get_stop_waypoints()
                if stop_wps:
                    v_loc = vehicle.get_transform().location
                    tl_dist = min(
                        math.sqrt((wp.transform.location.x - v_loc.x)**2 +
                                  (wp.transform.location.y - v_loc.y)**2)
                        for wp in stop_wps)
                    tl_dist = max(tl_dist, 2.0)  # never inject at < 2m — car is past the line
                else:
                    tl_dist = 5.0  # conservative fallback
            except Exception:
                tl_dist = 5.0
            return state, tl_dist, "STOP" if state == "Red" else "GO"

    vehicle_transform = vehicle.get_transform()
    vehicle_loc = vehicle_transform.location
    vehicle_fwd = vehicle_transform.get_forward_vector()

    speed_mps = vehicle.get_velocity()
    speed_kmh = 3.6 * math.sqrt(speed_mps.x**2 + speed_mps.y**2)

    # Scan radius scales with speed for early detection
    if speed_kmh > 15.0:
        scan_radius = 80.0
    elif speed_kmh > 5.0:
        scan_radius = 50.0
    else:
        scan_radius = 30.0

    # NOTE: world.traffic_lights is pre-fetched and cached every ~60 frames
    # in World.tick() to avoid calling get_actors() every frame here.
    lights = getattr(world, '_cached_traffic_lights', None)
    if lights is None:
        lights = world.get_actors().filter('traffic.traffic_light')

    closest_light = None
    closest_dist  = scan_radius

    for light in lights:
        light_loc = light.get_transform().location
        # Fast AABB pre-filter (much cheaper than sqrt)
        dx = abs(light_loc.x - vehicle_loc.x)
        dy = abs(light_loc.y - vehicle_loc.y)
        if dx > scan_radius or dy > scan_radius:
            continue

        dist = math.sqrt(dx*dx + dy*dy)
        if dist >= closest_dist:
            continue

        to_light = carla.Vector3D(
            light_loc.x - vehicle_loc.x, light_loc.y - vehicle_loc.y, 0)
        length = math.sqrt(to_light.x**2 + to_light.y**2)
        if length == 0:
            continue

        # Forward-cone: light must be directly ahead (relaxed to 35° cone)
        dot = (vehicle_fwd.x * to_light.x + vehicle_fwd.y * to_light.y) / length
        if dot < 0.82:
            continue

        # Orientation: light must face roughly toward us.
        # Relaxed check if very close to the signal/stopline to handle complex geometry.
        light_fwd = light.get_transform().get_forward_vector()
        light_dot = vehicle_fwd.x * light_fwd.x + vehicle_fwd.y * light_fwd.y
        if light_dot >= -0.3 and dist > 15.0:
            continue

        closest_light = light
        closest_dist  = dist

    if closest_light:
        state  = str(closest_light.get_state()).split('.')[-1]
        action = "STOP" if state == "Red" else ("WARN" if state == "Yellow" else "GO")
        return state, closest_dist, action

    return "Green", 999.0, "GO"

def get_upcoming_speed_limit(vehicle, world):
    vehicle_transform = vehicle.get_transform()
    vehicle_loc = vehicle_transform.location
    vehicle_fwd = vehicle_transform.get_forward_vector()
    speed_signs = world.get_actors().filter('traffic.speed_limit.*')
    upcoming_limit = None
    min_dist = 50.0 

    for sign in speed_signs:
        sign_loc = sign.get_transform().location
        if abs(sign_loc.x - vehicle_loc.x) > min_dist or abs(sign_loc.y - vehicle_loc.y) > min_dist: continue
        dist = math.sqrt((sign_loc.x - vehicle_loc.x)**2 + (sign_loc.y - vehicle_loc.y)**2)
        if dist < min_dist:
            to_sign = carla.Vector3D(sign_loc.x - vehicle_loc.x, sign_loc.y - vehicle_loc.y, 0)
            length = math.sqrt(to_sign.x**2 + to_sign.y**2)
            if length > 0:
                dot = (vehicle_fwd.x * to_sign.x + vehicle_fwd.y * to_sign.y) / length
                if dot > 0.95: # TIGHTER CONE (approx 18 degrees)
                    
                    # ORIENTATION CHECK
                    # Sign Forward vector must oppose vehicle forward vector
                    sign_fwd = sign.get_transform().get_forward_vector()
                    sign_dot = (vehicle_fwd.x * sign_fwd.x + vehicle_fwd.y * sign_fwd.y)
                    
                    # Only accept signs facing us (dot < -0.80, approx 35 deg play)
                    if sign_dot < -0.80:
                        try:
                            limit_str = sign.type_id.split('.')[-1]
                            upcoming_limit = int(limit_str)
                            min_dist = dist
                        except: continue
    return upcoming_limit


# ==============================================================================
# -- PIDController -------------------------------------------------------------
# ==============================================================================


class PIDController:
    """Simple PID controller used by ACC for longitudinal speed control."""
    def __init__(self, Kp, Ki, Kd, dt=0.03):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.error_sum = 0.0
        self.last_error = 0.0

    def run(self, error):
        self.error_sum += error * self.dt
        self.error_sum = max(-10.0, min(self.error_sum, 10.0))  # anti-windup
        derivative = (error - self.last_error) / self.dt
        self.last_error = error
        return (self.Kp * error) + (self.Ki * self.error_sum) + (self.Kd * derivative)


# ==============================================================================
# -- LKAController (MathWorks 4-Stage architecture, from f.py) -----------------
# ==============================================================================


class LKAController(object):
    """
    Lane Keeping Assist — RoamingAgent-based, MathWorks 4-Stage architecture.

    The original 50% flat authority works well up to ~60 km/h. Above that the
    RoamingAgent's waypoint-following PID outputs increasingly large steer values
    for the same lateral deviation (its look-ahead geometry projects errors further
    at speed), so 50% of a larger number is still too aggressive.

    Three fixes are applied inside compute_steer(), leaving all behaviour below
    60 km/h completely unchanged:

      1. Speed-adaptive authority
         Authority scales linearly from 0.50 at 60 km/h down to 0.28 at 130 km/h.
         Below 60 km/h it stays at exactly 0.50 — identical to the original.

      2. Per-frame steer rate limiter
         Maximum normalised steer change per frame = RATE_BASE / v_ms.
         At 50 km/h the cap is loose enough to never fire; at 100 km/h it limits
         the wheel to roughly half the rate, preventing snap corrections.

      3. One-pole IIR low-pass filter
         y[n] = α·x[n] + (1−α)·y[n−1].  α transitions from 0.55 (city) to 0.25
         (highway) so jitter from the agent PID is smoothed out at speed without
         making low-speed corrections feel sluggish.

    Speed ceiling raised from 80 km/h to 130 km/h — now safe because items 1–3
    keep highway corrections gentle.

    Suppression gates (any → LKA off):
      • speed < MIN_SPEED_MS  (2.8 m/s ≈ 10 km/h)
      • inside junction
      • within 15 m of a junction (turning_ahead)
      • wrong side of road
      • driver actively steering
    """
    THRESH_DEPART   = 0.45      # m   — departure detection threshold
    THRESH_CLEAR    = 0.22      # m   — hysteresis clear threshold
    MIN_SPEED_MS    = 2.8       # m/s — no LKA below ~10 km/h
    MAX_SPEED_MS    = 36.1      # m/s — 130 km/h ceiling (raised from 80 km/h)

    # ── Authority table ──────────────────────────────────────────────────────
    AUTH_SPD_LO  = 16.7         # m/s (60 km/h)  — full authority
    AUTH_SPD_HI  = 36.1         # m/s (130 km/h) — minimum authority
    AUTH_LO      = 0.50         # authority at ≤ 60 km/h  (unchanged from original)
    AUTH_HI      = 0.28         # authority at 130 km/h

    # ── Rate limiter ─────────────────────────────────────────────────────────
    # Per-frame cap = RATE_BASE / v_ms.
    # At 60 km/h:  0.22/16.7 ≈ 0.013/frame → ~0.8/s  (loose, rarely fires)
    # At 100 km/h: 0.22/27.8 ≈ 0.008/frame → ~0.5/s  (halved vs city speed)
    RATE_BASE    = 0.22

    # ── IIR low-pass filter ───────────────────────────────────────────────────
    ALPHA_LO     = 0.55         # light smoothing at ≤ 60 km/h
    ALPHA_HI     = 0.25         # heavy smoothing at ≥ 100 km/h
    ALPHA_SPD_HI = 27.8         # m/s (100 km/h) — full smoothing from here

    def __init__(self, vehicle, carla_map):
        self.vehicle   = vehicle
        self.map       = carla_map
        # CARLA RoamingAgent drives the waypoint PID controller
        if _AGENTS_AVAILABLE:
            self._agent = RoamingAgent(vehicle)
        else:
            self._agent = None
        # Observable telemetry (read by World.tick → HUD)
        self.lateral_dev        = 0.0
        self.yaw_error          = 0.0
        self.curvature          = 0.0
        self.steer_cmd          = 0.0
        self.departure_detected = False
        self.assist_active      = False
        # Internal hysteresis latch
        self._departed          = False
        # Rate limiter state — tracks last rate-limited output
        self._prev_rate_out     = 0.0
        # IIR filter state — tracks last filtered output
        self._filtered_out      = 0.0

    # ── Speed-adaptive helpers ────────────────────────────────────────────────
    @classmethod
    def _authority(cls, v_ms):
        """Returns 0.50 at ≤60 km/h, linearly reduces to 0.28 at 130 km/h."""
        if v_ms <= cls.AUTH_SPD_LO:
            return cls.AUTH_LO
        if v_ms >= cls.AUTH_SPD_HI:
            return cls.AUTH_HI
        t = (v_ms - cls.AUTH_SPD_LO) / (cls.AUTH_SPD_HI - cls.AUTH_SPD_LO)
        return cls.AUTH_LO + t * (cls.AUTH_HI - cls.AUTH_LO)

    @classmethod
    def _rate_cap(cls, v_ms):
        """Max steer change per frame. Tightens proportionally above 60 km/h."""
        return cls.RATE_BASE / max(v_ms, cls.AUTH_SPD_LO)

    @classmethod
    def _alpha(cls, v_ms):
        """IIR smoothing coefficient — decreases (more smoothing) above 60 km/h."""
        if v_ms <= cls.AUTH_SPD_LO:
            return cls.ALPHA_LO
        if v_ms >= cls.ALPHA_SPD_HI:
            return cls.ALPHA_HI
        t = (v_ms - cls.AUTH_SPD_LO) / (cls.ALPHA_SPD_HI - cls.AUTH_SPD_LO)
        return cls.ALPHA_LO + t * (cls.ALPHA_HI - cls.ALPHA_LO)

    # ── Stage 1 – Estimate Lane Centre ────────────────────────────────────────
    def _estimate_lane_center(self):
        """Return (lateral_dev, yaw_error, curvature, is_junction, wp)."""
        transform = self.vehicle.get_transform()
        location  = transform.location
        try:
            wp = self.map.get_waypoint(
                location, project_to_road=True,
                lane_type=carla.LaneType.Driving)
            if wp is None:
                return 0.0, 0.0, 0.0, False, None
        except Exception:
            return 0.0, 0.0, 0.0, False, None

        # Lateral deviation — signed distance from lane centre
        wp_loc   = wp.transform.location
        wp_right = wp.transform.get_right_vector()
        dx = location.x - wp_loc.x
        dy = location.y - wp_loc.y
        lateral_dev = dx * wp_right.x + dy * wp_right.y

        # Relative yaw
        v_fwd  = transform.get_forward_vector()
        wp_fwd = wp.transform.get_forward_vector()
        cross_z   = v_fwd.x * wp_fwd.y - v_fwd.y * wp_fwd.x
        dot       = max(-1.0, min(1.0, v_fwd.x * wp_fwd.x + v_fwd.y * wp_fwd.y))
        yaw_error = math.atan2(cross_z, dot)

        # Previewed curvature over 2-step look-ahead
        v     = self.vehicle.get_velocity()
        speed = max(0.0, math.sqrt(v.x**2 + v.y**2))
        ahead = max(8.0, 2.0 * speed)
        curvature = 0.0
        try:
            nwps = wp.next(ahead)
            if nwps:
                n_fwd = nwps[0].transform.get_forward_vector()
                c_z   = wp_fwd.x * n_fwd.y - wp_fwd.y * n_fwd.x
                c_dot = max(-1.0, min(1.0, wp_fwd.x * n_fwd.x + wp_fwd.y * n_fwd.y))
                curvature = math.atan2(c_z, c_dot) / max(ahead, 0.1)
        except Exception:
            pass

        return lateral_dev, yaw_error, curvature, wp.is_junction, wp

    # ── Stage 3 – Detect Departure (hysteresis) ───────────────────────────────
    def _detect_departure(self, lat_dev):
        if self._departed:
            self._departed = abs(lat_dev) > self.THRESH_CLEAR
        else:
            self._departed = abs(lat_dev) > self.THRESH_DEPART
        self.departure_detected = self._departed
        return self._departed

    # ── Stage 2+4 – Compute Steer (called each frame from KeyboardControl) ────
    def compute_steer(self, current_waypoint):
        """
        Returns (conditioned_steer, assist_active).

        'conditioned_steer' is ready to write into steer_cache directly —
        do NOT multiply by any authority factor at the call site.

        Pipeline:
          agent.run_step()          → raw PID steer
          × _authority(v_ms)        → Fix 1: scale down above 60 km/h
          rate-limited by _rate_cap → Fix 2: cap per-frame wheel movement
          IIR filtered by _alpha    → Fix 3: smooth out PID jitter at speed
        """
        if self._agent is None:
            return 0.0, False

        lat_dev, yaw_err, curvature, is_junction, wp = self._estimate_lane_center()
        self.lateral_dev = lat_dev
        self.yaw_error   = yaw_err
        self.curvature   = curvature

        depart = self._detect_departure(lat_dev)

        # Lock the agent's local planner to the current lane
        if current_waypoint is not None:
            lp = self._agent._local_planner
            if (lp.target_waypoint is not None and
                    (lp.target_waypoint.lane_id != current_waypoint.lane_id or
                     lp.target_waypoint.road_id != current_waypoint.road_id)):
                lp._waypoints_queue.clear()
                lp._waypoint_buffer.clear()
                lp._waypoints_queue.append((current_waypoint, RoadOption.LANEFOLLOW))

        # Raw steer from the agent's waypoint-following PID
        raw_steer = self._agent.run_step().steer

        # Current vehicle speed for all three fixes
        v    = self.vehicle.get_velocity()
        v_ms = math.sqrt(v.x**2 + v.y**2 + v.z**2)

        # ── Fix 1: Speed-adaptive authority ───────────────────────────────────
        # Below 60 km/h: exactly 0.50 — identical to the original.
        # Above 60 km/h: linearly reduces to 0.28 at 130 km/h.
        scaled = raw_steer * self._authority(v_ms)

        # ── Fix 2: Per-frame steer rate limiter ───────────────────────────────
        # Prevents sudden large PID corrections from snapping the wheel.
        # Cap tightens with speed so highway corrections feel as smooth as city.
        cap      = self._rate_cap(v_ms)
        delta    = max(-cap, min(cap, scaled - self._prev_rate_out))
        rate_out = self._prev_rate_out + delta
        self._prev_rate_out = rate_out

        # ── Fix 3: IIR low-pass filter ────────────────────────────────────────
        # Smooths residual PID jitter. Heavier smoothing above 80 km/h.
        a        = self._alpha(v_ms)
        filtered = a * rate_out + (1.0 - a) * self._filtered_out
        self._filtered_out = filtered

        self.steer_cmd     = filtered      # exposed to HUD telemetry
        self.assist_active = depart
        return filtered, depart



# ==============================================================================
# -- ADAS class (ACC + FCW, from ACC & FCW.py) ---------------------------------
# ==============================================================================


class ADAS(object):
    def __init__(self, world):
        self.world = world
        self.active = False
        self.acc_active = False
        self.acc_active = False
        self.fcw_active = True # FCW always on by default if ADAS is enabled
        
        self.target_speed = 30.0 # km/h
        self.time_gap = 2.0      # Seconds
        
        # PID Controllers
        self.lon_pid = PIDController(Kp=1.0, Ki=0.05, Kd=0.1)  # Throttle/Brake

        # State
        self.lead_vehicle = None
        self.lead_dist = 0.0
        self.lead_vel = 0.0 # Relative velocity
        self.speed_limit = 0.0
        self.status = "OFF" 
        self.overspeed_warning = False
        
        self.radar_front = None
        self.radar_rear = None

    def set_radars(self, front, rear):
        self.radar_front = front
        self.radar_rear = rear

    def toggle(self):
        self.active = not self.active
        self.status = "STANDBY" if self.active else "OFF"
        if not self.active:
            self.acc_active = False

    def toggle_fcw(self):
        self.fcw_active = not self.fcw_active

    def change_speed(self, amount):
        self.target_speed = max(20, min(150, self.target_speed + amount))

    def run_step(self, player, control):
        if not player:
            return control
            
        ego_loc = player.get_location()
        ego_tf = player.get_transform()
        ego_speed = get_speed(player)
        
        # Reset Status for this step (Fixes "stuck" warning)
        if self.active: self.status = "STANDBY"
        else: self.status = "OFF"
        self.overspeed_warning = False
        self.speed_limit = 0.0

        # Speed Limit Detection
        limit = player.get_speed_limit()
        if limit is not None:
             # Ignore very low limits (e.g. 10-20km/h) as they are often map artifacts
             if float(limit) > 20.0:
                 self.speed_limit = float(limit)

        # 1. PERCEPTION
        self.lead_vehicle = None
        self.lead_dist = 100.0
        self.lead_vel = 0.0
        
        # A. SIMULATION GROUND TRUTH (Cameras/Lidar equivalent)
        # Detect Vehicles AND Walkers
        actors = self.world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.*')
        all_obstacles = list(vehicles) + list(walkers)
        
        ego_fwd = ego_tf.get_forward_vector()
        
        for target in all_obstacles:
            if target.id == player.id: continue
            
            target_loc = target.get_location()
            target_vec = target_loc - ego_loc
            dist = math.sqrt(target_vec.x**2 + target_vec.y**2 + target_vec.z**2)
            
            if dist > 120.0: continue
            
            target_dir = target_vec / dist
            dot = ego_fwd.x * target_dir.x + ego_fwd.y * target_dir.y + ego_fwd.z * target_dir.z
            
            # Improved Lane Check (Lateral Distance)
            fwd_dist = dot * dist
            
            # Check if in front
            if fwd_dist > 0:
                # Calculate lateral distance
                lat_dist = math.sqrt(max(0, dist**2 - fwd_dist**2))
                
                # Lane width is typically 3.5m. Half is 1.75m. 
                # We use 2.5m to allow for some margin (curved roads, sloppy driving) but exclude next lane.
                if lat_dist < 2.5: 
                    if dist < self.lead_dist:
                        self.lead_dist = dist
                        self.lead_vehicle = target
                        # Calculate rel speed roughly from sim state? 
                        # For now relying on derived rel_speed later
                    
        # B. RADAR FUSION (Real Sensor Data)
        # Detects walls, static objects, terrain, etc.
        
        front_dist = 100.0
        rear_dist = 100.0
        
        # Process Front Radar
        if self.radar_front and hasattr(self.radar_front, 'data'):
             for detection in self.radar_front.data:
                 d = detection['dist']
                 azi = detection['azi']
                 alt = detection['alt']
                 
                 # Filtering Logic
                 # 1. Lateral Distance: strictly within Ego's lane (+/- 1.5m)
                 lat_dist_radar = abs(d * math.sin(math.radians(azi)))
                 
                 # 2. Height: Approximate Z relative to sensor
                 pitch_rad = math.radians(5.0)
                 alt_rad = math.radians(alt)
                 z_rel = d * math.sin(pitch_rad + alt_rad)
                 
                 if lat_dist_radar < 1.5: 
                     # Ground Filter: Ignore points below -0.2m relative to sensor (so >0.8m above ground)
                     if z_rel > -0.2: 
                         if d > 2.0: # Ignore self/bonnet
                             if d < front_dist: front_dist = d
        
        # Process Rear Radar
        if self.radar_rear and hasattr(self.radar_rear, 'data'):
             for detection in self.radar_rear.data:
                 d = detection['dist']
                 azi = detection['azi']
                 if abs(azi) < 10.0 and d > 2.0:
                     if d < rear_dist: rear_dist = d

        # Directional Logic (Smart Throttle & AEB source)
        is_reverse = control.reverse
        
        if is_reverse:
            # REVERSE LOGIC
            # Completely ignore forward perception
            self.lead_vehicle = None 
            self.lead_dist = rear_dist # Look behind
            
            # Rear Safety Stop
            if rear_dist < 5.0:
                control.throttle = 0.0
                control.brake = 1.0
                self.status = "REQ: BLOCKED REAR"
                
        else:
            # FORWARD LOGIC
            # Use Ground Truth (visual) only for ACC to prevent radar ghost-braking.
            pass

        # 2. FORWARD COLLISION WARNING (FCW)
        # Warns but DOES NOT INTERVENE
        # Only active if vehicle is moving (ego_speed > 0.5) to prevent stationary warnings.
        # Use only visual ground-truth (lead_dist) when moving forward to prevent fake radar warnings on clear roads!
        warn_dist = self.lead_dist if not is_reverse else rear_dist
        if self.active and self.fcw_active and ego_speed > 0.5 and warn_dist < 100.0:
             if warn_dist < 10.0:
                 self.status = "WARNING: RED"
             elif warn_dist < 25.0:
                 self.status = "WARNING: ORANGE"
             elif warn_dist < 40.0:
                 self.status = "WARNING: YELLOW"

        # 3. MANUAL MODE SPEED LIMITER (When ACC is OFF)
        if self.active and not self.acc_active and self.speed_limit > 0:
             # Logic: If speed > limit, just display warning but do not control speed
             if ego_speed > self.speed_limit:
                 self.overspeed_warning = True

        # If Master Switch is OFF, return here
        if not self.active:
            return control

        # 4. ACC
        if self.acc_active:
            # Adaptive to Speed Limit
            if self.speed_limit > 0:
                 self.target_speed = self.speed_limit

            if self.target_speed < 20: self.target_speed = 20 # Minimum functional speed

            target_v = self.target_speed
            
            # Traffic Light Detection (Red Light Stop)
            if player.is_at_traffic_light():
                traffic_light = player.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red or \
                   traffic_light.get_state() == carla.TrafficLightState.Yellow:
                    
                    # USER REQUEST: 
                    # (i) If first at signal (no lead vehicle close), let manual take over.
                    # (ii) If following (queue), let ADAS handle it.
                    
                    if self.lead_dist > 15.0: # No car immediately ahead
                         target_v = 0.0
                         self.status = "ACC: RED LIGHT"
                         # Logic: instead of False, keep True and set target_v=0 for auto-resume
                    
                    # Else: Queue logic
                    target_v = 0.0
                    self.status = "ACC: TRAFFIC LIGHT (Q)"
            
            if self.lead_dist < 100.0:
                safe_dist = (ego_speed / 3.6) * self.time_gap + 6.0 # + Margin
                
                # STOP HOLD LOGIC (ACC)
                # If we are stopped and object is close, HOLD brake
                if ego_speed < 0.5 and self.lead_dist < 6.0:
                     target_v = 0.0
                     control.brake = 1.0
                     self.status = "ACC: HOLD"
                     control.throttle = 0.0
                     return control

                if self.lead_dist < safe_dist:
                    # Follow logic
                    # We want to match lead speed, but we don't know it exactly if no lead_vehicle object (Radar only)
                    # Simple P controller on distance
                    
                    error_dist = self.lead_dist - safe_dist
                    # Target speed based on distance error
                    target_v = max(0.0, ego_speed + (error_dist * 2.0)) # Simple P-control for speed
                    
            error = target_v - ego_speed
            
            output = self.lon_pid.run(error)
            
            if output > 0:
                control.throttle = min(1.0, output)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(1.0, abs(output))
            
            self.status = f"ACC {int(self.target_speed)}" 
            if target_v == 0 and self.status == "ACC: RED LIGHT": self.status = "acc: RED LIGHT"
            if self.status == "ACC: HOLD": pass # Keep HOLD status if set earlier? Logic overridden by status line above.
            if ego_speed < 0.5 and self.lead_dist < 6.0: self.status = "ACC: HOLD" # Re-assert for display



        return control



# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world          = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            import sys; sys.exit(1)
        self.hud    = hud
        self.player = None

        # --- Sensors ---
        self.collision_sensor      = None
        self.lane_invasion_sensor  = None
        self.gnss_sensor           = None
        self.imu_sensor            = None
        self.radar_sensor          = None
        self.rear_radar_sensor     = None
        self.obstacle_sensor       = None
        self.semantic_lidar_sensor = None
        self.camera_manager        = None

        self._weather_presets  = find_weather_presets()
        self._weather_index    = 0
        self._actor_filter     = args.filter
        self._gamma            = args.gamma

        # AEB / ADAS state
        self.adas_enabled      = True
        self.speed_limit       = 30
        self.speed_warning     = False
        self.obstacle_distance = None
        self.aeb_active        = False
        self.aeb_state         = 0
        self.brake_demand      = 0.0
        self.aeb_decel_req     = 0.0
        self.bsm_left          = False
        self.bsm_right         = False
        self.current_road_name = "Unknown"
        self.tl_info           = {}
        self.target_threat_actor = None
        self.rel_speed_history = collections.deque(maxlen=3)
        self.pending_sign      = None
        self.sign_timer        = 0
        self.standstill_hold   = False

        self._tick_counter          = 0
        self._cached_actors         = None
        self._cached_vehicles       = None
        self._cached_walkers        = None
        self._cached_traffic_lights = None
        self._stall_ticks           = 0
        self._bsm_immunity          = 0
        self._aeb_immunity          = 0
        self._threat_persist_ticks  = 0
        self._sem_persist           = 0
        self._sem_persist_rear      = 0
        self.aeb_state_timer        = 0

        # LKA / LDW state
        self.lka_enabled        = True
        self.ldw_enabled        = True
        self.lka_controller     = None
        self.lka_lateral_dev    = 0.0
        self.lka_yaw_err        = 0.0
        self.lka_steer_cmd      = 0.0
        self.lka_departure      = False
        self.lka_assist_active  = False
        self.manual_steer_value = 0.0
        self.lka_steer_applied  = 0.0

        # ACC / FCW
        self.adas = ADAS(self.world)
        self.hud.set_adas(self.adas)

        self.hud.load_blueprints(self.world.get_blueprint_library())
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled         = False
        self.recording_start           = 0
        self.constant_velocity_enabled = False
        self.current_map_layer         = 0
        self.map_layer_names           = [
            carla.MapLayer.NONE,         carla.MapLayer.Buildings,
            carla.MapLayer.Decals,       carla.MapLayer.Foliage,
            carla.MapLayer.Ground,       carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,    carla.MapLayer.Props,
            carla.MapLayer.StreetLights, carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        self.player_max_speed      = 1.589
        self.player_max_speed_fast = 3.713
        cam_index     = self.camera_manager.index           if self.camera_manager else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager else 0

        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute("role_name", self.actor_role_name)
        if blueprint.has_attribute("color"):
            blueprint.set_attribute("color", random.choice(
                blueprint.get_attribute("color").recommended_values))
        if blueprint.has_attribute("driver_id"):
            blueprint.set_attribute("driver_id", random.choice(
                blueprint.get_attribute("driver_id").recommended_values))

        if self.player is not None:
            try:
                spawn_point            = self.player.get_transform()
                spawn_point.location.z += 2.0
                spawn_point.rotation.roll  = 0.0
                spawn_point.rotation.pitch = 0.0
                self.destroy()
            except Exception:
                pass
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.standstill_hold = False

        while self.player is None:
            if not self.map.get_spawn_points():
                import sys; sys.exit(1)
            spawn_point = random.choice(self.map.get_spawn_points())
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        self.modify_vehicle_physics(self.player)

        self.collision_sensor      = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor  = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor           = GnssSensor(self.player)
        self.imu_sensor            = IMUSensor(self.player)
        self.radar_sensor          = RadarSensor(self.player, role="front")
        self.rear_radar_sensor     = RadarSensor(self.player, role="rear")
        self.obstacle_sensor       = ObstacleSensor(self.player)
        self.semantic_lidar_sensor = SemanticLidarSensor(self.player)
        self.adas.set_radars(self.radar_sensor, self.rear_radar_sensor)
        self.lka_controller        = LKAController(self.player, self.map)
        self.camera_manager        = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.hud.notification(get_actor_display_name(self.player))

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification("LayerMap selected: %s" % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification("Unloading map layer: %s" % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification("Loading map layer: %s" % selected)
            self.world.load_map_layer(selected)

    def modify_vehicle_physics(self, vehicle):
        physics = vehicle.get_physics_control()
        physics.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics)

    def toggle_lka_ldw(self):
        """Cycle: BOTH ON -> LKA OFF/LDW ON -> BOTH OFF -> BOTH ON"""
        if self.lka_enabled and self.ldw_enabled:
            self.lka_enabled = False;  self.ldw_enabled = True
            msg = "LKA: OFF  |  LDW: ON"
        elif not self.lka_enabled and self.ldw_enabled:
            self.lka_enabled = False;  self.ldw_enabled = False
            msg = "LKA: OFF  |  LDW: OFF"
        else:
            self.lka_enabled = True;   self.ldw_enabled = True
            msg = "LKA: ON   |  LDW: ON"
        self.lka_assist_active = False
        self.lka_departure     = False
        if self.lka_controller:
            self.lka_controller.assist_active      = False
            self.lka_controller.departure_detected = False
        self.hud.notification(msg, seconds=3.0)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player, role="front")
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.toggle_display()

    def get_smart_threat(self):
        """Ground-truth actor-level threat detection with lateral corridor check."""
        if not self.player:
            return None, 999.0, 999.0, 0.0
        my_trans = self.player.get_transform()
        my_loc   = my_trans.location
        my_fwd   = my_trans.get_forward_vector()
        my_vel   = self.player.get_velocity()
        vehicles = self._cached_vehicles or []
        walkers  = self._cached_walkers  or []
        targets  = list(vehicles) + list(walkers)
        closest_actor     = None
        min_dist          = 100.0
        min_ttc           = 999.0
        min_closing_speed = 0.0

        for actor in targets:
            if actor.id == self.player.id:
                continue
            a_loc   = actor.get_location()
            dist_sq = (a_loc.x - my_loc.x)**2 + (a_loc.y - my_loc.y)**2
            if dist_sq > 10000:
                continue
            dist = math.sqrt(dist_sq)
            to_actor = carla.Vector3D(a_loc.x - my_loc.x, a_loc.y - my_loc.y, 0)
            fwd_dist = to_actor.x * my_fwd.x + to_actor.y * my_fwd.y
            if fwd_dist < -1.0:
                continue
            my_right      = my_trans.get_right_vector()
            lat_dist      = to_actor.x * my_right.x + to_actor.y * my_right.y
            steer         = abs(self.player.get_control().steer)
            speed_kmh     = 3.6 * math.sqrt(my_vel.x**2 + my_vel.y**2)
            
            # --- DYNAMIC TUNNEL & CORNERING FILTER ---
            # 1. Fetch road geometry
            try:
                wp = self.map.get_waypoint(my_loc)
                lane_w = wp.lane_width if wp else 3.5
            except:
                lane_w = 3.5

            # 2. Path-Aware projection: Get waypoint at future distance
            # For cornering, we evaluate the actor against the ego's future trajectory
            # This prevents tunnel pillars/adjacent cars from being "seen" as in-path mid-turn.
            future_wp_dist = max(10.0, 1.2 * speed_kmh / 3.6)
            try:
                lookahead_wps = wp.next(future_wp_dist)
                if lookahead_wps:
                    target_wp = lookahead_wps[0]
                    f_tf      = target_wp.transform
                    f_fwd     = f_tf.get_forward_vector()
                    f_right   = f_tf.get_right_vector()
                else:
                    f_fwd, f_right = my_fwd, my_right
            except:
                f_fwd, f_right = my_fwd, my_right

            # --- ONCOMING & ADJACENT LANE FILTER ---
            a_vel = actor.get_velocity()
            a_speed_fwd = a_vel.x * my_fwd.x + a_vel.y * my_fwd.y
            if a_speed_fwd < -2.0 and abs(lat_dist) > 0.8: continue
            if abs(a_speed_fwd) > 1.0 and abs(lat_dist) > 1.4: continue
            
            # 3. Dynamic Margin: Tighten in tunnels/narrow roads
            # If we are in LKA mode and stable, we can afford a tighter corridor.
            base_margin = 0.8 if speed_kmh > 50.0 else 1.0
            if self.lka_assist_active: base_margin -= 0.1
            
            # Corridor reflects lane width (ignore objects outside 85% of lane half-width)
            lane_margin = min(base_margin, (lane_w * 0.42))
            
            t_half_w = actor.bounding_box.extent.y if hasattr(actor,"bounding_box") else 0.5
            t_half_w = min(t_half_w, 1.1)
            eff_margin = lane_margin + t_half_w
            
            # Strict cap for tunnels
            eff_margin = min(eff_margin, 1.6) 

            in_path = abs(lat_dist) < eff_margin
            
            # 4. PATH-AWARE FUTURE CHECK (Cornering Fix)
            # Evaluate actor's future position relative to EGO'S future frame
            rel_vel       = carla.Vector3D(a_vel.x - my_vel.x, a_vel.y - my_vel.y, 0)
            closing_speed = -(rel_vel.x*to_actor.x + rel_vel.y*to_actor.y) / max(dist,0.01)
            
            future_pos    = a_loc + a_vel * 1.0 # 1s prediction
            to_future     = carla.Vector3D(future_pos.x - my_loc.x, future_pos.y - my_loc.y, 0)
            # Use future right vector for lateral check around curves
            fut_lat_dist  = to_future.x * f_right.x + to_future.y * f_right.y
            in_fut_path   = abs(fut_lat_dist) < eff_margin
            
            if dist < 10.0:
                avg_cs = max(0.0, closing_speed)
            else:
                self.rel_speed_history.append(max(0.0, closing_speed))
                avg_cs = sum(self.rel_speed_history)/len(self.rel_speed_history) if self.rel_speed_history else 0.0
            
            v_rel_cl   = max(0.0, closing_speed)
            bb         = actor.bounding_box.extent if hasattr(actor,"bounding_box") else None
            has_volume = bb is not None and (bb.x * bb.y) > 0.08
            my_spd_mps = math.sqrt(my_vel.x**2 + my_vel.y**2)

            if (in_path or in_fut_path) and fwd_dist > 1.0 and has_volume and dist > 1.5:
                eff_cl = v_rel_cl if v_rel_cl > 0.1 else my_spd_mps
                if eff_cl > 0.1:
                    dyn_ttc = dist / eff_cl
                    if dist < min_dist:
                        min_dist = dist; min_ttc = dyn_ttc; min_closing_speed = eff_cl
                        closest_actor = actor
        return closest_actor, min_dist, min_ttc, min_closing_speed

    def tick(self, clock):
        self.hud.tick(self, clock)
        self._tick_counter  += 1
        self._cached_actors  = self.world.get_actors()
        self._cached_vehicles = self._cached_actors.filter("vehicle.*")
        self._cached_walkers  = self._cached_actors.filter("walker.*")
        if self._tick_counter % 60 == 0 or self._cached_traffic_lights is None:
            self._cached_traffic_lights = self._cached_actors.filter("traffic.traffic_light")

        v     = self.player.get_velocity()
        t     = self.player.get_transform()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        v_ms  = speed / 3.6

        try:
            wp = self.map.get_waypoint(t.location)
            self.current_road_name = wp.road_id if wp else "Off-road"
        except Exception:
            self.current_road_name = "Unknown"

        tl_state, tl_dist, tl_action = get_traffic_light_state(self.player, self.world)
        self.tl_info = {"state": tl_state, "dist": tl_dist, "action": tl_action}

        current_limit  = self.player.get_speed_limit()
        if current_limit is None: current_limit = 30
        upcoming_limit = get_upcoming_speed_limit(self.player, self.world)
        if upcoming_limit == self.pending_sign:
            self.sign_timer += 1
        else:
            self.pending_sign = upcoming_limit
            self.sign_timer   = 0
        self.speed_limit   = self.pending_sign if (self.sign_timer > 15 and self.pending_sign) else current_limit
        self.speed_warning = speed > (self.speed_limit + 3.0)

        # --- REALISTIC SENSOR-BASED SLOPE DETECTION (IMU) ---
        # 1. IMU Gyro Pitch (t.rotation.pitch perfectly maps to the vehicle chassis IMU)
        is_steep_slope = (t.rotation.pitch > 4.5)

        is_dynamic_threat = False
        if self.radar_sensor and self.radar_sensor.radar_threat:
            r_vel = self.radar_sensor.radar_threat["rel_vel"]
            if abs(r_vel + v_ms) > 2.0: is_dynamic_threat = True

        # ================================================================
        # AEB LOGIC
        # ================================================================
        if not self.adas_enabled:
            self.aeb_active = False; self.brake_demand = 0.0
        else:
            final_dist = None; final_ttc = 99.0
            radar_vel_gate = max(2.5, v_ms * 0.8)
            if self.radar_sensor and self.radar_sensor.radar_threat:
                dist    = self.radar_sensor.radar_threat["dist"]
                rel_vel = self.radar_sensor.radar_threat["rel_vel"]
                if rel_vel > radar_vel_gate * 0.1:
                    ttc = dist / rel_vel if rel_vel > 0.1 else 999.0
                    final_ttc = min(final_ttc, ttc); final_dist = dist

            obs_dist         = self.obstacle_sensor.distance
            obs_dist_for_hud = obs_dist
            self.obstacle_sensor.distance = None
            if obs_dist is not None and 0.1 < obs_dist <= 50.0:
                ttc = obs_dist / max(v_ms, 0.05)
                final_ttc = min(final_ttc, ttc)
                if final_dist is None or obs_dist < final_dist: final_dist = obs_dist

            v_mps          = max(v_ms, 0.1)
            t_react        = 1.2                         # m/s² reaction time (UNECE R152)
            a_partial      = 0.35 * 9.8
            a_full         = 9.81                        # ~1g full AEB
            d_brake_part   = (v_mps**2) / (2*a_partial)
            d_brake_full   = (v_mps**2) / (2*a_full)

            # Speed-proportional distances (correct for all speeds including highway)
            d_warn         = v_mps * t_react + 0.5 * a_partial * t_react**2 + 6.0
            d_partial      = v_mps * t_react + d_brake_part + 4.0
            d_full         = v_mps * 0.8     + d_brake_full + 2.5

            smart_actor, smart_dist, smart_ttc, smart_speed = self.get_smart_threat()
            sem_dist     = self.semantic_lidar_sensor.dist

            primary_dist = 999.0; primary_ttc = 99.0
            if final_dist is not None:
                primary_dist = min(primary_dist, final_dist)
                primary_ttc  = min(primary_ttc,  final_ttc)
            if smart_dist < primary_dist:
                primary_dist = smart_dist; primary_ttc = min(primary_ttc, smart_ttc)

            primary_active = primary_dist < 50.0

            # --- SENSOR FUSION GHOST FILTER ---
            # Realistic logic: If the front raycast/radar detects an obstacle (like a slope),
            # but Semantic Lidar (which actively classifies and ignores road/ground surfaces) 
            # does NOT see anything structural there, it's a false positive caused by the road pitch.
            # SENSOR FUSION GHOST FILTER (Restored full coverage)
            # Nosedives under hard braking cause the Raycast to hit the asphalt at 2-3m.
            # Semantic Lidar classifies the road as a `surface_tag` and ignores it. 
            # So if Raycast sees an obstacle but Semantic Lidar doesn't, it is a ground ghost.
            # FIX: Only suppress if moving at speed (> 0.5 m/s). If stopped, TRUST the sensors.
            if primary_active and not is_dynamic_threat and v_ms > 0.5:
                if sem_dist is None or sem_dist > (primary_dist + 3.0):
                    primary_dist = 999.0
                    primary_ttc = 99.0
                    primary_active = False

            merged_dist   = primary_dist; final_ttc = primary_ttc

            if primary_active and sem_dist is not None and 5.0 < sem_dist < 35.0:
                merged_dist = min(merged_dist, sem_dist)

            target_dist = merged_dist
            if primary_active and obs_dist_for_hud is not None and 3.0 < obs_dist_for_hud <= 8.0:
                target_dist = min(target_dist, obs_dist_for_hud)

            steer_bsm = self.player.get_control().steer
            if self._bsm_immunity > 0:
                self._bsm_immunity -= 1
            elif (steer_bsm < -0.2 and self.bsm_left) or (steer_bsm > 0.2 and self.bsm_right):
                merged_dist = min(merged_dist, d_partial - 0.1); target_dist = merged_dist

            # Removed AEB traffic light trigger (ACC handles TL smoothly now)

            new_state = 0; brake_demand = 0.0
            
            # CRITICAL (Full Brake)
            if target_dist < d_full:
                if is_steep_slope and not is_dynamic_threat:
                    pass # Physical pitch suppression for bumps
                else:
                    new_state = 3; brake_demand = 1.0
            
            # STAGE 2: Partial Braking (Now with Linear Ramping)
            elif target_dist < d_partial:
                new_state = 2
                # Calculate a linear ramp: 0.1 at d_warn down to 0.45 at d_partial
                # Using a 10% floor for smoothness, ramping up as we get closer
                ramp_f = (d_warn - target_dist) / (d_warn - d_partial)
                brake_demand = np.clip(0.1 + (ramp_f * 0.35), 0.1, 0.45)
            
            # WARNING only (With persistence filter)
            elif target_dist < d_warn:
                new_state = 1
                brake_demand = 0.0

            control_now = self.player.get_control()
            if abs(control_now.steer) > 0.15 and new_state == 1 and target_dist > 10.0:
                new_state = 0; brake_demand = 0.0

            import pygame
            from pygame.locals import K_w, K_UP, K_s, K_DOWN
            keys = pygame.key.get_pressed()
            driver_accelerating = keys[K_w] or keys[K_UP]

            # CREEP / LOW-SPEED OVERRIDE
            # Removed `v_ms > 0.1` gate. We want it to catch crawls down to 0 mph.
            is_reversing = control_now.reverse
            creep_thresh = 3.0 if is_reversing else 2.5
            if (target_dist < creep_thresh and not is_steep_slope and
                    self._threat_persist_ticks >= 2 and primary_active):
                new_state = 3; brake_demand = 1.0

            # LOW-SPEED WALL STOP (FORWARD) — ISO 22737:2021 LSAEB Ported from abc.py
            sem_persist = getattr(self, '_sem_persist', 0)
            close_wall = sem_dist is not None and sem_dist < 3.5 and v_ms < 1.39
            imminent_wall = sem_dist is not None and sem_dist < 2.0 and v_ms > 1.39
            
            if close_wall or imminent_wall:
                self._sem_persist = sem_persist + 1
            else:
                self._sem_persist = max(0, sem_persist - 1)
                
            # Removed `v_ms > 0.05` gate. The wall does not stop existing when we stop moving.
            if getattr(self, '_sem_persist', 0) >= 3 and sem_dist is not None and not is_reversing:
                wall_dist_thresh = 3.5 if v_ms < 1.39 else 2.0
                if sem_dist < wall_dist_thresh:
                    new_state = 3
                    brake_demand = 1.0
                    target_dist = min(target_dist, sem_dist)

            # ── AEB STANDSTILL HOLD & DRIVER OVERRIDE ──
            # Once triggered by a wall at standstill, lock the brakes until Reverse is selected or threat cleared.
            is_wall_near = (sem_dist is not None and sem_dist < 3.2)
            if self.aeb_state == 3 and v_ms < 0.2 and is_wall_near:
                self.standstill_hold = True
            
            if self.standstill_hold:
                if is_wall_near and not is_reversing:
                    new_state = 3
                    brake_demand = 1.0
                    self.aeb_state_timer = 30 # Maintain latch 
                    if driver_accelerating:
                        self.hud.notification("STANDSTILL HOLD: WALL AHEAD", seconds=1.0)
                else:
                    self.standstill_hold = False # Reset if reversed or wall gone

            # ── ADAPTIVE HYSTERESIS LATCH ──
            if new_state >= self.aeb_state:
                if new_state > 0:
                    # Emergency (3) holds longer (0.6s), others shorter (0.2s) for responsiveness
                    self.aeb_state_timer = 35 if new_state == 3 else 12
            else:
                if getattr(self, 'aeb_state_timer', 0) > 0:
                    if driver_accelerating and not is_wall_near:
                        self.aeb_state_timer = 0
                    else:
                        self.aeb_state_timer -= 1
                        new_state = self.aeb_state 
                        # Use the same ramping logic if held in state 2
                        if new_state == 2:
                            ramp_f = (d_warn - target_dist) / (d_warn - d_partial)
                            brake_demand = np.clip(0.1 + (ramp_f * 0.35), 0.1, 0.45)
                        else:
                            brake_demand = 1.0 if new_state == 3 else 0.0

            sem_dist_rear = getattr(self.semantic_lidar_sensor, "dist_rear", None)
            if (self._sem_persist_rear >= 3 and sem_dist_rear is not None and
                    sem_dist_rear < 3.0 and is_reversing and v_ms > 0.1):
                new_state = 3; brake_demand = 1.0

            # Simplified Persistence filters (LSAEB cleaned up)
            TTC_W = 2.6; TTC_P = 1.8; TTC_E = 1.2
            if primary_active and final_ttc < 99.0:
                if final_ttc < TTC_E and new_state < 3:   new_state = 3; brake_demand = 1.0
                elif final_ttc < TTC_P and new_state < 2: new_state = 2; brake_demand = 0.6
                elif final_ttc < TTC_W and new_state < 1: new_state = 1

            PERSIST = 3
            # Stage 2+ needs 3 frames (50ms)
            if new_state >= 2 and final_ttc >= 2.0:
                self._threat_persist_ticks += 1
                if self._threat_persist_ticks < PERSIST:
                    new_state = min(new_state, 1); brake_demand = 0.0
            # Stage 1 (Warning) needs 2 frames (30ms) to filter noise/flicker
            elif new_state == 1:
                self._warn_persist_ticks = getattr(self, '_warn_persist_ticks', 0) + 1
                if self._warn_persist_ticks < 2:
                    new_state = 0
            else:
                self._threat_persist_ticks = PERSIST if new_state >= 2 else 0
                self._warn_persist_ticks = 0

            if new_state >= self.aeb_state:
                self.aeb_state = new_state
                if new_state >= 2 and self.aeb_state_timer == 0: self.aeb_state_timer = 20
                self.brake_demand = brake_demand
            else:
                if new_state <= 1:
                    self.aeb_state = new_state; self.brake_demand = 0.0 if new_state == 0 else brake_demand
                    self.aeb_state_timer = 0
                elif self.aeb_state_timer > 0:
                    self.aeb_state_timer -= 1
                else:
                    self.aeb_state    = max(new_state, self.aeb_state - 1)
                    self.brake_demand = 0.0 if self.aeb_state == 0 else brake_demand
                    self.aeb_state_timer = 10

            is_stall = ((self.aeb_state == 3 and v_ms < 0.3) or
                        (self.aeb_state in (1,2) and v_ms < 0.5 and self.brake_demand > 0.0))
            if is_stall:
                self._stall_ticks += 1
                if self._stall_ticks > (45 if self.aeb_state == 3 else 20):
                    # FIX: Do NOT release if primary_active is True OR we still see a wall very close.
                    if not primary_active and not is_wall_near:
                        self.aeb_state = 0; self.brake_demand = 0.0; self.aeb_state_timer = 0
                        self.standstill_hold = False
                    self._stall_ticks = 0; self._bsm_immunity = 90
            elif v_ms > 0.7:
                self._stall_ticks = 0

            self.aeb_active    = self.aeb_state >= 2
            self.aeb_decel_req = self.brake_demand * 10.0

            # BSM
            self.bsm_left = self.bsm_right = False
            self.bsm_left_id = self.bsm_right_id = -1
            for vehicle in (self._cached_vehicles or []):
                if vehicle.id == self.player.id: continue
                v_loc = vehicle.get_location()
                p_loc = t.location
                dist  = math.sqrt((v_loc.x-p_loc.x)**2 + (v_loc.y-p_loc.y)**2)
                if dist < 12.0:
                    fwd   = t.get_forward_vector()
                    to_v  = carla.Vector3D(v_loc.x-p_loc.x, v_loc.y-p_loc.y, 0)
                    cz    = fwd.x*to_v.y - fwd.y*to_v.x
                    dot   = fwd.x*to_v.x + fwd.y*to_v.y
                    if abs(dot) < 0.5:
                        if cz >  0.2:
                            self.bsm_right = True; self.bsm_right_id = vehicle.id
                        elif cz < -0.2:
                            self.bsm_left  = True; self.bsm_left_id = vehicle.id

            if (steer_bsm < -0.2 and self.bsm_left) or (steer_bsm > 0.2 and self.bsm_right):
                self.hud.notification("BSM: COLLISION RISK IN BLIND SPOT!")

            trigger_src = ""
            if self.aeb_state >= 1:
                if tl_state in ("Red","Yellow") and tl_dist < d_warn and v_ms > 1.4:
                    trigger_src = "RED LIGHT"
                elif smart_dist < merged_dist + 1.0 and smart_dist < 50:
                    trigger_src = "VEHICLE/PED"
                elif sem_dist is not None and sem_dist < 35.0 and sem_dist < merged_dist + 1.0:
                    trigger_src = "LIDAR WALL"
                elif self.radar_sensor and self.radar_sensor.radar_threat:
                    trigger_src = "RADAR"
                elif obs_dist_for_hud is not None and obs_dist_for_hud <= 30.0:
                    trigger_src = "RAYCAST"
                else:
                    trigger_src = "SENSOR"
                if (steer_bsm < -0.2 and self.bsm_left) or (steer_bsm > 0.2 and self.bsm_right):
                    trigger_src = "BSM"

        self.target_threat_actor = smart_actor if self.aeb_state >= 1 else None

        # LKA state pass-through
        if self.lka_controller:
            self.lka_lateral_dev   = self.lka_controller.lateral_dev
            self.lka_yaw_err       = self.lka_controller.yaw_error
            self.lka_steer_cmd     = self.lka_controller.steer_cmd
            self.lka_departure     = self.lka_controller.departure_detected
            self.lka_assist_active = self.lka_controller.assist_active

        # Radar points for TeslaDashboard
        radar_pts = self.radar_sensor.points_to_draw if self.radar_sensor else []

        self.hud.adas_data = {
            "tl_state":    tl_state,    "tl_dist":      tl_dist,
            "limit":       self.speed_limit,             "obs_dist":     final_dist if self.adas_enabled else None,
            "aeb":         self.aeb_active,              "aeb_state":    self.aeb_state if self.adas_enabled else 0,
            "warning":     self.speed_warning,           "ttc":          final_ttc if self.adas_enabled else 99.0,
            "bsm_left":    self.bsm_left,                "bsm_right":    self.bsm_right,
            "bsm_left_id": getattr(self, "bsm_left_id", -1), "bsm_right_id": getattr(self, "bsm_right_id", -1),
            "threat_actor": self.target_threat_actor,   "steering":     self.player.get_control().steer,
            "radar_points": radar_pts,                   "trigger_src":  trigger_src if self.adas_enabled else "",
            "lka_enabled":  self.lka_enabled,            "ldw_enabled":  self.ldw_enabled,
            "lka_active":   self.lka_assist_active,      "lka_departure": self.lka_departure,
            "lka_lat_dev":  self.lka_lateral_dev,        "lka_yaw_err":  self.lka_yaw_err,
            "lka_steer":    self.lka_steer_cmd,          "manual_steer": self.manual_steer_value,
            "lka_steer_now": self.lka_steer_applied,
        }

        # Tesla dashboard actor update (if HUD has it)
        if hasattr(self.hud, "tesla_dashboard") and self.hud.show_tesla_hud:
            self.hud.tesla_dashboard.tick(self)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display, world=self)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index  = None

    def destroy(self):
        sensors = [
            self.camera_manager.sensor   if self.camera_manager else None,
            self.collision_sensor.sensor if self.collision_sensor else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor else None,
            self.gnss_sensor.sensor      if self.gnss_sensor else None,
            self.imu_sensor.sensor       if self.imu_sensor else None,
            self.radar_sensor.sensor     if self.radar_sensor else None,
            self.rear_radar_sensor.sensor if self.rear_radar_sensor else None,
            self.obstacle_sensor.sensor  if self.obstacle_sensor else None,
        ]
        if self.semantic_lidar_sensor:
            try:
                self.semantic_lidar_sensor.destroy()
            except Exception:
                pass
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass
        if self.player is not None:
            try:
                self.player.destroy()
            except Exception:
                pass


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Keyboard driver interface — ADAS-aware (AEB override + LKA blending)."""

    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        self._lka_no_steer_time = 0.0
        self._lka_steer_lerp = 0.0
        # Steer contribution tracking for HUD display
        self._manual_steer_applied = 0.0   # steer added by the driver this frame
        self._lka_steer_applied    = 0.0   # steer added by LKA this frame
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_F2:
                    world.hud.toggle_adas_panel()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.camera_manager.next_sensor()
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.recording_enabled:
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is Off")
                    else:
                        client.start_recorder("manual_recording.log")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is On")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    if not world.recording_enabled:
                        client.start_recorder(os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            "_out", "recording.log"))
                        world.recording_enabled = True
                        world.hud.notification("Recorder is On")
                    else:
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is Off")
                elif event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self._control.gear = world.player.get_control().gear
                    world.hud.notification(
                        "%s Transmission" % ("Manual" if self._control.manual_gear_shift else "Automatic"))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Autopilot %s" % ("On" if self._autopilot_enabled else "Off"))
                elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                    current_lights ^= carla.VehicleLightState.Special1
                elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                    current_lights ^= carla.VehicleLightState.HighBeam
                elif event.key == K_l:
                    current_lights ^= carla.VehicleLightState.Position
                    if not (current_lights & carla.VehicleLightState.Position):
                        current_lights &= ~(carla.VehicleLightState.LowBeam | carla.VehicleLightState.HighBeam)
                    if current_lights & carla.VehicleLightState.Position:
                        current_lights |= carla.VehicleLightState.LowBeam
                elif event.key == K_i:
                    current_lights ^= carla.VehicleLightState.Interior
                elif event.key == K_z:
                    current_lights ^= carla.VehicleLightState.LeftBlinker
                elif event.key == K_x:
                    current_lights ^= carla.VehicleLightState.RightBlinker

                # ---- ADAS key bindings ----
                elif event.key == K_PERIOD:
                    world.adas_enabled = not world.adas_enabled
                    if not world.adas_enabled:
                        world.aeb_state   = 0
                        world.brake_demand = 0.0
                    world.hud.notification("AEB System: %s" % ("ON" if world.adas_enabled else "OFF"))
                elif event.key == K_j:
                    world.adas.toggle()
                    world.hud.notification("ADAS (ACC/FCW): %s" % world.adas.status)
                elif event.key == K_y:
                    world.adas.acc_active = not world.adas.acc_active
                    world.hud.notification("ACC: %s" % ("ON" if world.adas.acc_active else "OFF"))
                elif event.key == K_u:
                    world.adas.change_speed(5)
                    world.hud.notification("Cruise speed: %d km/h" % world.adas.target_speed)
                elif event.key == K_o:
                    world.adas.change_speed(-5)
                    world.hud.notification("Cruise speed: %d km/h" % world.adas.target_speed)
                elif event.key == K_k:
                    world.toggle_lka_ldw()
                elif event.key == K_g:
                    if world.radar_sensor:
                        world.radar_sensor.toggle_display()
                elif event.key == K_t:
                    world.hud.show_tesla_hud = not world.hud.show_tesla_hud
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_MINUS:
                    if pygame.key.get_mods() & KMOD_CTRL:
                        if world.player.is_alive:
                            world.player.set_velocity(carla.Vector3D(0, 0, 0))
                    else:
                        world.hud.notification("Velocity cleared")

            if event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # scroll up
                    world.camera_manager.next_sensor()
                elif event.button == 5:  # scroll down
                    world.camera_manager.set_sensor(world.camera_manager.index - 1)

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time(), world)
                self._control.reverse = self._control.gear < 0
                if isinstance(current_lights, int):
                    world.player.set_light_state(carla.VehicleLightState(current_lights))
                    self._lights = carla.VehicleLightState(current_lights)
                else:
                    world.player.set_light_state(current_lights)
                    self._lights = current_lights

                # ---- ACC / FCW run_step ----
                if world.adas.active:
                    self._control = world.adas.run_step(world.player, self._control)

                # ---- AEB HARD OVERRIDE (always wins over ALL other ADAS) ----
                # Priority order: AEB > ACC > LKA
                # State 3 (Emergency): full brake, zero throttle, zero LKA steer
                # State 2 (Partial):   partial brake, cut throttle
                # State 1 (Warning):   throttle cap only
                if world.adas_enabled and world.brake_demand > 0.0:
                    if world.aeb_state >= 2:
                        self._control.throttle = 0.0
                        self._control.brake     = max(self._control.brake, world.brake_demand)
                        # Zero LKA steer contribution — steer to controlled straight
                        # (don't force steer=0 outright; just remove assist so driver stays in control)
                        world.lka_steer_applied = 0.0
                        if world.lka_controller:
                            world.lka_controller.assist_active      = False
                            world.lka_controller._prev_steer_out    = 0.0
                        # Freeze ACC PID integrator to prevent wind-up
                        if world.adas.acc_active:
                            world.adas.lon_pid.error_sum = 0.0
                    elif world.aeb_state == 1:
                        self._control.throttle = min(self._control.throttle, 0.3)

                world.player.apply_control(self._control)
                world.manual_steer_value = self._control.steer

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)

        if isinstance(current_lights, int):
            self._lights = carla.VehicleLightState(self._lights)
        return False

    def _parse_vehicle_keys(self, keys, milliseconds, world):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        manual_steering = keys[K_LEFT] or keys[K_a] or keys[K_RIGHT] or keys[K_d]

        if manual_steering:
            # Driver is actively steering — reset LKA ramp immediately
            self._lka_no_steer_time = 0.0
            self._lka_steer_lerp = 0.0
            if keys[K_LEFT] or keys[K_a]:
                if self._steer_cache > 0:
                    self._steer_cache = 0
                else:
                    self._steer_cache -= steer_increment
            elif keys[K_RIGHT] or keys[K_d]:
                if self._steer_cache < 0:
                    self._steer_cache = 0
                else:
                    self._steer_cache += steer_increment
            self._manual_steer_applied = self._steer_cache
            self._lka_steer_applied    = 0.0
        else:
            # Smooth decay back to centre (0.82 per frame at 60 fps ≈ ~0.1 s to 10%)
            self._steer_cache = self._steer_cache * 0.82
            self._lka_no_steer_time += milliseconds
            self._manual_steer_applied = 0.0

        # ---------------------------------------------------------------
        # LKA  —  MathWorks 4-Stage Lane Keep Assist
        # compute_steer() now handles all conditioning internally:
        #   · speed-adaptive authority (0.50 at ≤60 km/h → 0.28 at 130 km/h)
        #   · per-frame rate limiter  (cap tightens with speed)
        #   · IIR low-pass filter     (heavier smoothing above 80 km/h)
        # Speed ceiling raised from 80 km/h → 130 km/h.
        # ---------------------------------------------------------------
        if world.lka_enabled and not manual_steering:
            is_reverse = self._control.gear < 0 or self._control.reverse

            v          = world.player.get_velocity()
            speed_ms   = math.sqrt(v.x**2 + v.y**2 + v.z**2)

            current_waypoint = world.map.get_waypoint(world.player.get_location())
            is_in_junction   = current_waypoint.is_junction if current_waypoint else False

            # Look 15 m ahead to detect upcoming junctions
            turning_ahead = False
            if current_waypoint:
                next_wps = current_waypoint.next(15.0)
                if next_wps and next_wps[0].is_junction:
                    turning_ahead = True

            # Wrong-side-of-road detection
            v_forward  = world.player.get_transform().get_forward_vector()
            wp_forward = (current_waypoint.transform.get_forward_vector()
                          if current_waypoint else v_forward)
            dot_product   = v_forward.x * wp_forward.x + v_forward.y * wp_forward.y
            is_wrong_side = dot_product < -0.5

            # Ceiling raised to MAX_SPEED_MS (130 km/h).
            # Aggressiveness above 60 km/h is handled inside compute_steer.
            lka_ready = (
                not is_reverse and
                not is_in_junction and
                not turning_ahead and
                not is_wrong_side and
                self._lka_no_steer_time > 100.0 and
                LKAController.MIN_SPEED_MS <= speed_ms <= LKAController.MAX_SPEED_MS
            )

            if lka_ready and world.lka_controller is not None:
                # compute_steer returns a fully conditioned value — write directly.
                # Do NOT multiply by any authority factor; it is applied inside.
                # The 500 ms lerp ramp is replaced by the per-frame rate limiter
                # which gives smoother, speed-aware engagement with no jerk.
                conditioned, assist_active = world.lka_controller.compute_steer(
                    current_waypoint)

                # Push telemetry for HUD
                world.lka_assist_active = assist_active
                world.lka_lateral_dev   = world.lka_controller.lateral_dev
                world.lka_yaw_err       = world.lka_controller.yaw_error
                world.lka_steer_cmd     = world.lka_controller.steer_cmd
                world.lka_departure     = world.lka_controller.departure_detected

                self._lka_steer_lerp    = 1.0           # keep flag consistent
                self._steer_cache       = conditioned
                self._lka_steer_applied = conditioned
            else:
                self._lka_steer_lerp     = 0.0
                self._steer_cache        = self._steer_cache * 0.82  # smooth release
                self._lka_steer_applied  = 0.0
                world.lka_assist_active  = False
                world.lka_departure      = False

        elif not world.lka_enabled:
            # LKA OFF — only decay when NOT driving manually (no cap on manual steer)
            self._lka_steer_lerp    = 0.0
            self._lka_steer_applied = 0.0
            if not manual_steering:
                # Hands-off centre return
                self._steer_cache = self._steer_cache * 0.82

        # Expose per-frame steer contributions to world for HUD
        world.manual_steer_value = self._manual_steer_applied
        world.lka_steer_applied  = self._lka_steer_applied

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 2)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font           = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name      = "courier" if os.name == "nt" else "mono"
        fonts          = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font   = "ubuntumono"
        mono           = default_font if default_font in fonts else (fonts[0] if fonts else None)
        mono_font      = pygame.font.match_font(mono)
        self._font_mono     = pygame.font.Font(mono_font, 12 if os.name == "nt" else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help           = HelpText(pygame.font.Font(mono_font, 16), width, height)
        self.server_fps     = 0
        self.frame          = 0
        self.simulation_time = 0
        self._show_info     = True
        self._show_adas_panel = True
        self._info_text     = []
        self._server_clock  = pygame.time.Clock()
        self.adas_data      = {}
        self._adas          = None
        self.show_tesla_hud = True
        self.tesla_dashboard = TeslaDashboard(width, height)
        self._flash_timer   = 0
        self._ldw_active    = True
        self._vehicle_blueprints = []

    def set_adas(self, adas_obj):
        self._adas = adas_obj

    def load_blueprints(self, bp_library):
        self._vehicle_blueprints = [
            bp.id for bp in bp_library.filter("vehicle.*")
            if bp.has_attribute("number_of_wheels")]

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps      = self._server_clock.get_fps()
        self.frame           = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        self._flash_timer = (self._flash_timer + 1) % 30
        if not self._show_info:
            return

        t  = world.player.get_transform()
        v  = world.player.get_velocity()
        c  = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = "N" if compass > 270.5 or compass < 89.5 else (
            "S" if 90.5 < compass < 269.5 else (
                "E" if 0.5 < compass < 179.5 else "W"))

        imu_acc = world.imu_sensor.accelerometer
        imu_gyro = world.imu_sensor.gyroscope

        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist.get(x + self.frame - 200, 0) for x in range(0, 200)]
        max_col   = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles  = world.world.get_actors().filter("vehicle.*")

        self._info_text = [
            "Server:  % 16.0f FPS" % self.server_fps,
            "Client:  % 16.0f FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % get_actor_display_name(world.player, truncate=20),
            "Map:     % 20s" % world.map.name,
            "Simulation time: % 12s" % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u"Compass:% 17.0f\N{DEGREE SIGN} % 2s" % (compass, heading),
            "Accel:  (%5.1f, %5.1f, %5.1f)" % imu_acc,
            "Gyro:   (%5.1f, %5.1f, %5.1f)" % imu_gyro,
            "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (t.location.x, t.location.y)),
            "GNSS:% 24s" % ("(% 2.6f, % 3.6f)" % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            "Height:  % 18.0f m" % t.location.z,
            "",
        ]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ("Throttle:", c.throttle, 0.0, 1.0),
                ("Steer:",    c.steer,    -1.0, 1.0),
                ("Brake:",    c.brake,    0.0, 1.0),
                ("Reverse:",  c.reverse),
                ("Hand brake:", c.hand_brake),
                ("Manual:",   c.manual_gear_shift),
                "Gear:        %s" % {-1: "R", 0: "N"}.get(c.gear, c.gear),
            ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ("Speed:",    c.speed, 0.0, 5.556),
                ("Jump:",     c.jump),
            ]
        self._info_text += [
            "",
            "Number of vehicles: % 8d" % len(list(vehicles)),
        ]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]
            distance = lambda l: math.sqrt(
                (l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles
                        if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0: break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append("% 4dm %s" % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def toggle_adas_panel(self):
        self._show_adas_panel = not self._show_adas_panel

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text("Error: %s" % text, (255, 0, 0))

    def render(self, display, world=None):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18

        self._notifications.render(display)
        self.help.render(display)

        if world is None or not self._show_adas_panel:
            if self.show_tesla_hud and world is not None:
                self.tesla_dashboard.render(display, self.adas_data)
            return

        # ----------------------------------------------------------------
        # Flash timer
        flash_on = (self._flash_timer < 15)
        dw, dh   = self.dim

        # ----------------------------------------------------------------
        # LKA / LDW top-centre pill (from f.py)
        # ----------------------------------------------------------------
        lka_en  = self.adas_data.get("lka_enabled",  True)
        ldw_en  = self.adas_data.get("ldw_enabled",  True)
        lka_act = self.adas_data.get("lka_active",   False)
        ldw_dep = self.adas_data.get("lka_departure",False)

        pill_w, pill_h = 260, 28
        pill_x = dw // 2 - pill_w // 2
        pill_y = 6
        pill_surf = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)

        lka_col = (40, 180, 80)  if lka_en  else (80, 80, 80)
        ldw_col = (40, 180, 80)  if ldw_en  else (80, 80, 80)
        if lka_act: lka_col = (255, 180, 0)
        if ldw_dep and flash_on: ldw_col = (255, 60, 60)

        pill_surf.fill((20, 25, 35, 200))
        pygame.draw.rect(display, (40, 45, 60), (pill_x, pill_y, pill_w, pill_h), border_radius=14)

        f_pill = pygame.font.Font(None, 20)
        lka_txt = f_pill.render("LKA " + ("ON" if lka_en else "OFF"), True, lka_col)
        ldw_txt = f_pill.render("LDW " + ("ON" if ldw_en else "OFF"), True, ldw_col)
        sep_txt = f_pill.render("|", True, (80, 85, 100))

        display.blit(lka_txt, (pill_x + 12, pill_y + 6))
        display.blit(sep_txt, (pill_x + pill_w // 2 - 4, pill_y + 6))
        display.blit(ldw_txt, (pill_x + pill_w // 2 + 10, pill_y + 6))

        # Assist badge
        if lka_act:
            act_txt = pygame.font.Font(None, 17).render("ASSIST", True, (255, 220, 80))
            display.blit(act_txt, (pill_x + pill_w - act_txt.get_width() - 4, pill_y - 14))

        # LDW full-width departure warning banner
        if ldw_dep and flash_on and ldw_en:
            ldw_bw = 380; ldw_bh = 32
            ldw_bx = dw // 2 - ldw_bw // 2; ldw_by = 38
            ldw_s  = pygame.Surface((ldw_bw, ldw_bh), pygame.SRCALPHA)
            ldw_s.fill((120, 40, 0, 210))
            display.blit(ldw_s, (ldw_bx, ldw_by))
            pygame.draw.rect(display, (255, 120, 0), (ldw_bx, ldw_by, ldw_bw, ldw_bh), 2)
            ldw_msg = pygame.font.Font(None, 28).render("⚠  LANE DEPARTURE WARNING", True, (255, 180, 60))
            display.blit(ldw_msg, (ldw_bx + ldw_bw // 2 - ldw_msg.get_width() // 2,
                                   ldw_by + ldw_bh // 2 - ldw_msg.get_height() // 2))

        # ----------------------------------------------------------------
        # RIGHT-SIDE MASTER ADAS PANEL (Consolidated)
        # ----------------------------------------------------------------
        rp_w = 260; rp_h = 320
        rp_x = dw - rp_w - 10; rp_y = 10
        panel = pygame.Surface((rp_w, rp_h), pygame.SRCALPHA)
        aeb_state  = self.adas_data.get("aeb_state", 0)
        border_col = ((180,220,80) if aeb_state == 0 else
                      (220,160,0) if aeb_state == 1 else (220,50,50))
        panel.fill((15, 18, 28, 200))
        display.blit(panel, (rp_x, rp_y))
        pygame.draw.rect(display, border_col, (rp_x, rp_y, rp_w, rp_h), 2, border_radius=8)

        fs = pygame.font.Font(None, 22)
        sb = pygame.font.Font(None, 20)
        cy = rp_y + 10

        # Title
        aeb_label_col = border_col
        aeb_label_str = (["AEB READY","AEB WARN ⚠","AEB BRAKE","AEB EMERG 🔴"])[min(aeb_state, 3)]
        aeb_lbl = fs.render(aeb_label_str, True, aeb_label_col)
        display.blit(aeb_lbl, (rp_x + rp_w // 2 - aeb_lbl.get_width() // 2, cy))
        cy += 26

        # ── Row 1: TSR — Traffic Light (graphical circle) + Speed Limit Sign ──
        tl_state = self.adas_data.get("tl_state", "Green")
        tl_dist  = self.adas_data.get("tl_dist", 999.0)
        limit    = self.adas_data.get("limit", 30)
        warn_now = self.adas_data.get("warning", False)

        fm = pygame.font.Font(None, 26)

        # -- Traffic light housing (vertical 3-lamp box) --
        tl_box_x = rp_x + 8; tl_box_y = cy
        tl_box_w = 26;        tl_box_h = 70
        pygame.draw.rect(display, (30, 30, 30), (tl_box_x, tl_box_y, tl_box_w, tl_box_h), border_radius=4)
        pygame.draw.rect(display, (90, 90, 90), (tl_box_x, tl_box_y, tl_box_w, tl_box_h), 1, border_radius=4)
        lamp_cx = tl_box_x + tl_box_w // 2
        lamp_states = [("Red", tl_box_y + 12), ("Yellow", tl_box_y + 35), ("Green", tl_box_y + 58)]
        tl_colors_on  = {"Red": (255, 40, 40), "Yellow": (255, 210, 0), "Green": (40, 220, 80)}
        tl_colors_off = {"Red": (60, 10, 10), "Yellow": (50, 42, 0), "Green": (10, 50, 15)}
        for lamp_name, lamp_y in lamp_states:
            is_on = (lamp_name == tl_state)
            col   = tl_colors_on[lamp_name] if is_on else tl_colors_off[lamp_name]
            pygame.draw.circle(display, col, (lamp_cx, lamp_y), 9)
            if is_on:  # glow ring
                pygame.draw.circle(display, col, (lamp_cx, lamp_y), 11, 2)

        # TL label + distance
        tl_label_col = tl_colors_on.get(tl_state, (200, 200, 200))
        tl_name_s    = fs.render("TL: %s" % tl_state, True, tl_label_col)
        display.blit(tl_name_s, (tl_box_x + tl_box_w + 6, cy + 2))
        if tl_dist < 200:
            tl_dist_s = fs.render("%dm" % int(tl_dist), True, (190, 190, 190))
            display.blit(tl_dist_s, (tl_box_x + tl_box_w + 6, cy + 20))

        # -- Speed limit sign (European round red-ring sign) --
        sl_cx = rp_x + rp_w - 32;  sl_cy = cy + 34
        sl_r  = 22
        # White fill
        pygame.draw.circle(display, (255, 255, 255), (sl_cx, sl_cy), sl_r)
        # Red ring (3px)
        pygame.draw.circle(display, (210, 20, 20), (sl_cx, sl_cy), sl_r, 3)
        # Speed number (black)
        lim_col_sign = (0, 0, 0)
        sl_txt = fm.render(str(int(limit)), True, lim_col_sign)
        display.blit(sl_txt, (sl_cx - sl_txt.get_width() // 2, sl_cy - sl_txt.get_height() // 2))
        # Overspeed flash ring
        if warn_now and flash_on:
            pygame.draw.circle(display, (255, 60, 0), (sl_cx, sl_cy), sl_r + 3, 2)
            ov_s = fs.render("OVER!", True, (255, 80, 0))
            display.blit(ov_s, (sl_cx - ov_s.get_width() // 2, sl_cy + sl_r + 4))

        cy += 78  # row height covers both TL box and speed sign


        # Row 2: Obs dist / TTC
        obs_dist = self.adas_data.get("obs_dist")
        ttc      = self.adas_data.get("ttc", 99.0)
        dist_str = ("%.1f m" % obs_dist) if obs_dist is not None else "CLEAR"
        ttc_str  = ("TTC %.1fs" % ttc) if ttc < 90.0 else "TTC ---"
        obs_col  = (255,100,100) if obs_dist is not None and obs_dist < 8.0 else (
                   (255,200,0) if obs_dist is not None and obs_dist < 20.0 else (0, 220, 255))
        ttc_col  = (255,100,100) if ttc < 2.5 else ((255,200,0) if ttc < 4.0 else (100, 220, 255))
        d_s = fm.render(dist_str, True, obs_col)
        t_s = fm.render(ttc_str,  True, ttc_col)
        display.blit(d_s, (rp_x + 10, cy))
        display.blit(t_s, (rp_x + 10, cy + 24))
        cy += 50

        # ADAS Status (ACC/FCW)
        if self._adas and self._adas.active:
            pygame.draw.line(display, (60, 65, 80), (rp_x + 10, cy), (rp_x + rp_w - 10, cy), 1)
            cy += 10
            status_col = (80, 200, 255) if not self._adas.overspeed_warning else (255, 100, 0)
            status_txt = fs.render("ADAS: %s" % self._adas.status, True, status_col)
            display.blit(status_txt, (rp_x + 10, cy)); cy += 22
            if self._adas.acc_active:
                spd_txt = sb.render("Target: %d km/h  Gap: %.1fs" % (
                    int(self._adas.target_speed), self._adas.time_gap), True, (200, 230, 255))
                display.blit(spd_txt, (rp_x + 10, cy)); cy += 20
            cy += 5

        # Row 3: LKA/BSM Indicators
        pygame.draw.line(display, (60, 65, 80), (rp_x + 10, cy), (rp_x + rp_w - 10, cy), 1)
        cy += 10
        bsm_left  = self.adas_data.get("bsm_left", False)
        bsm_right = self.adas_data.get("bsm_right", False)
        bsm_alert = bsm_left or bsm_right
        bsm_col   = (255,100,0) if bsm_alert else (80,220,100)
        bsm_lbl   = fs.render("BSM: " + ("ALERT ⚠" if bsm_alert else "CLEAN"), True, bsm_col)
        display.blit(bsm_lbl, (rp_x + 10, cy))
        
        lka_txt_col = (255, 180, 0) if lka_act else ((40, 180, 80) if lka_en else (100, 100, 100))
        lka_lbl     = fs.render("LKA: " + ("ACTIVE" if lka_act else ("READY" if lka_en else "OFF")), True, lka_txt_col)
        display.blit(lka_lbl, (rp_x + rp_w - lka_lbl.get_width() - 10, cy))
        cy += 24
        
        trigger_src = self.adas_data.get("trigger_src", "")
        if trigger_src:
            src_s = fs.render("Source: %s" % trigger_src, True, (255,210,120))
            display.blit(src_s, (rp_x + 10, cy))
        cy += 26
        
        # LKA Mini-Graph (LatDev)
        graph_w = rp_w - 40; graph_h = 14
        gx = rp_x + 20; gy = cy
        pygame.draw.rect(display, (40, 45, 55), (gx, gy, graph_w, graph_h), border_radius=7)
        pygame.draw.line(display, (100, 105, 120), (gx + graph_w//2, gy), (gx + graph_w//2, gy + graph_h), 1)
        lat_dev = self.adas_data.get("lka_lat_dev", 0.0)
        dev_px = int(np.clip(lat_dev * 40, -graph_w//2, graph_w//2))
        bar_col = (255, 200, 0) if abs(lat_dev) > 0.4 else (80, 220, 100)
        if dev_px != 0:
            bx = gx + graph_w//2 if dev_px > 0 else gx + graph_w//2 + dev_px
            pygame.draw.rect(display, bar_col, (bx, gy + 2, abs(dev_px), graph_h - 4))

        # ----------------------------------------------------------------
        # Steering Contribution bar (bottom-left, f.py style)
        # ----------------------------------------------------------------
        man_steer = self.adas_data.get("manual_steer", 0.0)
        lka_steer = self.adas_data.get("lka_steer_now", 0.0)
        if abs(man_steer) > 0.005 or abs(lka_steer) > 0.005:
            bar_lx = 228; bar_ly = dh - 80; bar_lw = 180; bar_lh = 60
            bar_bg = pygame.Surface((bar_lw, bar_lh), pygame.SRCALPHA)
            bar_bg.fill((15, 18, 28, 180))
            display.blit(bar_bg, (bar_lx, bar_ly))
            pygame.draw.rect(display, (60, 65, 80), (bar_lx, bar_ly, bar_lw, bar_lh), 1)
            steer_fnt = pygame.font.Font(None, 18)
            hdr_s = steer_fnt.render("STEER CONTRIBUTION", True, (140, 145, 160))
            display.blit(hdr_s, (bar_lx + 4, bar_ly + 4))
            bcy = bar_ly + 20
            for label, val, col in [
                ("DRV", man_steer, (100, 160, 255)),
                ("LKA", lka_steer, (80, 220, 120))]:
                seg_w = int(abs(val) * 60)
                seg_x = (bar_lx + bar_lw // 2) if val >= 0 else (bar_lx + bar_lw // 2 - seg_w)
                if seg_w > 0:
                    pygame.draw.rect(display, col, (seg_x, bcy, seg_w, 10))
                lbl = steer_fnt.render("%s %+.2f" % (label, val), True, col)
                display.blit(lbl, (bar_lx + 4, bcy)); bcy += 14

        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # SIDE SCREEN BSM MIRROR BARS (Mirror Alert Style)
        # ----------------------------------------------------------------
        bar_w = 12
        bar_h = dh // 3
        bar_y = dh // 2 - bar_h // 2
        for side in ("left", "right"):
            alert = bsm_left if side == "left" else bsm_right
            # If steering toward threat, turn color to RED and pulse
            steer = self.adas_data.get("manual_steer", 0.0)
            is_critical = alert and ((side == "left" and steer < -0.15) or (side == "right" and steer > 0.15))
            
            if alert:
                col = (255, 40, 40) if is_critical else (255, 140, 0)
                alpha = 200 if (not is_critical or flash_on) else 80
                bx = 0 if side == "left" else dw - bar_w
                
                bsm_bar = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
                bsm_bar.fill((*col, alpha))
                display.blit(bsm_bar, (bx, bar_y))
                
                # Visual Glow / Arrow
                f_bsm = pygame.font.Font(None, 40)
                arr_chr = "◀" if side == "left" else "▶"
                arr_s = f_bsm.render(arr_chr, True, col)
                if not is_critical or (is_critical and flash_on):
                    ax = bx + bar_w + 10 if side == "left" else bx - arr_s.get_width() - 10
                    display.blit(arr_s, (ax, bar_y + bar_h // 2 - arr_s.get_height() // 2))
                
                if is_critical and flash_on:
                    crit_s = sb.render("CRITICAL BLIND SPOT!", True, (255, 50, 50))
                    cx_p = bx + 20 if side == "left" else bx - crit_s.get_width() - 20
                    display.blit(crit_s, (cx_p, bar_y - 20))

        # ----------------------------------------------------------------
        # Bottom-centre AEB alert bar
        # ----------------------------------------------------------------
        if aeb_state >= 1:
            bar_w = 420; bar_h = 34
            bar_x = dw // 2 - bar_w // 2; bar_y = dh - bar_h - 8
            bar_col  = (60, 40, 0) if aeb_state == 1 else (60, 10, 10)
            bar_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
            bar_surf.fill((*bar_col, 220))
            display.blit(bar_surf, (bar_x, bar_y))
            pygame.draw.rect(display, border_col, (bar_x, bar_y, bar_w, bar_h), 2)
            alert_txts = {1: "⚠  AEB WARNING — Threat Detected",
                          2: "🔶  BRAKING — Obstacle Ahead",
                          3: "🔴  EMERGENCY BRAKING"}
            alert_cols = {1: (255,200,0), 2: (255,130,0), 3: (255,40,40)}
            if flash_on or aeb_state >= 2:
                at = pygame.font.Font(None, 30).render(
                    alert_txts.get(aeb_state, "AEB"), True,
                    alert_cols.get(aeb_state, (255,255,255)))
                display.blit(at, (bar_x + bar_w // 2 - at.get_width() // 2,
                                  bar_y + bar_h // 2 - at.get_height() // 2))

        # Speed warning
        if self.adas_data.get("warning") and flash_on:
            sw = pygame.font.Font(None, 28).render("SPEEDING!", True, (255, 100, 0))
            display.blit(sw, (rp_x + rp_w // 2 - sw.get_width() // 2, rp_y + rp_h + 4))

        # Tesla dashboard
        if self.show_tesla_hud:
            self.tesla_dashboard.render(display, self.adas_data)


# ==============================================================================
# -- TeslaDashboard ------------------------------------------------------------
# ==============================================================================


class TeslaDashboard(object):
    def __init__(self, w, h):
        self.dim = (250, 250)
        self.pos = (w - 260, h - 260)
        self.surface = pygame.Surface(self.dim)
        self.nearby_actors = []
        self.scale = 10.0
        self.steering = 0.0
        self.road_lines = []

    def tick(self, world):
        self.nearby_actors = []
        vehicle = world.player
        if not vehicle:
            return
        self.steering = vehicle.get_control().steer
        v_loc = vehicle.get_location()
        all_actors = world.world.get_actors()
        vehicles   = all_actors.filter("vehicle.*")
        walkers    = all_actors.filter("walker.*")
        lights     = all_actors.filter("traffic.traffic_light")
        targets    = list(vehicles) + list(walkers) + list(lights)
        for actor in targets:
            if actor.id == vehicle.id: continue
            a_loc = actor.get_location()
            if abs(a_loc.z - v_loc.z) > 4.0: continue
            dist = math.sqrt((v_loc.x - a_loc.x)**2 + (v_loc.y - a_loc.y)**2)
            if dist < 50:
                forward_vec = vehicle.get_transform().get_forward_vector()
                right_vec   = vehicle.get_transform().get_right_vector()
                diff_x = a_loc.x - v_loc.x; diff_y = a_loc.y - v_loc.y
                rel_x  = diff_x * forward_vec.x + diff_y * forward_vec.y
                rel_y  = diff_x * right_vec.x   + diff_y * right_vec.y
                a_type = "other"
                if "walker"       in actor.type_id: a_type = "walker"
                elif "vehicle"    in actor.type_id: a_type = "vehicle"
                elif "traffic_light" in actor.type_id: a_type = "light"
                self.nearby_actors.append((rel_x, rel_y, dist, a_type, actor.id, actor))

        self.road_lines = []
        try:
            v_trans  = vehicle.get_transform()
            fwd_vec  = v_trans.get_forward_vector()
            right_vec = v_trans.get_right_vector()
            current_wp = world.map.get_waypoint(v_loc)
            visited = set(); queue = collections.deque([(current_wp, 0)])
            max_dist = 40.0; step_size = 4.0
            while queue:
                wp, dist = queue.popleft()
                if dist > max_dist: continue
                if wp.id in visited: continue
                visited.add(wp.id)
                self.process_lane_lines(wp, v_loc, fwd_vec, right_vec, step_size)
                for n_wp in wp.next(step_size):
                    queue.append((n_wp, dist + step_size))
                if dist < 20.0:
                    ll = wp.get_left_lane()
                    rl = wp.get_right_lane()
                    if ll and ll.lane_type == carla.LaneType.Driving: queue.append((ll, dist))
                    if rl and rl.lane_type == carla.LaneType.Driving: queue.append((rl, dist))
        except Exception:
            pass

    def process_lane_lines(self, wp, v_loc, fwd, right, step):
        wp_trans  = wp.transform
        wp_right  = wp_trans.get_right_vector()
        hw        = wp.lane_width * 0.5
        l1 = wp_trans.location - (wp_right * hw)
        r1 = wp_trans.location + (wp_right * hw)
        next_wps  = wp.next(step)
        if not next_wps: return
        n_wp   = next_wps[0]; n_trans = n_wp.transform
        n_right = n_trans.get_right_vector(); nhw = n_wp.lane_width * 0.5
        l2 = n_trans.location - (n_right * nhw)
        r2 = n_trans.location + (n_right * nhw)
        p1l = self.world_to_screen(l1, v_loc, fwd, right)
        p2l = self.world_to_screen(l2, v_loc, fwd, right)
        p1r = self.world_to_screen(r1, v_loc, fwd, right)
        p2r = self.world_to_screen(r2, v_loc, fwd, right)
        if p1l and p2l: self.road_lines.append((p1l, p2l, wp.lane_type))
        if p1r and p2r: self.road_lines.append((p1r, p2r, wp.lane_type))

    def world_to_screen(self, loc, v_loc, fwd, right):
        dx = loc.x - v_loc.x; dy = loc.y - v_loc.y
        rel_fwd   = dx * fwd.x   + dy * fwd.y
        rel_right = dx * right.x + dy * right.y
        if rel_fwd < -5.0: return None
        cx = self.dim[0] // 2; cy = self.dim[1] - 50
        return (int(cx + rel_right * self.scale), int(cy - rel_fwd * self.scale))

    def render(self, display, adas_data):
        fade = pygame.Surface(self.dim); fade.fill((30, 30, 30)); fade.set_alpha(35)
        self.surface.blit(fade, (0, 0))
        cx = self.dim[0] // 2; cy = self.dim[1] - 50
        for p1, p2, l_type in self.road_lines:
            if ((0 < p1[0] < self.dim[0] and 0 < p1[1] < self.dim[1]) or
                    (0 < p2[0] < self.dim[0] and 0 < p2[1] < self.dim[1])):
                color = (220, 220, 220) if str(l_type) == "Driving" else (180, 180, 180)
                w = 2 if str(l_type) == "Driving" else 1
                pygame.draw.line(self.surface, color, p1, p2, w)
                pygame.draw.circle(self.surface, color, p1, w)
        steer    = self.steering
        curve_pts = []
        for i in range(15):
            d = i * 3.0; offset = steer * (d**2) * 0.03
            curve_pts.append((cx + offset * self.scale * 0.1, cy - d * self.scale))
        if len(curve_pts) > 1:
            pygame.draw.lines(self.surface, (0, 150, 255), False, curve_pts, 3)
        radar_points = adas_data.get("radar_points", [])
        for ring_m in [10, 20, 30]:
            ring_px = int(ring_m * self.scale)
            rr = pygame.Rect(cx - ring_px, cy - ring_px, ring_px * 2, ring_px * 2)
            pygame.draw.arc(self.surface, (50, 70, 50), rr, 0, math.pi, 1)
        if radar_points:
            front_pts = [p for p in radar_points if abs(p.get("azi", 90)) < 40]
            if front_pts:
                closest  = min(front_pts, key=lambda p: p["dist"])
                r_dist   = closest["dist"]; r_azi = math.radians(closest.get("azi", 0))
                r_x = r_dist * math.cos(r_azi); r_y = r_dist * math.sin(r_azi)
                sx  = int(cx + r_y * self.scale); sy  = int(cy - r_x * self.scale)
                if 0 < sx < self.dim[0] and 0 < sy < self.dim[1]:
                    sz = 5
                    pygame.draw.polygon(self.surface, (0, 220, 100), [
                        (sx, sy - sz), (sx + sz, sy), (sx, sy + sz), (sx - sz, sy)])
        pygame.draw.polygon(self.surface, (0, 200, 255), [
            (cx, cy - 20), (cx - 12, cy + 12), (cx + 12, cy + 12)])
        threat_actor = adas_data.get("threat_actor")
        threat_id    = threat_actor.id if threat_actor else -1
        for rel_x, rel_y, dist, a_type, a_id, raw_actor in self.nearby_actors:
            sx = cx + (rel_y * self.scale); sy = cy - (rel_x * self.scale)
            sx -= steer * rel_x * 0.2
            sx = int(sx); sy = int(sy)
            if 0 < sx < self.dim[0] and 0 < sy < self.dim[1]:
                color    = (200, 200, 200)
                is_threat = (a_id == threat_id)
                if is_threat or (a_id in [adas_data.get("bsm_left_id"), adas_data.get("bsm_right_id")]):
                    color = (255, 50, 50)
                if a_type == "vehicle":
                    wp = 2.4 * self.scale; lp = 5.0 * self.scale
                    rect = pygame.Rect(sx - wp/2, sy - lp/2, wp, lp)
                    pygame.draw.rect(self.surface, color, rect)
                    if is_threat: pygame.draw.rect(self.surface, (255, 255, 0), rect, 2)
                elif a_type == "walker":
                    pygame.draw.circle(self.surface, (255, 255, 255), (sx, sy), 4)
                elif a_type == "light":
                    state = str(raw_actor.get_state()).split(".")[-1]
                    lc    = (0, 255, 0) if state == "Green" else ((255, 0, 0) if state == "Red" else (255, 255, 0))
                    pygame.draw.circle(self.surface, lc, (sx, sy), 6)
        aeb_st      = adas_data.get("aeb_state", 0)
        panel_brd   = (60, 100, 60) if aeb_st == 0 else ((200, 140, 0) if aeb_st == 1 else (200, 50, 50))
        pygame.draw.rect(self.surface, panel_brd, (0, 0, 250, 250), 2)
        trig = adas_data.get("trigger_src", "")
        title_txt = ("ADAS FUSION  [%s]" % trig) if trig else "ADAS FUSION VIEW"
        lbl = pygame.font.Font(None, 18).render(title_txt, True,
              (180, 220, 180) if aeb_st == 0 else (255, 200, 100))
        self.surface.blit(lbl, (10, 10))
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font; self.dim = dim; self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds   = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split("\n")
        self.font = font; self.line_space = 18
        self.dim  = (780, len(lines) * self.line_space + 12)
        self.pos  = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor  = None
        self.history = []
        self._parent = parent_actor
        self.hud     = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self: return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification("Collision with %r" % actor_type)
        impulse   = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor (LDW-aware) -------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Lane invasion sensor — enhanced with LDW departure detection (f.py)."""
    def __init__(self, parent_actor, hud):
        self.sensor   = None
        self._parent  = parent_actor
        self.hud      = hud
        self._ldw_active = True
        self._last_invasion_time = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self: return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))
        if self._ldw_active:
            # Basic junction filter: check if parent is at junction
            wp = self._parent.get_world().get_map().get_waypoint(
                self._parent.get_location())
            if wp and wp.is_junction:
                return
            dangerous_types = {carla.LaneMarkingType.Solid,
                               carla.LaneMarkingType.SolidBroken,
                               carla.LaneMarkingType.BrokenSolid,
                               carla.LaneMarkingType.SolidSolid}
            for marking in event.crossed_lane_markings:
                if marking.type in dangerous_types:
                    self.hud.notification("⚠ LDW: LANE DEPARTURE DETECTED", seconds=3.0)
                    break


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor  = None
        self._parent = parent_actor
        self.lat = 0.0; self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.gnss")
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self: return
        self.lat = event.latitude; self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor       = None
        self._parent      = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope    = (0.0, 0.0, 0.0)
        self.compass      = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.imu")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self: return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor (abc.py version — supports front + rear roles) ----------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor, role="front"):
        self.sensor         = None
        self._parent        = parent_actor
        self.role           = role
        self.velocity_range = 7.5
        self.show_radar     = False
        self.points_to_draw = []
        self.radar_threat   = None
        self.data           = []   # raw detection list for ACC rear use
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find("sensor.other.radar")

        if role == "front":
            bp.set_attribute("horizontal_fov", str(35))
            bp.set_attribute("vertical_fov",   str(30))
            transform = carla.Transform(carla.Location(x=2.0, z=1.0), carla.Rotation(pitch=0))
        else:  # rear
            bp.set_attribute("horizontal_fov", str(30))
            bp.set_attribute("vertical_fov",   str(20))
            transform = carla.Transform(carla.Location(x=-2.0, z=1.0), carla.Rotation(pitch=0, yaw=180))

        self.sensor = world.spawn_actor(bp, transform, attach_to=self._parent)
        weak_self   = weakref.ref(self)
        self.sensor.listen(lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    def toggle_display(self):
        self.show_radar = not self.show_radar

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self: return
        current_rot     = radar_data.transform.rotation
        self.points_to_draw = []
        self.radar_threat   = None
        self.data           = []
        closest_dist        = 999.0

        for detect in radar_data:
            azi  = math.degrees(detect.azimuth)
            alt  = math.degrees(detect.altitude)
            dist = detect.depth
            x    = dist * math.cos(detect.azimuth) * math.cos(detect.altitude)
            y    = dist * math.sin(detect.azimuth) * math.cos(detect.altitude)
            z    = dist * math.sin(detect.altitude)
            if dist > 2.0:
                self.points_to_draw.append({"azi": azi, "alt": alt, "dist": dist, "vel": detect.velocity})
                self.data.append({"azi": azi, "alt": alt, "dist": dist, "vel": detect.velocity})
                steer    = self._parent.get_control().steer
                y_shifted = y - (x * steer * 0.5)
                if x > 0 and abs(y_shifted) < 2.2 and (-1.5 < z < 5.0):
                    if detect.velocity > 1.8 and dist < closest_dist:
                        closest_dist       = dist
                        self.radar_threat  = {"dist": dist, "rel_vel": detect.velocity}

        if not self.show_radar:
            return
        for detect in radar_data:
            azi = math.degrees(detect.azimuth); alt = math.degrees(detect.altitude)
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(carla.Location(),
                carla.Rotation(pitch=current_rot.pitch + alt,
                               yaw=current_rot.yaw   + azi,
                               roll=current_rot.roll)).transform(fw_vec)

            def clamp(mn, mx, val): return max(mn, min(val, mx))
            nv = detect.velocity / self.velocity_range
            r  = int(clamp(0.0, 1.0, 1.0 - nv) * 255.0)
            g  = int(clamp(0.0, 1.0, 1.0 - abs(nv)) * 255.0)
            b  = int(abs(clamp(-1.0, 0.0, -1.0 - nv)) * 255.0)
            self.debug.draw_point(radar_data.transform.location + fw_vec,
                                  size=0.075, life_time=0.06, persistent_lines=False,
                                  color=carla.Color(r, g, b))


# ==============================================================================
# -- ObstacleSensor ------------------------------------------------------------
# ==============================================================================


class ObstacleSensor(object):
    def __init__(self, parent_actor):
        self.sensor   = None
        self._parent  = parent_actor
        self.distance = None
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.obstacle")
        # Reduced distance to 40m and hit_radius to 0.4 to prevent tunnel ceiling/wall false triggering
        bp.set_attribute("distance",     "40")
        bp.set_attribute("hit_radius",   "0.4")
        bp.set_attribute("only_dynamics","False")
        self.sensor = world.spawn_actor(bp,
            carla.Transform(carla.Location(x=2.0, z=1.0)),
            attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: ObstacleSensor._on_obstacle(weak_self, event))

    @staticmethod
    def _on_obstacle(weak_self, event):
        self = weak_self()
        if not self: return
        self.distance = event.distance


# ==============================================================================
# -- SemanticLidarSensor -------------------------------------------------------
# ==============================================================================


class SemanticLidarSensor(object):
    def __init__(self, parent_actor):
        self.sensor      = None
        self._parent     = parent_actor
        self.dist        = None
        self.dist_rear   = None
        self.object_type = "None"
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        bp.set_attribute("range",            "50")
        bp.set_attribute("channels",         "64")
        bp.set_attribute("points_per_second","130000")
        bp.set_attribute("upper_fov",        "5")
        bp.set_attribute("lower_fov",        "-30")
        self.sensor = world.spawn_actor(bp,
            carla.Transform(carla.Location(x=0.0, z=1.8)),
            attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda pc: SemanticLidarSensor._on_lidar(weak_self, pc))

    @staticmethod
    def _on_lidar(weak_self, point_cloud):
        self = weak_self()
        if not self: return
        dtype = np.dtype([
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("cos", np.float32), ("idx", np.uint32), ("tag", np.uint32)])
        data = np.frombuffer(point_cloud.raw_data, dtype=dtype)
        my_id         = self._parent.id
        mask_not_self = (data["idx"] != my_id)
        tags          = data["tag"]
        surface_tags  = [0, 6, 7, 8, 14, 19, 21, 22, 24]
        mask_obs      = np.isin(tags, surface_tags, invert=True)
        x = data["x"]; y = data["y"]; z = data["z"]
        v     = self._parent.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        # Strict lateral limits. Must not exceed lane half-width (1.75m) to avoid adjacent lane lockups
        limit_y = 1.4 if speed < 30 else (1.2 if speed < 60 else 0.9)
        # Cut vertical limit for tunnel ceiling/pillars (z > 2.0 is likely bridge/tunnel ceiling)
        mask_geo  = (x > 0.0) & (x < 35.0) & (np.abs(y) < limit_y)
        vw_mask   = np.isin(tags, [4, 10])
        mask_ht   = (z < 2.5) & ((z > 0.2) | vw_mask)
        final_mask = mask_not_self & mask_obs & mask_geo & mask_ht
        mask_geo_r = (x < -0.5) & (np.abs(y) < 2.2)
        final_mask_r = mask_not_self & mask_obs & mask_geo_r & mask_ht
        self.dist        = float(np.min(x[final_mask]))     if np.any(final_mask) else None
        self.object_type = "Obstacle" if self.dist is not None else "None"
        self.dist_rear   = float(np.abs(np.max(x[final_mask_r]))) if np.any(final_mask_r) else None

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor.destroy()
            except Exception:
                pass
            self.sensor = None


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor    = None
        self.surface   = None
        self._parent   = parent_actor
        self.hud       = hud
        self.recording = False
        bound_y        = 0.5 + self._parent.bounding_box.extent.y
        Attachment     = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6,  z=1.7)),                           Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),                     Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)),Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)),                 Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ["sensor.camera.rgb",                  cc.Raw, "Camera RGB", {}],
            ["sensor.camera.depth",                cc.Raw, "Camera Depth (Raw)", {}],
            ["sensor.camera.depth",                cc.Depth, "Camera Depth (Gray Scale)", {}],
            ["sensor.camera.depth",                cc.LogarithmicDepth, "Camera Depth (Log Gray)", {}],
            ["sensor.camera.semantic_segmentation",cc.Raw, "Camera Sem Seg (Raw)", {}],
            ["sensor.camera.semantic_segmentation",cc.CityScapesPalette, "Camera Sem Seg (CityScapes)", {}],
            ["sensor.lidar.ray_cast",              None,   "Lidar (Ray-Cast)", {"range": "50"}],
            ["sensor.camera.dvs",                  cc.Raw, "Dynamic Vision Sensor", {}],
        ]
        world       = self._parent.get_world()
        bp_library  = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(hud.dim[0]))
                bp.set_attribute("image_size_y", str(hud.dim[1]))
                if bp.has_attribute("gamma"):
                    bp.set_attribute("gamma", str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith("sensor.lidar"):
                self.lidar_range = 50
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == "range": self.lidar_range = float(attr_value)
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy(); self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify: self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        if self.surface is not None: display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self: return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data  = np.fabs(lidar_data).astype(np.int32)
            lidar_data  = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img   = np.zeros(lidar_img_size, dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith("sensor.camera.dvs"):
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, Red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3][:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        # Force connection entirely to 127.0.0.1:2000 as requested
        import os
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)

        display_flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        if hasattr(args, "fullscreen") and args.fullscreen:
            display_flags |= pygame.FULLSCREEN

        display = pygame.display.set_mode((args.width, args.height), display_flags)
        pygame.display.set_caption("CARLA ADAS Fusion — Virtual Vahana 2026")
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud   = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if world and world.recording_enabled:
            client.stop_recorder()
        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description="CARLA ADAS Fusion — Virtual Vahana 2026")
    argparser.add_argument("-v", "--verbose", action="store_true", dest="debug",
                           help="print debug information")
    argparser.add_argument("--host", metavar="H", default="127.0.0.1",
                           help="IP of the host server (default: 127.0.0.1)")
    argparser.add_argument("-p", "--port", metavar="P", default=2000, type=int,
                           help="TCP port to listen to (default: 2000)")
    argparser.add_argument("-a", "--autopilot", action="store_true",
                           help="enable autopilot")
    argparser.add_argument("--res", metavar="WIDTHxHEIGHT", default="1280x720",
                           help="window resolution (default: 1280x720)")
    argparser.add_argument("--filter", metavar="PATTERN", default="vehicle.*",
                           help="actor filter (default: \"vehicle.*\")")
    argparser.add_argument("--rolename", metavar="NAME", default="hero",
                           help="actor role name (default: \"hero\")")
    argparser.add_argument("--gamma", default=2.2, type=float,
                           help="Gamma correction of the camera (default: 2.2)")
    argparser.add_argument("--fullscreen", action="store_true",
                           help="enable fullscreen mode")
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)
    logging.info("listening to server %s:%s", args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    main()