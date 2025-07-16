#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv
import os
from datetime import datetime

def save_evaluation_to_csv(evaluation_report, cross_track_list, detections, end_reason, logger):
    """Save evaluation data to a single CSV file."""
    try:
        # Create folder for saving data
        data_folder = "evaluation_data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            logger.info(f"📁 Created data saving folder: {data_folder}")
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get performance metrics
        traj_perf = evaluation_report['trajectory_performance']
        safety_perf = evaluation_report['safety_performance']
        eff_perf = evaluation_report['efficiency_performance']
        comfort_perf = evaluation_report['comfort_performance']
        perception_perf = evaluation_report['perception_performance']
        decision_perf = evaluation_report['decision_performance']
        overall_score = evaluation_report['overall_score']
        
        # Calculate driving skill score
        lane_score = max(0, 100 - traj_perf['mean_cross_track_error'] * 100)
        speed_score = max(0, 100 - traj_perf['mean_speed_error'] * 15)
        smooth_score = max(0, 100 - eff_perf['mean_jerk'] * 30)
        steering_score = max(0, 100 - comfort_perf['mean_steering_rate'] * 80)
        driving_skill_score = (lane_score * 0.35 + speed_score * 0.25 + smooth_score * 0.25 + steering_score * 0.15)
        driving_skill_score = min(100, max(0, driving_skill_score))
        
        # Calculate trajectory statistics
        cross_track_stats = {}
        if cross_track_list:
            cross_track_stats = {
                'trajectory_data_points': len(cross_track_list),
                'lateral_deviation_min_pixels': min(cross_track_list),
                'lateral_deviation_max_pixels': max(cross_track_list),
                'lateral_deviation_mean_pixels': round(np.mean(cross_track_list), 2),
                'lateral_deviation_std_pixels': round(np.std(cross_track_list), 2),
            }
        else:
            cross_track_stats = {
                'trajectory_data_points': 0,
                'lateral_deviation_min_pixels': 0,
                'lateral_deviation_max_pixels': 0,
                'lateral_deviation_mean_pixels': 0,
                'lateral_deviation_std_pixels': 0,
            }
        
        # Prepare complete CSV data
        evaluation_data = {
            # Basic information
            'timestamp': timestamp,
            'end_reason': end_reason,
            'overall_score': round(overall_score, 2),
            'driving_skill_score': round(driving_skill_score, 2),
            'simulation_frames': len(cross_track_list),
            'actual_simulation_time_seconds': round(eff_perf['total_time'], 2),
            'average_fps': round(len(cross_track_list)/max(eff_perf['total_time'], 1), 2),
            'detected_objects_total': sum(len(dets) if isinstance(dets, list) else 0 for dets in [detections]),
            
            # Trajectory tracking performance
            'mean_lateral_deviation_m': round(traj_perf['mean_cross_track_error'], 4),
            'max_lateral_deviation_m': round(traj_perf['max_cross_track_error'], 4),
            'lateral_deviation_std_m': round(traj_perf['std_cross_track_error'], 4),
            'mean_heading_error_rad': round(traj_perf['mean_heading_error'], 4),
            'mean_heading_error_degrees': round(np.degrees(traj_perf['mean_heading_error']), 2),
            'max_heading_error_rad': round(traj_perf['max_heading_error'], 4),
            'mean_speed_error_ms': round(traj_perf['mean_speed_error'], 3),
            'max_speed_error_ms': round(traj_perf['max_speed_error'], 3),
            
            # Safety performance
            'collision_count': safety_perf['collision_count'],
            'collision_rate_per1000frames': round(safety_perf['collision_rate'], 4),
            'near_collision_count': safety_perf['near_collision_count'],
            'near_collision_rate_per1000frames': round(safety_perf['near_collision_rate'], 4),
            'traffic_light_violations': safety_perf['traffic_light_violations'],
            'stop_sign_violations': safety_perf['stop_sign_violations'],
            'lane_change_violations': safety_perf['lane_change_violations'],
            'mean_safety_distance_m': round(safety_perf['mean_min_distance_to_objects'], 3),
            'min_safety_distance_m': round(safety_perf['min_distance_to_objects'], 3),
            
            # Efficiency performance
            'mean_speed_ms': round(eff_perf['mean_speed'], 3),
            'mean_speed_kmh': round(eff_perf['mean_speed'] * 3.6, 2),
            'max_speed_ms': round(eff_perf['max_speed'], 3),
            'max_speed_kmh': round(eff_perf['max_speed'] * 3.6, 2),
            'mean_acceleration_ms2': round(eff_perf['mean_acceleration'], 3),
            'max_acceleration_ms2': round(eff_perf['max_acceleration'], 3),
            'mean_jerk_ms3': round(eff_perf['mean_jerk'], 3),
            'max_jerk_ms3': round(eff_perf['max_jerk'], 3),
            'total_distance_m': round(eff_perf['total_distance'], 2),
            'route_completion_rate_percent': round(eff_perf['route_completion_rate'] * 100, 1),
            'average_travel_speed_ms': round(eff_perf['average_speed'], 3),
            
            # Comfort performance
            'mean_lateral_acceleration_ms2': round(comfort_perf['mean_lateral_acceleration'], 3),
            'max_lateral_acceleration_ms2': round(comfort_perf['max_lateral_acceleration'], 3),
            'mean_steering_angle_rad': round(comfort_perf['mean_steering_angle'], 4),
            'mean_steering_angle_degrees': round(np.degrees(comfort_perf['mean_steering_angle']), 2),
            'max_steering_angle_rad': round(comfort_perf['max_steering_angle'], 4),
            'max_steering_angle_degrees': round(np.degrees(comfort_perf['max_steering_angle']), 2),
            'mean_steering_rate_rads': round(comfort_perf['mean_steering_rate'], 4),
            'max_steering_rate_rads': round(comfort_perf['max_steering_rate'], 4),
            'mean_throttle_change_rate': round(comfort_perf['mean_throttle_change'], 4),
            'max_throttle_change_rate': round(comfort_perf['max_throttle_change'], 4),
            'mean_brake_change_rate': round(comfort_perf['mean_brake_change'], 4),
            'max_brake_change_rate': round(comfort_perf['max_brake_change'], 4),
            
            # Perception performance
            'object_detection_accuracy_percent': round(perception_perf['mean_object_detection_accuracy'] * 100, 1),
            'lane_detection_accuracy_percent': round(perception_perf['mean_lane_detection_accuracy'] * 100, 1),
            'traffic_light_detection_accuracy_percent': round(perception_perf['mean_traffic_light_detection_accuracy'] * 100, 1),
            
            # Decision performance
            'mean_decision_time_ms': round(decision_perf['mean_decision_time'] * 1000, 2),
            'max_decision_time_ms': round(decision_perf['max_decision_time'] * 1000, 2),
            'overtaking_success_rate_percent': round(decision_perf['overtaking_success_rate'] * 100, 1),
            'lane_change_success_rate_percent': round(decision_perf['lane_change_success_rate'] * 100, 1),
        }
        
        # Merge trajectory statistics
        evaluation_data.update(cross_track_stats)
        
        # Save as single CSV file
        csv_file = os.path.join(data_folder, f"driving_evaluation_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=evaluation_data.keys())
            writer.writeheader()
            writer.writerow(evaluation_data)
        
        logger.info(f"💾 Complete evaluation data saved to: {csv_file}")
        logger.info(f"📁 File location: {os.path.abspath(csv_file)}")
        logger.info(f"📊 Contains {len(evaluation_data)} evaluation metrics")
        
    except Exception as e:
        logger.error(f"❌ Error saving CSV data: {e}")

def generate_evaluation_report(eval_metrics, cross_track_list, detections, logger, end_reason="Simulation ended"):
    """Generate and print detailed evaluation report."""
    # Get complete evaluation report
    evaluation_report = eval_metrics.get_full_report()
    
    # Save CSV data
    save_evaluation_to_csv(evaluation_report, cross_track_list, detections, end_reason, logger)
    
    # Print evaluation report
    logger.info("=" * 60)
    logger.info("🚗 Autonomous Driving Performance Evaluation Report")
    logger.info("=" * 60)
    logger.info(f"🏁 End reason: {end_reason}")
    logger.info("")
    
    # Overall score
    overall_score = evaluation_report['overall_score']
    logger.info(f"🏆 Overall score: {overall_score:.1f}/100")
    
    # Calculate driving skill score (specialized technical evaluation)
    traj_perf = evaluation_report['trajectory_performance']
    comfort_perf = evaluation_report['comfort_performance']
    eff_perf = evaluation_report['efficiency_performance']
    
    # Driving skill score calculation
    lane_score = max(0, 100 - traj_perf['mean_cross_track_error'] * 100)
    speed_score = max(0, 100 - traj_perf['mean_speed_error'] * 15)
    smooth_score = max(0, 100 - eff_perf['mean_jerk'] * 30)
    steering_score = max(0, 100 - comfort_perf['mean_steering_rate'] * 80)
    
    driving_skill_score = (lane_score * 0.35 + speed_score * 0.25 + smooth_score * 0.25 + steering_score * 0.15)
    driving_skill_score = min(100, max(0, driving_skill_score))
    
    logger.info(f"🎯 Driving skill score: {driving_skill_score:.1f}/100")
    
    if overall_score >= 90:
        score_level = "Excellent 🥇"
    elif overall_score >= 80:
        score_level = "Good 🥈"
    elif overall_score >= 70:
        score_level = "Pass 🥉"
    else:
        score_level = "Needs Improvement ⚠️"
    
    if driving_skill_score >= 90:
        skill_level = "Professional 🏎️"
    elif driving_skill_score >= 80:
        skill_level = "Skilled 🚗"
    elif driving_skill_score >= 70:
        skill_level = "Average 📚"
    else:
        skill_level = "Novice 🔰"
    
    logger.info(f"📈 Overall rating: {score_level}")
    logger.info(f"🏁 Driving skill: {skill_level}")
    logger.info("")
    
    # Trajectory tracking performance
    traj_perf = evaluation_report['trajectory_performance']
    logger.info("🛣️ Trajectory tracking performance:")
    logger.info(f"   • Mean lateral deviation: {traj_perf['mean_cross_track_error']:.3f}m")
    logger.info(f"   • Max lateral deviation: {traj_perf['max_cross_track_error']:.3f}m")
    logger.info(f"   • Mean heading error: {traj_perf['mean_heading_error']:.3f}rad ({np.degrees(traj_perf['mean_heading_error']):.1f}°)")
    logger.info(f"   • Mean speed error: {traj_perf['mean_speed_error']:.2f}m/s")
    logger.info("")
    
    # Safety performance
    safety_perf = evaluation_report['safety_performance']
    logger.info("🛡️ Safety performance:")
    logger.info(f"   • Collision count: {safety_perf['collision_count']}")
    logger.info(f"   • Near collision count: {safety_perf['near_collision_count']}")
    logger.info(f"   • Traffic light violations: {safety_perf['traffic_light_violations']}")
    logger.info(f"   • Min distance to objects: {safety_perf['min_distance_to_objects']:.2f}m")
    logger.info(f"   • Mean safety distance: {safety_perf['mean_min_distance_to_objects']:.2f}m")
    logger.info("")
    
    # Efficiency performance
    eff_perf = evaluation_report['efficiency_performance']
    logger.info("⚡ Efficiency performance:")
    logger.info(f"   • Mean speed: {eff_perf['mean_speed']:.2f}m/s ({eff_perf['mean_speed']*3.6:.1f}km/h)")
    logger.info(f"   • Max speed: {eff_perf['max_speed']:.2f}m/s ({eff_perf['max_speed']*3.6:.1f}km/h)")
    logger.info(f"   • Mean acceleration: {eff_perf['mean_acceleration']:.2f}m/s²")
    logger.info(f"   • Mean jerk: {eff_perf['mean_jerk']:.2f}m/s³")
    logger.info(f"   • Total distance: {eff_perf['total_distance']:.1f}m")
    logger.info(f"   • Total time: {eff_perf['total_time']:.1f}s")
    logger.info(f"   • Route completion rate: {eff_perf['route_completion_rate']*100:.1f}%")
    logger.info("")
    
    # Comfort performance
    comfort_perf = evaluation_report['comfort_performance']
    logger.info("😌 Comfort performance:")
    logger.info(f"   • Mean lateral acceleration: {comfort_perf['mean_lateral_acceleration']:.2f}m/s²")
    logger.info(f"   • Max lateral acceleration: {comfort_perf['max_lateral_acceleration']:.2f}m/s²")
    logger.info(f"   • Mean steering angle: {comfort_perf['mean_steering_angle']:.3f}rad ({np.degrees(comfort_perf['mean_steering_angle']):.1f}°)")
    logger.info(f"   • Mean steering rate: {comfort_perf['mean_steering_rate']:.3f}rad/s")
    logger.info("")
    
    # Perception performance
    perception_perf = evaluation_report['perception_performance']
    logger.info("👁️ Perception performance:")
    logger.info(f"   • Object detection accuracy: {perception_perf['mean_object_detection_accuracy']*100:.1f}%")
    logger.info(f"   • Lane detection accuracy: {perception_perf['mean_lane_detection_accuracy']*100:.1f}%")
    logger.info(f"   • Traffic light detection accuracy: {perception_perf['mean_traffic_light_detection_accuracy']*100:.1f}%")
    logger.info("")
    
    # Decision performance
    decision_perf = evaluation_report['decision_performance']
    logger.info("🧠 Decision performance:")
    logger.info(f"   • Mean decision time: {decision_perf['mean_decision_time']*1000:.1f}ms")
    logger.info(f"   • Max decision time: {decision_perf['max_decision_time']*1000:.1f}ms")
    logger.info("")
    
    # Statistics
    logger.info("📈 Statistics:")
    logger.info(f"   • Simulation frames: {len(cross_track_list)}")
    logger.info(f"   • Actual simulation time: {eff_perf['total_time']:.1f}s")
    logger.info(f"   • Average frame rate: {len(cross_track_list)/max(eff_perf['total_time'], 1):.1f} FPS")
    logger.info(f"   • Total detected objects: {sum(len(dets) if isinstance(dets, list) else 0 for dets in [detections])}")
    logger.info("=" * 60)
    
    # Generate improvement suggestions
    logger.info("💡 Improvement suggestions:")
    
    suggestions_count = 0
    
    # Lateral deviation suggestions (lower threshold)
    if traj_perf['mean_cross_track_error'] > 0.05:  # above 5cm
        logger.info("   • Suggest improving lane tracking algorithm to reduce lateral deviation")
        suggestions_count += 1
    
    # Heading deviation suggestions (new addition)
    if traj_perf['mean_heading_error'] > 0.1:  # above ~5.7 degrees
        logger.info("   • Suggest optimizing heading control to reduce vehicle oscillation")
        suggestions_count += 1
    
    # Speed tracking suggestions
    if traj_perf['mean_speed_error'] > 1.0:  # above 1m/s deviation
        logger.info("   • Suggest improving speed tracking controller to enhance speed stability")
        suggestions_count += 1
    
    # Smooth driving suggestions (based on jerk rate)
    if eff_perf['mean_jerk'] > 10.0:  # excessive jerk
        logger.info("   • Suggest optimizing acceleration control to improve driving smoothness")
        suggestions_count += 1
    
    # Steering smoothness suggestions
    if comfort_perf['mean_steering_rate'] > 1.5:  # steering changes too fast
        logger.info("   • Suggest optimizing steering control to reduce abrupt steering actions")
        suggestions_count += 1
    
    # Lateral acceleration suggestions (lower threshold)
    if comfort_perf['mean_lateral_acceleration'] > 1.5:
        logger.info("   • Suggest reducing lateral acceleration to improve ride comfort")
        suggestions_count += 1
    
    # Safety distance suggestions
    if safety_perf['mean_min_distance_to_objects'] < 5.0:
        logger.info("   • Suggest increasing safety distance from other objects")
        suggestions_count += 1
    
    # Collision and violation suggestions
    if safety_perf['collision_count'] > 0:
        logger.info("   • Suggest enhancing collision detection and avoidance capabilities")
        suggestions_count += 1
    
    if safety_perf['traffic_light_violations'] > 0:
        logger.info("   • Suggest improving traffic signal recognition and compliance")
        suggestions_count += 1
    
    # Perception accuracy suggestions
    if perception_perf['mean_object_detection_accuracy'] < 0.85:
        logger.info("   • Suggest improving object detection algorithm accuracy")
        suggestions_count += 1
    
    if perception_perf['mean_lane_detection_accuracy'] < 0.9:
        logger.info("   • Suggest enhancing lane detection algorithm performance")
        suggestions_count += 1
    
    # If no specific suggestions, give general suggestions
    if suggestions_count == 0:
        if overall_score < 90:
            logger.info("   • Continue optimizing all performance metrics to achieve excellent level")
            logger.info("   • Focus on improving driving skill score")
            suggestions_count += 2
    
    if suggestions_count == 0:
        logger.info("   • Current performance is excellent, keep it up! 🎉")
    
    logger.info("🎯 Evaluation complete!")
    logger.info("=" * 60) 