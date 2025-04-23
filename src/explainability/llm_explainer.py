#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
from pathlib import Path

class LLMExplainer:
    """Class to generate natural language explanations for driving decisions using LLMs."""
    
    def __init__(self, config, model_type='local'):
        """
        Initialize the LLM explainer.
        
        Args:
            config: Configuration dictionary for the explainer.
            model_type: Type of LLM to use ('local', 'openai', 'anthropic').
        """
        self.config = config
        self.model_type = model_type
        self.last_explanation_time = 0
        self.explanation_cooldown = 0.5  # seconds, to avoid generating explanations too frequently
        
        # Initialize the model based on type
        if model_type == 'local':
            self._init_local_model(config.get('local', {}))
        elif model_type == 'openai':
            self._init_openai_model(config.get('openai', {}))
        elif model_type == 'anthropic':
            self._init_anthropic_model(config.get('anthropic', {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load prompt template
        self.prompt_template = config.get('prompt_template', self._default_prompt_template())
        
        print(f"LLM explainer initialized with model type: {model_type}")
    
    def _init_local_model(self, config):
        """Initialize local LLM model."""
        try:
            from llama_cpp import Llama
            
            model_path = config.get('model_path', 'models/llama-2-7b-chat.gguf')
            context_length = config.get('context_length', 2048)
            temperature = config.get('temperature', 0.3)
            top_p = config.get('top_p', 0.9)
            
            # If model path is specified but doesn't exist, print a warning
            if not os.path.exists(model_path):
                print(f"Warning: Local model file not found at {model_path}. Will proceed with mock inference.")
                self._mock_model = True
                return
            
            # Load model
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_length,
                verbose=False
            )
            
            # Store configuration
            self.model_config = {
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': 100
            }
            
            self._mock_model = False
        except ImportError:
            print("Warning: llama_cpp not installed. Using mock LLM responses.")
            self._mock_model = True
        except Exception as e:
            print(f"Error loading local model: {e}. Using mock LLM responses.")
            self._mock_model = True
    
    def _init_openai_model(self, config):
        """Initialize OpenAI API client."""
        try:
            import openai
            
            # Check for API key
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OPENAI_API_KEY environment variable not set. Using mock LLM responses.")
                self._mock_model = True
                return
            
            openai.api_key = api_key
            
            # Store model configuration
            self.model_config = {
                'model': config.get('model', 'gpt-3.5-turbo'),
                'temperature': config.get('temperature', 0.3),
                'max_tokens': config.get('max_tokens', 100)
            }
            
            self._mock_model = False
        except ImportError:
            print("Warning: openai package not installed. Using mock LLM responses.")
            self._mock_model = True
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}. Using mock LLM responses.")
            self._mock_model = True
    
    def _init_anthropic_model(self, config):
        """Initialize Anthropic API client."""
        try:
            import anthropic
            
            # Check for API key
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                print("Warning: ANTHROPIC_API_KEY environment variable not set. Using mock LLM responses.")
                self._mock_model = True
                return
            
            self.model = anthropic.Anthropic(api_key=api_key)
            
            # Store model configuration
            self.model_config = {
                'model': config.get('model', 'claude-3-haiku'),
                'temperature': config.get('temperature', 0.3),
                'max_tokens': config.get('max_tokens', 100)
            }
            
            self._mock_model = False
        except ImportError:
            print("Warning: anthropic package not installed. Using mock LLM responses.")
            self._mock_model = True
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}. Using mock LLM responses.")
            self._mock_model = True
    
    def _default_prompt_template(self):
        """Return default prompt template if none is provided in config."""
        return """You are an autonomous driving system's explainability module. Explain the vehicle's current actions in clear, concise language.

Vehicle State:
- Speed: {speed} km/h
- Current controls: {controls}

Environment:
{environment}

Planning Decision:
{planning_decision}

Based on this information, explain why the vehicle is taking its current action in 1-2 short sentences:"""
    
    def explain(self, data):
        """
        Generate a natural language explanation for the current driving decision.
        
        Args:
            data: Dictionary containing information about the vehicle state, perception, prediction, and planning.
            
        Returns:
            A natural language explanation.
        """
        # Rate limiting - don't generate explanations too frequently
        current_time = time.time()
        if current_time - self.last_explanation_time < self.explanation_cooldown:
            return "Driving based on previous assessment."
        
        self.last_explanation_time = current_time
        
        # Format input data for the model
        formatted_input = self._format_data(data)
        
        # Generate explanation
        if self._mock_model:
            explanation = self._mock_explanation(formatted_input)
        elif self.model_type == 'local':
            explanation = self._explain_local(formatted_input)
        elif self.model_type == 'openai':
            explanation = self._explain_openai(formatted_input)
        elif self.model_type == 'anthropic':
            explanation = self._explain_anthropic(formatted_input)
        else:
            explanation = "Driving normally."
        
        return explanation
    
    def _format_data(self, data):
        """Format the input data for the LLM."""
        # Extract information from data
        vehicle_state = data.get('vehicle_state', {})
        perception = data.get('perception', {})
        prediction = data.get('prediction', {})
        planning = data.get('planning', {})
        
        # Format vehicle speed
        speed = vehicle_state.get('speed', 0)
        
        # Format controls
        control = vehicle_state.get('control', {})
        controls_text = f"throttle={control.throttle:.2f}, brake={control.brake:.2f}, steer={control.steer:.2f}"
        
        # Format environment information (detected objects, etc.)
        environment_items = []
        
        # Add detected objects
        detected_objects = perception.get('detected_objects', [])
        for obj in detected_objects[:3]:  # Limit to most relevant objects
            cls_id = obj.get('class_id', 'unknown')
            bbox = obj.get('bbox', [0, 0, 0, 0])
            
            # Simple estimation of distance and position (would use actual 3D info in a real system)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            center_x = (bbox[0] + bbox[2]) / 2
            
            # Estimate position (left, center, right)
            position = "left" if center_x < 320 else ("right" if center_x > 640 else "ahead")
            
            # Estimate distance (crude approximation)
            size = width * height
            distance = "very close" if size > 40000 else ("close" if size > 10000 else "distant")
            
            # Map class ID to name
            class_names = {
                0: 'pedestrian',
                1: 'bicycle',
                2: 'car',
                3: 'motorcycle',
                5: 'bus',
                7: 'truck',
                9: 'traffic light',
                11: 'stop sign'
            }
            class_name = class_names.get(cls_id, f"object (id={cls_id})")
            
            environment_items.append(f"{class_name} {position}, {distance}")
        
        # Add lane information
        lane_info = perception.get('lane_info', {})
        lanes = lane_info.get('lanes', [])
        if lanes:
            environment_items.append(f"{len(lanes)} lane(s) detected")
        
        # Format environment text
        environment_text = "- " + "\n- ".join(environment_items) if environment_items else "No significant objects detected."
        
        # Format planning decision
        behavior = planning.get('behavior', 'unknown')
        reason = planning.get('reason', 'driving normally')
        planning_text = f"{behavior}: {reason}"
        
        # Format using the prompt template
        return self.prompt_template.format(
            speed=speed,
            controls=controls_text,
            environment=environment_text,
            planning_decision=planning_text
        )
    
    def _explain_local(self, formatted_input):
        """Generate explanation using local LLM."""
        try:
            if self._mock_model:
                return self._mock_explanation(formatted_input)
            
            # Call the local model
            output = self.model(
                formatted_input,
                temperature=self.model_config['temperature'],
                top_p=self.model_config['top_p'],
                max_tokens=self.model_config['max_tokens']
            )
            
            return output['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error generating explanation with local model: {e}")
            return self._mock_explanation(formatted_input)
    
    def _explain_openai(self, formatted_input):
        """Generate explanation using OpenAI API."""
        try:
            import openai
            
            if self._mock_model:
                return self._mock_explanation(formatted_input)
            
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model_config['model'],
                messages=[
                    {"role": "system", "content": "You are a autonomous driving system's explainability module."},
                    {"role": "user", "content": formatted_input}
                ],
                temperature=self.model_config['temperature'],
                max_tokens=self.model_config['max_tokens']
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating explanation with OpenAI API: {e}")
            return self._mock_explanation(formatted_input)
    
    def _explain_anthropic(self, formatted_input):
        """Generate explanation using Anthropic API."""
        try:
            if self._mock_model:
                return self._mock_explanation(formatted_input)
            
            # Call the Anthropic API
            response = self.model.messages.create(
                model=self.model_config['model'],
                messages=[
                    {"role": "user", "content": formatted_input}
                ],
                temperature=self.model_config['temperature'],
                max_tokens=self.model_config['max_tokens']
            )
            
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error generating explanation with Anthropic API: {e}")
            return self._mock_explanation(formatted_input)
    
    def _mock_explanation(self, formatted_input):
        """Generate a mock explanation when the LLM is not available."""
        # Simple rule-based explanation based on the input string
        if "emergency" in formatted_input.lower():
            return "Performing emergency stop due to an obstacle in our path."
        elif "traffic light" in formatted_input.lower() and "red" in formatted_input.lower():
            return "Stopping for a red traffic light ahead."
        elif "follow_vehicle" in formatted_input.lower():
            return "Following the vehicle ahead at a safe distance."
        elif "lane_change_left" in formatted_input.lower():
            return "Changing to the left lane to pass slower traffic."
        elif "lane_change_right" in formatted_input.lower():
            return "Moving to the right lane to maintain proper lane discipline."
        elif "pedestrian" in formatted_input.lower():
            return "Slowing down for a pedestrian crossing ahead."
        elif "speed" in formatted_input.lower() and any(str(speed) in formatted_input for speed in range(50, 151)):
            return "Maintaining cruising speed on an open road."
        else:
            return "Driving normally, following the current lane." 