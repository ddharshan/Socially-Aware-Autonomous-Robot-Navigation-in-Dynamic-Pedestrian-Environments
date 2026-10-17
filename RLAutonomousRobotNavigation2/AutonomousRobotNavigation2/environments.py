import numpy as np
from typing import Tuple
from typing import Optional
from collections import namedtuple
from os import path

import gym
from gym import spaces
from gym.utils import seeding
gym.logger.set_level(40)  # noqa

from AutonomousRobotNavigation2.agents import BaseAgent  #change---------------------------------- 
from AutonomousRobotNavigation2.agents import MovingAgent  #change---------------------------------------



# Action Id
ACCELERATE = 0    
TURN = 1
BREAK = 2
   

Target = namedtuple('Target', ['x', 'y', 'radius'])


class Action:
    """"
    Action class to store and standardize the action for the environment.
    """
    def __init__(self, id_: int, parameters: list):
        """"
        Initialization of an action.

        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        """"
        Property method to return the parameter related to the action selected.

        Returns:
            The parameter related to this action_id
        """
        if len(self.parameters) == 2:
            return self.parameters[self.id]
        else:
            return self.parameters[0]


class BaseEnv(gym.Env):
    """"
    Gym environment parent class.
    """
    def __init__(
            self,
            seed: Optional[int] = None,
            max_turn: float = np.pi/9,
            max_acceleration: float = 1,
            delta_t: float = 0.005,
            max_step: int = 1500,
            penalty: float = 0.001,
            break_value: float = 0.1,
    ):
        """Initialization of the gym environment.

        Args:
            seed (int): Seed used to get reproducible results.
            max_turn (float): Maximum turn during one step (in radian).
            max_acceleration (float): Maximum acceleration during one step.
            delta_t (float): Time duration of one step.
            max_step (int): Maximum number of steps in one episode.
            penalty (float): Score penalty given at the agent every step.
            break_value (float): Break value when performing break action.
        """
        # Agent Parameters
        self.max_turn = max_turn
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 2.0
        self.target_radius = 0.10
        self.penalty = penalty

        # Initialization
        self.seed(seed)
        self.target = None
        self.viewer = None
        self.current_step = None
        self.agent = BaseAgent(break_value=break_value, delta_t=delta_t)
        self.pedestrian1 = BaseAgent(break_value=break_value, delta_t=delta_t)
        self.pedestrian2 = BaseAgent(break_value=break_value, delta_t=delta_t)
        self.pedestrian3 = BaseAgent(break_value=break_value, delta_t=delta_t)
        self.pedestrian4 = BaseAgent(break_value=break_value, delta_t=delta_t)
        self.pedestrian5 = BaseAgent(break_value=break_value, delta_t=delta_t)
        self.pedestrian6 = BaseAgent(break_value=break_value, delta_t=delta_t)



        parameters_min = np.array([0, -1])
        parameters_max = np.array([1, +1])

        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(parameters_min, parameters_max)))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(60,),
            dtype=np.float32
        ) 

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self) -> list:
        self.current_step = 0

        # Episode metric tracking only.
        # These values are returned in info and do not change reward, state, action, or dynamics.
        self.episode_return = 0.0
        self.episode_min_pedestrian_distance = float("inf")
        self.episode_speed_sum = 0.0
        self.episode_speed_change_sum = 0.0
        self.episode_action_switch_count = 0
        self.episode_turn_action_count = 0
        self.episode_accelerate_action_count = 0
        self.episode_break_action_count = 0
        self.episode_social_norm_count = 0
        self.episode_near_pedestrian_count = 0
        self.episode_turn_amount_sum = 0.0
        self.previous_action_id = None
        self.previous_speed = 0.0

        #self.target = Target(0.9,-0.85,self.target_radius) #position of the goal
        
        #self.agent.reset(0,0,0,0,0.8, (np.pi)/2,0, -0.21,1.5*(np.pi))#tiallizing the robot
        
        #Randomizing the initial position of pedestrian1 which will ensure the dynmaic environment(Rather than fixing the starting position of pedestrian, randomizing will be more effective. Because the dynamic is high)   
        low = [-2, 0, 0, 0, -2, np.pi/2, 0, -2, np.pi/2, 0, -2, np.pi/2, 0, -2, np.pi/2, 0, -2, np.pi/2, 0, -2, np.pi/2]   #limiting the starting of randomization of pedestrian 1 greater than -0.3 in x plane which will ensure not collide with the robot at starting 
        high = [0, 2, 0, 2, 0, np.pi, 2, 0, np.pi, 2, 0, np.pi, 2, 0, np.pi, 2, 0, np.pi, 2, 0, np.pi]  
        
        #self.pedestrian1.reset(*self.np_random.uniform(low, high)) #randomizing the starting of pedestrian 1
        self.agent.reset(*self.np_random.uniform(low, high))  #Initiallizing the robot
        self.pedestrian1.reset(*self.np_random.uniform(low, high)) #randomizing the starting of pedestrian 1
        self.pedestrian2.reset(*self.np_random.uniform(low, high))    #randomizing the starting of pedestrian 2
        self.pedestrian3.reset(*self.np_random.uniform(low, high)) 
        self.pedestrian4.reset(*self.np_random.uniform(low, high)) 
        self.pedestrian5.reset(*self.np_random.uniform(low, high)) 
        self.pedestrian6.reset(*self.np_random.uniform(low, high)) 
        
 
         
        lower = [0, -2, 0]
        higher = [2, 0, 0]
        self.target = Target(*self.np_random.uniform(lower, higher))
    
        #Initial acceleration which means, the constant velocity due to the absense of the increament in def step()
        self.pedestrian1.p1accelerate(1.2) #Constant speed or initial speed
        self.pedestrian2.p2accelerate(1.2) #Constant speed or initial speed
        self.pedestrian3.p3accelerate(1.2) #Constant speed or initial speed
        self.pedestrian4.p4accelerate(1.2) #Constant speed or initial speed
        self.pedestrian5.p5accelerate(1.2) #Constant speed or initial speed
        self.pedestrian6.p6accelerate(1.2) #Constant speed or initial speed
      
       
        
      
        

        # Store initial speed for smoothness metrics.
        self.previous_speed = abs(self.agent.speed)

        return self.get_state()

    def step(self, raw_action: Tuple[int, list]) -> Tuple[list, float, bool, dict]:
        action = Action(*raw_action)
        last_distance = self.distance
        self.current_step += 1

        # Step metric tracking only.
        # These variables do not change the action, reward logic, or dynamics.
        speed_before_action = abs(self.agent.speed)
        rotation_abs = 0.0
        norm_applied = False

        if action.id == TURN:
            rotation = self.max_turn * max(min(action.parameter, 1), -1)
            rotation_abs = abs(rotation)
            self.agent.turn(rotation)
            if self.pedestrian1.p1y <= -self.field_size or self.pedestrian1.p1y >= self.field_size or self.pedestrian1.p1x <= -self.field_size or self.pedestrian1.p1x >= self.field_size : #q
                    self.pedestrian1.p1turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian1.p1accelerate(0) #Constant speed 
               
            elif self.pedestrian2.p2y <= -self.field_size or self.pedestrian2.p2y >= self.field_size or self.pedestrian2.p2x <= -self.field_size or self.pedestrian2.p2x >= self.field_size:
                    self.pedestrian2.p2turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian2.p2accelerate(0) #Constant speed
                    
            elif self.pedestrian3.p3y <= -self.field_size or self.pedestrian3.p3y >= self.field_size or self.pedestrian3.p3x <= -self.field_size or self.pedestrian3.p3x >= self.field_size:
                    self.pedestrian3.p3turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian3.p3accelerate(0) #Constant speed
                    
            elif self.pedestrian4.p4y <= -self.field_size or self.pedestrian4.p4y >= self.field_size or self.pedestrian4.p4x <= -self.field_size or self.pedestrian4.p4x >= self.field_size:
                    self.pedestrian4.p4turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian4.p4accelerate(0) #Constant speed
	    
            elif self.pedestrian5.p5y <= -self.field_size or self.pedestrian5.p5y >= self.field_size or self.pedestrian5.p5x <= -self.field_size or self.pedestrian5.p5x >= self.field_size:
                    self.pedestrian5.p5turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian5.p5accelerate(0) #Constant speed
            
            elif self.pedestrian6.p6y <= -self.field_size or self.pedestrian6.p6y >= self.field_size or self.pedestrian6.p6x <= -self.field_size or self.pedestrian6.p6x >= self.field_size:
                    self.pedestrian6.p6turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian6.p6accelerate(0) #Constant speed
                   
            else:
                    self.pedestrian1.p1accelerate(0) #Constant speed 
                    self.pedestrian2.p2accelerate(0) #Constant speed
                    self.pedestrian3.p3accelerate(0) #Constant speed
                    self.pedestrian4.p4accelerate(0) #Constant speed
                    self.pedestrian5.p5accelerate(0) #Constant speed
                    self.pedestrian6.p6accelerate(0) #Constant speed
                   
                   
                  
        elif action.id == ACCELERATE:
            if self.agent.speed <= 1:  #acceleration is applied up to 2m/s (agent's maximum speed is 2m/s. Due to increment of max=0.1, the max.speed is 1.999~2m/s)
                    acceleration = self.max_acceleration * max(min(action.parameter, 1), 0) #Randomizing the acceleration - The maximum acceleration is by 0.1. However, 0.1*1=0.1 also can be achieved
                    self.agent.accelerate(acceleration)
                    if self.pedestrian1.p1y <= -self.field_size or self.pedestrian1.p1y >= self.field_size or self.pedestrian1.p1x <= -self.field_size or self.pedestrian1.p1x >= self.field_size : #q
                            self.pedestrian1.p1turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian1.p1accelerate(0) #Constant speed 
               
                    elif self.pedestrian2.p2y <= -self.field_size or self.pedestrian2.p2y >= self.field_size or self.pedestrian2.p2x <= -self.field_size or self.pedestrian2.p2x >= self.field_size:
                            self.pedestrian2.p2turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian2.p2accelerate(0) #Constant speed
                            
                            
                    elif self.pedestrian3.p3y <= -self.field_size or self.pedestrian3.p3y >= self.field_size or self.pedestrian3.p3x <= -self.field_size or self.pedestrian3.p3x >= self.field_size:
                            self.pedestrian3.p3turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian3.p3accelerate(0) #Constant speed
                            
                            
                    
                    elif self.pedestrian4.p4y <= -self.field_size or self.pedestrian4.p4y >= self.field_size or self.pedestrian4.p4x <= -self.field_size or self.pedestrian4.p4x >= self.field_size:
                            self.pedestrian4.p4turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian4.p4accelerate(0) #Constant speed
				
                    elif self.pedestrian5.p5y <= -self.field_size or self.pedestrian5.p5y >= self.field_size or self.pedestrian5.p5x <= -self.field_size or self.pedestrian5.p5x >= self.field_size:
                            self.pedestrian5.p5turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian5.p5accelerate(0) #Constant speed
			
                    elif self.pedestrian6.p6y <= -self.field_size or self.pedestrian6.p6y >= self.field_size or self.pedestrian6.p6x <= -self.field_size or self.pedestrian6.p6x >= self.field_size:
                            self.pedestrian6.p6turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian6.p6accelerate(0) #Constant speed
               
                    else:
                            self.pedestrian1.p1accelerate(0) #Constant speed 
                            self.pedestrian2.p2accelerate(0) #Constant speed 
                            self.pedestrian3.p3accelerate(0) #Constant speed 
                            self.pedestrian4.p4accelerate(0) #Constant speed
                            self.pedestrian5.p5accelerate(0) #Constant speed
                            self.pedestrian6.p6accelerate(0) #Constant speed
                   
                        
            else:
                    acceleration = 0
                    self.agent.accelerate(acceleration)
                    
                    if self.pedestrian1.p1y <= -self.field_size or self.pedestrian1.p1y >= self.field_size or self.pedestrian1.p1x <= -self.field_size or self.pedestrian1.p1x >= self.field_size : #q
                            self.pedestrian1.p1turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian1.p1accelerate(0) #Constant speed 
                    
                    
                    elif self.pedestrian2.p2y <= -self.field_size or self.pedestrian2.p2y >= self.field_size or self.pedestrian2.p2x <= -self.field_size or self.pedestrian2.p2x >= self.field_size:
                            self.pedestrian2.p2turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian2.p2accelerate(0) #Constant speed
                            
                            
                    elif self.pedestrian3.p3y <= -self.field_size or self.pedestrian3.p3y >= self.field_size or self.pedestrian3.p3x <= -self.field_size or self.pedestrian3.p3x >= self.field_size:
                            self.pedestrian3.p3turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian3.p3accelerate(0) #Constant speed
                            
                    
                    elif self.pedestrian4.p4y <= -self.field_size or self.pedestrian4.p4y >= self.field_size or self.pedestrian4.p4x <= -self.field_size or self.pedestrian4.p4x >= self.field_size:
                            self.pedestrian4.p4turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian4.p4accelerate(0) #Constant speed
				
				
                    elif self.pedestrian5.p5y <= -self.field_size or self.pedestrian5.p5y >= self.field_size or self.pedestrian5.p5x <= -self.field_size or self.pedestrian5.p5x >= self.field_size:
                            self.pedestrian5.p5turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian5.p5accelerate(0) #Constant speed
		 	
                    elif self.pedestrian6.p6y <= -self.field_size or self.pedestrian6.p6y >= self.field_size or self.pedestrian6.p6x <= -self.field_size or self.pedestrian6.p6x >= self.field_size:
                            self.pedestrian6.p6turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                            self.pedestrian6.p6accelerate(0) #Constant speed
               
               
                    else:
                            self.pedestrian1.p1accelerate(0) #Constant speed 
                            self.pedestrian2.p2accelerate(0) #Constant speed
                            self.pedestrian3.p3accelerate(0) #Constant speed
                            self.pedestrian4.p4accelerate(0) #Constant speed
                            self.pedestrian5.p5accelerate(0) #Constant speed
                            self.pedestrian6.p6accelerate(0) #Constant speed
                           
                           
                    
                    
        elif action.id == BREAK:
            self.agent.break_()
            if self.pedestrian1.p1y <= -self.field_size or self.pedestrian1.p1y >= self.field_size or self.pedestrian1.p1x <= -self.field_size or self.pedestrian1.p1x >= self.field_size : #q
                    self.pedestrian1.p1turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian1.p1accelerate(0) #Constant speed 
                    
                    
            elif self.pedestrian2.p2y <= -self.field_size or self.pedestrian2.p2y >= self.field_size or self.pedestrian2.p2x <= -self.field_size or self.pedestrian2.p2x >= self.field_size:
                    self.pedestrian2.p2turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian2.p2accelerate(0) #Constant speed
                    
            elif self.pedestrian3.p3y <= -self.field_size or self.pedestrian3.p3y >= self.field_size or self.pedestrian3.p3x <= -self.field_size or self.pedestrian3.p3x >= self.field_size:
                    self.pedestrian3.p3turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian3.p3accelerate(0) #Constant speed
                    
            elif self.pedestrian4.p4y <= -self.field_size or self.pedestrian4.p4y >= self.field_size or self.pedestrian4.p4x <= -self.field_size or self.pedestrian4.p4x >= self.field_size:
                    self.pedestrian4.p4turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian4.p4accelerate(0) #Constant speed
	
            elif self.pedestrian5.p5y <= -self.field_size or self.pedestrian5.p5y >= self.field_size or self.pedestrian5.p5x <= -self.field_size or self.pedestrian5.p5x >= self.field_size:
                    self.pedestrian5.p5turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian5.p5accelerate(0) #Constant speed
			
			
            elif self.pedestrian6.p6y <= -self.field_size or self.pedestrian6.p6y >= self.field_size or self.pedestrian6.p6x <= -self.field_size or self.pedestrian6.p6x >= self.field_size:
                    self.pedestrian6.p6turn(np.pi) #Turn by 180 degree which will ensure the pedestrian is not out of the environment 
                    self.pedestrian6.p6accelerate(0) #Constant speed
               
            else:
                    self.pedestrian1.p1accelerate(0) #Constant speed 
                    self.pedestrian2.p2accelerate(0) #Constant speed
                    self.pedestrian3.p3accelerate(0) #Constant speed
                    self.pedestrian4.p4accelerate(0) #Constant speed
                    self.pedestrian5.p5accelerate(0) #Constant speed
                    self.pedestrian6.p6accelerate(0) #Constant speed
                   
                   
                  
        
#------------------Expert Knowledge (Reward-function) -------------------------------------------

        done_reason = "none"

        if self.distance <= self.target_radius: # Robot reached the goal
            reward = self.get_reward(last_distance, True, False)
            done = True
            done_reason = "goal"

        elif abs(self.agent.x) > self.field_size or abs(self.agent.y) > self.field_size: # Robot moving out of env
            # MODIFIED REWARD: boundary penalty increased from -1 to -5
            reward = -5.0
            done = True
            done_reason = "boundary"

        elif self.current_step > self.max_step: # Maximum steps reached
            # MODIFIED REWARD: timeout penalty increased from -1 to -5
            reward = -5.0
            done = True
            done_reason = "timeout"

        elif self.collision1 <= 0.21 or self.collision2 <= 0.21 or self.collision3 <= 0.21 or self.collision4 <= 0.21 or self.collision5 <= 0.21 or self.collision6 <= 0.21:  # Collision
            # MODIFIED REWARD: collision terminal penalty remains -8.0
            reward = -8.0
            done = True
            done_reason = "collision"

        elif (self.distance > 0.4) and (0.30 <= self.collision1 < 0.35): #social-norm inducing reward for P1        
            
            if ((0.75*(np.pi)) < abs(self.thetaP1n - self.thetaRn) < np.pi):  #Passing of P1
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif (((np.pi/4) < (self.thetaP1n - self.thetaRn) < 0.75*(np.pi)) and (abs(self.pedestrian1.p1speed) - abs(self.agent.speed) > 0)):  #Crossing of P1
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif ((0 < (self.thetaRn - self.thetaP1n) < (np.pi/4)) and (abs(self.agent.speed) > abs(self.pedestrian1.p1speed))): #overtaking of P1
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            else:
                reward = self.get_reward(last_distance) 
                done = False
                
                
        elif (self.distance > 0.4) and (0.30 <= self.collision2 < 0.35): #social-norm inducing reward for P1        
            
            if ((0.75*(np.pi)) < abs(self.thetaP2n - self.thetaRn) < np.pi):  #Passing of P1
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif (((np.pi/4) < (self.thetaP2n - self.thetaRn) < 0.75*(np.pi)) and (abs(self.pedestrian2.p2speed) - abs(self.agent.speed) > 0)):  #Crossing of P1
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif ((0 < (self.thetaRn - self.thetaP2n) < (np.pi/4)) and (abs(self.agent.speed) > abs(self.pedestrian2.p2speed))): #overtaking of P1
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            else:
                reward = self.get_reward(last_distance) 
                done = False
                
                
        elif (self.distance > 0.4) and (0.30 <= self.collision3 < 0.35): #social-norm inducing reward for P3        
            
            if ((0.75*(np.pi)) < abs(self.thetaP3n - self.thetaRn) < np.pi):  #Passing of P3
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif (((np.pi/4) < (self.thetaP3n - self.thetaRn) < 0.75*(np.pi)) and (abs(self.pedestrian3.p3speed) - abs(self.agent.speed) > 0)):  #Crossing of P3
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif ((0 < (self.thetaRn - self.thetaP3n) < (np.pi/4)) and (abs(self.agent.speed) > abs(self.pedestrian3.p3speed))): #overtaking of P1
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            else:
                reward = self.get_reward(last_distance) 
                done = False
        
        elif (self.distance > 0.4) and (0.30 <= self.collision4 < 0.35): #social-norm inducing reward for P4        
            
            if ((0.75*(np.pi)) < abs(self.thetaP4n - self.thetaRn) < np.pi):  #Passing of P4
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif (((np.pi/4) < (self.thetaP4n - self.thetaRn) < 0.75*(np.pi)) and (abs(self.pedestrian4.p4speed) - abs(self.agent.speed) > 0)):  #Crossing of P4
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif ((0 < (self.thetaRn - self.thetaP4n) < (np.pi/4)) and (abs(self.agent.speed) > abs(self.pedestrian4.p4speed))): #overtaking of P4
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            else:
                reward = self.get_reward(last_distance) 
                done = False
		
	
        elif (self.distance > 0.4) and (0.30 <= self.collision5 < 0.35): #social-norm inducing reward for P5        
            
            if ((0.75*(np.pi)) < abs(self.thetaP5n - self.thetaRn) < np.pi):  #Passing of P5
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif (((np.pi/4) < (self.thetaP5n - self.thetaRn) < 0.75*(np.pi)) and (abs(self.pedestrian5.p5speed) - abs(self.agent.speed) > 0)):  #Crossing of P5
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif ((0 < (self.thetaRn - self.thetaP5n) < (np.pi/4)) and (abs(self.agent.speed) > abs(self.pedestrian5.p5speed))): #overtaking of P5
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            else:
                reward = self.get_reward(last_distance) 
                done = False
		
		
        elif (self.distance > 0.4) and (0.30 <= self.collision6 < 0.35): #social-norm inducing reward for P6       
            
            if ((0.75*(np.pi)) < abs(self.thetaP6n - self.thetaRn) < np.pi):  #Passing of P6
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif (((np.pi/4) < (self.thetaP6n - self.thetaRn) < 0.75*(np.pi)) and (abs(self.pedestrian6.p6speed) - abs(self.agent.speed) > 0)):  #Crossing of P6
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            elif ((0 < (self.thetaRn - self.thetaP6n) < (np.pi/4)) and (abs(self.agent.speed) > abs(self.pedestrian6.p6speed))): #overtaking of P6
                norm_applied = True
                reward = self.get_reward(last_distance, False, True)
                done = False
            
            else:
                reward = self.get_reward(last_distance) 
                done = False
        
        
        
        else:
            reward = self.get_reward(last_distance) 
            done = False
            
#------------------------------------------------------------------------------

        # Update metrics after reward is known.
        self._update_episode_metrics(
            action=action,
            norm_applied=norm_applied,
            speed_before_action=speed_before_action,
            rotation_abs=rotation_abs,
            reward=reward,
        )

        info = self._get_episode_info(done_reason)

        return self.get_state(), reward, done, info

    def _nearest_pedestrian_distance(self) -> float:
        return min(
            self.collision1,
            self.collision2,
            self.collision3,
            self.collision4,
            self.collision5,
            self.collision6,
        )

    def _update_episode_metrics(
            self,
            action: Action,
            norm_applied: bool,
            speed_before_action: float,
            rotation_abs: float,
            reward: float,
    ) -> None:
        """
        Update episode statistics for monitoring and publication metrics.
        This function does not change reward, observation, action, or dynamics.
        """
        nearest_pedestrian_distance = self._nearest_pedestrian_distance()

        self.episode_min_pedestrian_distance = min(
            self.episode_min_pedestrian_distance,
            nearest_pedestrian_distance,
        )

        if nearest_pedestrian_distance < 0.30:
            self.episode_near_pedestrian_count += 1

        speed_now = abs(self.agent.speed)
        self.episode_speed_sum += speed_now
        self.episode_speed_change_sum += abs(speed_now - speed_before_action)

        if action.id == ACCELERATE:
            self.episode_accelerate_action_count += 1
        elif action.id == TURN:
            self.episode_turn_action_count += 1
        elif action.id == BREAK:
            self.episode_break_action_count += 1

        if self.previous_action_id is not None and action.id != self.previous_action_id:
            self.episode_action_switch_count += 1

        self.previous_action_id = action.id
        self.previous_speed = speed_now

        self.episode_turn_amount_sum += rotation_abs

        if norm_applied:
            self.episode_social_norm_count += 1

        self.episode_return += float(reward)

    def _get_episode_info(self, done_reason: str) -> dict:
        """
        Return monitoring information for each step.
        At the terminal step, these values summarize the whole episode.
        """
        steps = max(int(self.current_step), 1)
        min_pedestrian_distance = self.episode_min_pedestrian_distance

        if not np.isfinite(min_pedestrian_distance):
            min_pedestrian_distance = self._nearest_pedestrian_distance()

        info = {
            # Termination reason
            "done_reason": done_reason,
            "is_success": done_reason == "goal",
            "is_collision": done_reason == "collision",
            "is_boundary": done_reason == "boundary",
            "is_timeout": done_reason == "timeout",

            # Episode length / efficiency
            "episode_steps": int(self.current_step),
            "episode_time": float(self.current_step * self.delta_t),
            "episode_return": float(self.episode_return),

            # Safety
            "nearest_pedestrian_distance": float(self._nearest_pedestrian_distance()),
            "min_pedestrian_distance": float(min_pedestrian_distance),
            "near_pedestrian_count": int(self.episode_near_pedestrian_count),
            "near_pedestrian_rate": float(self.episode_near_pedestrian_count / steps),

            # Speed / motion behavior
            "avg_speed": float(self.episode_speed_sum / steps),
            "avg_abs_speed_change": float(self.episode_speed_change_sum / steps),
            "turn_amount_sum": float(self.episode_turn_amount_sum),
            "avg_turn_amount": float(self.episode_turn_amount_sum / steps),

            # Discrete action behavior / smoothness
            "accelerate_action_count": int(self.episode_accelerate_action_count),
            "turn_action_count": int(self.episode_turn_action_count),
            "break_action_count": int(self.episode_break_action_count),
            "action_switch_count": int(self.episode_action_switch_count),
            "action_switch_rate": float(self.episode_action_switch_count / steps),

            # Social norm monitoring
            "social_norm_count": int(self.episode_social_norm_count),
            "social_norm_rate": float(self.episode_social_norm_count / steps),
        }

        return info

    def get_state(self) -> list:
        state = [
            
            #for robot and goal
            self.agent.x,
            self.agent.y,
            self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta),
            self.target.x,
            self.target.y,
            self.distance,
            0 if self.distance > self.target_radius else 1,
            self.current_step / self.max_step,
            
            #for collision1
            self.pedestrian1.p1x,
            self.pedestrian1.p1y,
            self.collision1,
            np.cos(self.pedestrian1.p1theta),
            np.sin(self.pedestrian1.p1theta),
            self.pedestrian1.p1speed,
            
            #for collision 2
            self.collision2,
            np.cos(self.pedestrian2.p2theta),
            np.sin(self.pedestrian2.p2theta),
            self.pedestrian2.p2x,
            self.pedestrian2.p2y,
            self.pedestrian2.p2speed,
            
            #for collision 3
            self.collision3,
            np.cos(self.pedestrian3.p3theta),
            np.sin(self.pedestrian3.p3theta),
            self.pedestrian3.p3x,
            self.pedestrian3.p3y,
            self.pedestrian3.p3speed,
            
            #for collision 4
            self.collision4,
            np.cos(self.pedestrian4.p4theta),
            np.sin(self.pedestrian4.p4theta),
            self.pedestrian4.p4x,
            self.pedestrian4.p4y,
            self.pedestrian4.p4speed,
		
		
	    #for collision 5
            self.collision5,
            np.cos(self.pedestrian5.p5theta),
            np.sin(self.pedestrian5.p5theta),
            self.pedestrian5.p5x,
            self.pedestrian5.p5y,
            self.pedestrian5.p5speed,
		
	    #for collision 6
            self.collision6,
            np.cos(self.pedestrian6.p6theta),
            np.sin(self.pedestrian6.p6theta),
            self.pedestrian6.p6x,
            self.pedestrian6.p6y,
            self.pedestrian6.p6speed,
            
            #pedestrian norm
            self.thetaP1n,
            self.thetaP2n, 
            self.thetaP3n,
            self.thetaP4n,
	    self.thetaP5n,
	    self.thetaP6n,
            self.thetaRn, 
            self.agent.theta, 
            self.pedestrian1.p1theta,  
            self.pedestrian2.p2theta,  
            self.pedestrian3.p3theta,  
            self.pedestrian4.p4theta,  
            self.pedestrian5.p5theta,
            self.pedestrian6.p6theta  
         
        ]
        return state

    def get_reward(self, last_distance: float, goal: bool = False, norm: bool = False) -> float:
        """
        MODIFIED REWARD FUNCTION

        Implements reward shaping:
        1. Scale progress reward: 3.0 * (last_distance - current_distance)
        2. Goal reward: +5.0
        3. Goal-directed social norm reward:
           +0.005 only when the robot respects pedestrian norms and moves toward the goal
        4. Near-pedestrian danger penalty when nearest pedestrian distance < 0.30
        5. Boundary safety penalty when robot moves close to the environment boundary
        6. Near-goal speed penalty when distance to goal < 0.30
        7. Modified stuck penalty so it is not applied inside the near-goal slow-down zone
        8. Keep time penalty as -0.001 per step

        Terminal penalties are handled separately in step():
        collision  -> -8.0 and done=True
        boundary   -> -5.0 and done=True
        timeout    -> -5.0 and done=True
        """

        # ---------------- MODIFIED REWARD CONSTANTS ----------------
        PROGRESS_SCALE = 3.0
        TIME_PENALTY = 0.001

        GOAL_REWARD = 5.0
        NORM_REWARD = 0.005

        # Goal-directed social norm reward gate.
        # The robot receives social norm reward only when it respects
        # passing/crossing/overtaking while still moving toward the goal.
        SOCIAL_NORM_PROGRESS_THRESHOLD = 0.0002

        # Pedestrian danger penalty
        DANGER_DISTANCE = 0.30
        NEAR_PED_WEIGHT = 0.65

        # Boundary safety penalty
        BOUNDARY_MARGIN = 0.20
        BOUNDARY_WEIGHT = 0.05

        # Near-goal slow-down shaping
        D_SLOW = 0.30
        NEAR_GOAL_SPEED_WEIGHT = 0.02

        # Modified stuck penalty
        STUCK_PROGRESS_THRESHOLD = 0.0005
        STUCK_PENALTY = 0.002
        # -----------------------------------------------------------

        # 1. Progress reward: positive if robot moves closer to goal.
        progress = last_distance - self.distance
        reward = PROGRESS_SCALE * progress

        # 2. Time penalty: discourages wasting steps.
        reward -= TIME_PENALTY

        # 3. Goal reward: stronger reward when goal is reached.
        if goal:
            reward += GOAL_REWARD

        # 4. Goal-directed social norm reward:
        # Reward passing/crossing/overtaking only if the robot is also
        # making progress toward the goal.
        #
        # Old term: w_norm * I_norm
        # New term: w_norm * I_norm * I_progress
        # I_progress = 1 if (d_t - d_{t+1}) > epsilon_p, otherwise 0.
        if norm and progress > SOCIAL_NORM_PROGRESS_THRESHOLD:
            reward += NORM_REWARD

        # 5. Near-pedestrian danger penalty.
        nearest_pedestrian_distance = min(
            self.collision1,
            self.collision2,
            self.collision3,
            self.collision4,
            self.collision5,
            self.collision6
        )

        if nearest_pedestrian_distance < DANGER_DISTANCE:
            reward -= NEAR_PED_WEIGHT * (DANGER_DISTANCE - nearest_pedestrian_distance)

        # 6. Mild boundary safety penalty:
        # Penalize the robot when it moves close to the environment boundary.
        boundary_distance = self.field_size - max(abs(self.agent.x), abs(self.agent.y))

        if boundary_distance < BOUNDARY_MARGIN:
            reward -= BOUNDARY_WEIGHT * (BOUNDARY_MARGIN - boundary_distance)

        # 7. Near-goal speed penalty:
        # If the robot is near the goal, penalize high speed so it learns to slow down.
        if self.distance < D_SLOW:
            reward -= NEAR_GOAL_SPEED_WEIGHT * abs(self.agent.speed)

        # 8. Modified stuck penalty:
        # Penalize almost no progress only when the robot is far from the goal.
        # This avoids punishing the robot for slowing down near the goal.
        if abs(progress) < STUCK_PROGRESS_THRESHOLD and self.distance > D_SLOW and not goal:
            reward -= STUCK_PENALTY

        return reward

#----------------------Define the distance-----------------

    @property
    def distance(self) -> float:
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

#----------------------Define the collision1---------------------
    @property
    def collision1(self) -> float:  # Define variables for difinition of collision
        return self.get_collision1(self.agent.x, self.agent.y, self.pedestrian1.p1x, self.pedestrian1.p1y)
    
    
    @staticmethod
    def get_collision1(x3: float, y3: float, x4: float, y4: float) -> float:   #Define the collision
        return np.sqrt(((x3 - x4) ** 2) + ((y3 - y4) ** 2))

    
#-------------------Define the thetaRn for norm to wrapp [-pi, pi]----------------------
    
    @property
    def thetaRn(self) -> float:  # Define variables for difinition of thetaRn
        return self.get_thetaRn(self.agent.theta)
    
    @staticmethod
    def get_thetaRn(x5: float) -> float:   #Define the thetaRn
        if x5 > np.pi:
            x5 = -(2*(np.pi)) + x5
        else:
            x5
        return x5

#-------------------Define the thetaP1n for norm to wrapp [-pi, pi]----------------------
    
    @property
    def thetaP1n(self) -> float:  # Define variables for difinition of thetaP1n
        return self.get_thetaP1n(self.pedestrian1.p1theta)
    
    @staticmethod
    def get_thetaP1n(x6: float) -> float:   #Define the thetaP1n
        if x6 > np.pi:
            x6 = -(2*(np.pi)) + x6
        else:
            x6
        return x6

#-------------------Define the thetaP2n for norm to wrapp [-pi, pi]----------------------
    
    @property
    def thetaP2n(self) -> float:  # Define variables for difinition of thetaP2n
        return self.get_thetaP2n(self.pedestrian2.p2theta)
    
    @staticmethod
    def get_thetaP2n(x7: float) -> float:   #Define the thetaP1n
        if x7 > np.pi:
            x7 = -(2*(np.pi)) + x7
        else:
            x7
        return x7

#----------------------Define the collision2---------------------
    @property
    def collision2(self) -> float:  # Define variables for difinition of collision2
        return self.get_collision2(self.agent.x, self.agent.y, self.pedestrian2.p2x, self.pedestrian2.p2y)
    
    
    @staticmethod
    def get_collision2(x8: float, y8: float, x9: float, y9: float) -> float:   #Define the collision
        return np.sqrt(((x8 - x9) ** 2) + ((y8 - y9) ** 2))

#------------------------------------------------------------------


#----------------------Define the collision3---------------------
    @property
    def collision3(self) -> float:  # Define variables for difinition of collision2
        return self.get_collision3(self.agent.x, self.agent.y, self.pedestrian3.p3x, self.pedestrian3.p3y)
    
    
    @staticmethod
    def get_collision3(x10: float, y10: float, x11: float, y11: float) -> float:   #Define the collision
        return np.sqrt(((x10 - x11) ** 2) + ((y10 - y11) ** 2))

#-------------------Define the thetaP3n for norm to wrapp [-pi, pi]----------------------
    
    @property
    def thetaP3n(self) -> float:  # Define variables for difinition of thetaP2n
        return self.get_thetaP3n(self.pedestrian3.p3theta)
    
    @staticmethod
    def get_thetaP3n(x12: float) -> float:   #Define the thetaP1n
        if x12 > np.pi:
            x12 = -(2*(np.pi)) + x12
        else:
            x12
        return x12

#----------------------Define the collision4---------------------
    @property
    def collision4(self) -> float:  # Define variables for difinition of collision2
        return self.get_collision4(self.agent.x, self.agent.y, self.pedestrian4.p4x, self.pedestrian4.p4y)
    
    
    @staticmethod
    def get_collision4(x13: float, y13: float, x14: float, y14: float) -> float:   #Define the collision
        return np.sqrt(((x13 - x14) ** 2) + ((y13 - y14) ** 2))

#-------------------Define the thetaP4n for norm to wrapp [-pi, pi]----------------------
    
    @property
    def thetaP4n(self) -> float:  # Define variables for difinition of thetaP2n
        return self.get_thetaP4n(self.pedestrian4.p4theta)
    
    @staticmethod
    def get_thetaP4n(x15: float) -> float:   #Define the thetaP1n
        if x15 > np.pi:
            x15 = -(2*(np.pi)) + x15
        else:
            x15
        return x15
#------------------------------------------------------------------



#----------------------Define the collision5---------------------
    @property
    def collision5(self) -> float:  # Define variables for difinition of collision2
        return self.get_collision5(self.agent.x, self.agent.y, self.pedestrian5.p5x, self.pedestrian5.p5y)
    
    
    @staticmethod
    def get_collision5(x16: float, y16: float, x17: float, y17: float) -> float:   #Define the collision
        return np.sqrt(((x16 - x17) ** 2) + ((y16 - y17) ** 2))

#-------------------Define the thetaP5n for norm to wrapp [-pi, pi]----------------------
    
    @property
    def thetaP5n(self) -> float:  # Define variables for difinition of thetaP5n
        return self.get_thetaP5n(self.pedestrian5.p5theta)
    
    @staticmethod
    def get_thetaP5n(x18: float) -> float:   #Define the thetaP1n
        if x18 > np.pi:
            x18 = -(2*(np.pi)) + x18
        else:
            x18
        return x18
#------------------------------------------------------------------


#----------------------Define the collision6---------------------
    @property
    def collision6(self) -> float:  # Define variables for difinition of collision2
        return self.get_collision6(self.agent.x, self.agent.y, self.pedestrian6.p6x, self.pedestrian6.p6y)
    
    
    @staticmethod
    def get_collision6(x19: float, y19: float, x20: float, y20: float) -> float:   #Define the collision
        return np.sqrt(((x19 - x20) ** 2) + ((y19 - y20) ** 2))

#-------------------Define the thetaP5n for norm to wrapp [-pi, pi]----------------------
    
    @property
    def thetaP6n(self) -> float:  # Define variables for difinition of thetaP5n
        return self.get_thetaP6n(self.pedestrian6.p6theta)
    
    @staticmethod
    def get_thetaP6n(x21: float) -> float:   #Define the thetaP1n
        if x21 > np.pi:
            x21 = -(2*(np.pi)) + x21
        else:
            x21
        return x21
#------------------------------------------------------------------



    def render(self, mode='human'):
        # Fast auto-fit render.
        # This changes only the render window size, not the real environment size.
        # Real environment remains: self.field_size = 2.0

        if self.viewer is None:
            try:
                import tkinter as tk

                root = tk.Tk()
                root.withdraw()

                monitor_width = root.winfo_screenwidth()
                monitor_height = root.winfo_screenheight()

                root.destroy()

                # Fit large on laptop/monitor but leave space for title bar/taskbar.
                window_size = min(monitor_width - 80, monitor_height - 120)

                # Safety limits.
                window_size = max(560, min(int(window_size), 1100))

            except Exception:
                window_size = 760

            self._screen_width = window_size
            self._screen_height = window_size

            # Visual margin only so objects do not get cut at the screen edge.
            # This does NOT change the real simulation boundary.
            self._render_field_size = self.field_size + 0.10

            self._unit_x = self._screen_width / (2 * self._render_field_size)
            self._unit_y = self._screen_height / (2 * self._render_field_size)

            # Keep pedestrian shoulder/body visual size correct for field_size = 2.0.
            render_scale = 1.0 / self.field_size
            shoulder_length = 8.432 * render_scale
            shoulder_diameter = 16.66 * render_scale

            agent_inner_radius = 0.03
            agent_outer_radius = 0.04
            pedestrian_radius = 0.0175

            # RENDER ONLY:
            # This is the pedestrian boundary radius shown in the video.
            # It is NOT changing the actual collision logic.
            # Actual collision condition remains robot centre to pedestrian reference point <= 0.21 m.
            # Visual explanation: robot outer radius 0.04 + pedestrian boundary 0.1705 = 0.2105 ~= 0.21 m.
            pedestrian_boundary_radius = 0.1705

            def to_screen(x, y):
                return (
                    self._unit_x * (self._render_field_size + x),
                    self._unit_y * (self._render_field_size + y)
                )

            def get_pedestrian_values(index):
                pedestrian = getattr(self, f'pedestrian{index}')
                prefix = f'p{index}'

                x = getattr(pedestrian, f'{prefix}x')
                y = getattr(pedestrian, f'{prefix}y')
                theta = getattr(pedestrian, f'{prefix}theta')

                return pedestrian, x, y, theta

            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(
                self._screen_width,
                self._screen_height
            )

            # Goal
            target = rendering.make_circle(self._unit_x * self.target_radius)
            self.target_trans = rendering.Transform(
                translation=to_screen(self.target.x, self.target.y)
            )
            target.add_attr(self.target_trans)
            target.set_color(0.80, 0, 0)
            self.viewer.add_geom(target)

            # Robot inner circle
            agentinner = rendering.make_circle(
                self._unit_x * agent_inner_radius
            )
            self.agentinner_trans = rendering.Transform(
                translation=to_screen(self.agent.x, self.agent.y)
            )
            agentinner.add_attr(self.agentinner_trans)

            # Robot outer circle
            agentouter = rendering.make_circle(
                self._unit_x * agent_outer_radius
            )
            self.agentouter_trans = rendering.Transform(
                translation=to_screen(self.agent.x, self.agent.y)
            )
            agentouter.add_attr(self.agentouter_trans)
            agentouter.set_color(1, 0.5, 0)
            self.viewer.add_geom(agentouter)

            agentinner.set_color(0, 0, 0)
            self.viewer.add_geom(agentinner)

            # Robot direction arrow
            t = 0.04 * self._unit_x
            r = 0.0212 * self._unit_y
            m = 0.0212 * self._unit_x

            arrow = rendering.FilledPolygon([(t, 0), (m, r), (m, -r)])
            self.arrow_trans = rendering.Transform(rotation=self.agent.theta)
            arrow.add_attr(self.arrow_trans)
            arrow.add_attr(self.agentinner_trans)
            arrow.set_color(0, 0, 0)
            self.viewer.add_geom(arrow)

            # Pedestrians 1 to 6
            for i in range(1, 7):
                pedestrian, px, py, ptheta = get_pedestrian_values(i)

                pedestrian_trans = rendering.Transform(
                    translation=to_screen(px, py)
                )
                setattr(self, f'pedestrian{i}_trans', pedestrian_trans)

                # Pedestrian boundary area, radius = 0.1705 m.
                # This is drawn as the pedestrian's personal/body boundary,
                # not as the full 0.21 m collision circle.
                # Because robot radius is 0.04 m, contact condition is:
                # 0.1705 + 0.04 = 0.2105 ~= 0.21 m.
                pedestrian_boundary_fill = rendering.make_circle(
                    self._unit_x * pedestrian_boundary_radius,
                    res=80,
                    filled=True
                )
                pedestrian_boundary_fill.add_attr(pedestrian_trans)
                pedestrian_boundary_fill.set_color(1.0, 0.86, 0.86)
                self.viewer.add_geom(pedestrian_boundary_fill)
                setattr(self, f'p{i}_pedestrian_boundary_fill', pedestrian_boundary_fill)

                pedestrian_boundary_outline = rendering.make_circle(
                    self._unit_x * pedestrian_boundary_radius,
                    res=80,
                    filled=False
                )
                pedestrian_boundary_outline.add_attr(pedestrian_trans)
                pedestrian_boundary_outline.set_color(1.0, 0.0, 0.0)
                try:
                    pedestrian_boundary_outline.set_linewidth(2)
                except Exception:
                    pass
                self.viewer.add_geom(pedestrian_boundary_outline)
                setattr(self, f'p{i}_pedestrian_boundary_outline', pedestrian_boundary_outline)

                # Pedestrian head
                pedestrian_head = rendering.make_circle(
                    self._unit_x * pedestrian_radius
                )
                pedestrian_head.add_attr(pedestrian_trans)

                # Right shoulder
                rshoulder = rendering.make_capsule(
                    shoulder_length,
                    shoulder_diameter
                )
                rshoulder_trans = rendering.Transform(
                    rotation=ptheta + (np.pi / 2)
                )
                setattr(self, f'p{i}rshoulder_trans', rshoulder_trans)

                rshoulder.add_attr(rshoulder_trans)
                rshoulder.add_attr(pedestrian_trans)
                rshoulder.set_color(0.5, 0.5, 0.5)
                self.viewer.add_geom(rshoulder)

                # Left shoulder
                lshoulder = rendering.make_capsule(
                    -shoulder_length,
                    shoulder_diameter
                )
                lshoulder_trans = rendering.Transform(
                    rotation=ptheta + (np.pi / 2)
                )
                setattr(self, f'p{i}lshoulder_trans', lshoulder_trans)

                lshoulder.add_attr(lshoulder_trans)
                lshoulder.add_attr(pedestrian_trans)
                lshoulder.set_color(0.5, 0.5, 0.5)
                self.viewer.add_geom(lshoulder)

                # Head on top
                pedestrian_head.set_color(0, 0, 0)
                self.viewer.add_geom(pedestrian_head)

                # Pedestrian direction arrow
                a = 0.1705 * self._unit_x
                b = 0.036 * self._unit_y
                c = 0.108146 * self._unit_x

                parrow = rendering.FilledPolygon([(a, 0), (c, b), (c, -b)])
                parrow_trans = rendering.Transform(rotation=ptheta)
                setattr(self, f'p{i}arrow_trans', parrow_trans)

                parrow.add_attr(parrow_trans)
                parrow.add_attr(pedestrian_trans)

                # Pedestrian body / stride rectangle
                x = 0.0245 * self._unit_x
                y = 0.036 * self._unit_y
                u = 0.1705 * self._unit_x

                pstride = rendering.FilledPolygon(
                    [(x, y), (x, -y), (u, -y), (u, y)]
                )
                pstride_trans = rendering.Transform(rotation=ptheta)
                setattr(self, f'p{i}stride_trans', pstride_trans)

                pstride.add_attr(pstride_trans)
                pstride.add_attr(pedestrian_trans)
                pstride.set_color(0.39, 0.58, 0.93)
                self.viewer.add_geom(pstride)

                # Arrow on top
                parrow.set_color(0, 0, 0)
                self.viewer.add_geom(parrow)

        def to_screen_update(x, y):
            return (
                self._unit_x * (self._render_field_size + x),
                self._unit_y * (self._render_field_size + y)
            )

        def get_pedestrian_values_update(index):
            pedestrian = getattr(self, f'pedestrian{index}')
            prefix = f'p{index}'

            x = getattr(pedestrian, f'{prefix}x')
            y = getattr(pedestrian, f'{prefix}y')
            theta = getattr(pedestrian, f'{prefix}theta')

            return pedestrian, x, y, theta

        # Update goal
        self.target_trans.set_translation(
            *to_screen_update(self.target.x, self.target.y)
        )

        # Update robot
        self.arrow_trans.set_rotation(self.agent.theta)

        self.agentinner_trans.set_translation(
            *to_screen_update(self.agent.x, self.agent.y)
        )

        self.agentouter_trans.set_translation(
            *to_screen_update(self.agent.x, self.agent.y)
        )

        # Update pedestrians
        for i in range(1, 7):
            pedestrian, px, py, ptheta = get_pedestrian_values_update(i)

            getattr(self, f'pedestrian{i}_trans').set_translation(
                *to_screen_update(px, py)
            )

            getattr(self, f'p{i}arrow_trans').set_rotation(ptheta)

            getattr(self, f'p{i}stride_trans').set_rotation(ptheta)

            getattr(self, f'p{i}rshoulder_trans').set_rotation(
                ptheta + (np.pi / 2)
            )

            getattr(self, f'p{i}lshoulder_trans').set_rotation(
                ptheta + (np.pi / 2)
            )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class MovingEnv(BaseEnv):
    def __init__(
            self,
            seed: int = None,
            max_turn: float = np.pi/9,
            max_acceleration: float = 1,
            delta_t: float = 0.005,
            max_step: int = 1500,
            penalty: float = 0.001,
            break_value: float = 0.1,
    ):

        super(MovingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
        )

        self.agent = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )
        
        self.pedestrian1 = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )
        
        self.pedestrian2 = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )
        
        self.pedestrian3 = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )
        
        self.pedestrian4 = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )
	
        self.pedestrian5 = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )
       
        self.pedestrian6 = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )
 
