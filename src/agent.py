from typing import Any, List, Mapping, Union
from citylearn.agents.base import Agent
from citylearn.agents.rbc import RBC
from citylearn.citylearn import CityLearnEnv

class DummyController(Agent):
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        actions = [[0.0 if 'storage' in n else None for n in a] for a in self.action_names]
        self.actions = actions
        self.next_time_step()
        
        return actions

class AustinEnergyTOURBC(RBC):
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)    

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        actions = []

        for a, n, o in zip(self.action_names, self.observation_names, observations):
            hour = o[n.index('hour')]
            day_type = o[n.index('day_type')]
            actions_ = []

            for _ in a:
                if 2 <= day_type <= 6:
                    if hour >= 22 or hour < 7:
                        actions_.append(1.0/9.0)
                    
                    elif 15 <= hour < 18:
                        actions_.append(-0.5/3.0)

                    else:
                        actions_.append(-0.5/12.0)
                
                else:
                    actions_.append(1/48)
            
            actions.append(actions_)

        self.actions = actions
        self.next_time_step()
        
        return actions
    
class AustinEnergyEmissionReductionRBC(RBC):
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        actions = []

        for a, n, o in zip(self.action_names, self.observation_names, observations):
            hour = o[n.index('hour')]
            actions_ = []

            for _ in a:
                if hour <= 7:
                    actions_.append(1.0/7.0)

                elif 12 <= hour <= 22:
                    actions_.append(-1.0/11.0)

                else:
                    actions_.append(0.0)
                
            actions.append(actions_)

        self.actions = actions
        self.next_time_step()
        
        return actions
    
class PeakReductionRBC(RBC):
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        actions = []

        for a, n, o in zip(self.action_names, self.observation_names, observations):
            hour = o[n.index('hour')]
            actions_ = []

            for _ in a:
                if (hour == 6) or (11 <= hour <= 22):
                    actions_.append(-1.0/14.0)

                elif hour < 6:
                    actions_.append(1.0/5.0)

                else:
                    actions_.append(0.0)
                
            actions.append(actions_)

        self.actions = actions
        self.next_time_step()
        
        return actions