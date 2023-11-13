from typing import Any, List, Mapping, Union
from citylearn.data import ZERO_DIVISION_PLACEHOLDER
from citylearn.reward_function import RewardFunction

class SignalReward(RewardFunction):
    def __init__(self, env_metadata: Mapping[str, Any], signal: str = None, exponent: float = None, window: int = None):
        self.signal = signal
        self.exponent = exponent
        self.window = 1 if window is None else window
        self.reward_list = [[] for _ in range(self.window)]
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for _, (_, o) in enumerate(zip(self.env_metadata['buildings'], observations)):
            signal = 1.0 if self.signal is None else o[self.signal]
            reward = (max(0.0, o['net_electricity_consumption'])*signal)**self.exponent
            reward *= -1.0
            reward_list.append(reward)
        
        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward

class StorageSignalReward(SignalReward):
    def __init__(self, env_metadata: Mapping[str, Any], signal: str = None, exponent: float = None):
        self.previous_soc = None
        self.storage_keys = ['dhw', 'cooling', 'heating', 'electrical']
        super().__init__(env_metadata, signal=signal, exponent=exponent)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for i, (b, o) in enumerate(zip(self.env_metadata['buildings'], observations)):
            signal = 1.0 if self.signal is None else o[self.signal]
            current_soc = [o[f'{k}_storage_soc'] for k in self.storage_keys]
            soc_difference = [c - p for c, p in zip(current_soc, self.previous_soc[i])]

            if o['net_electricity_consumption'] > 0.0:
                capacities = [b[f'{k}_storage']['capacity'] for k in self.storage_keys]
                reward = sum([d*c for d, c in zip(soc_difference, capacities)])
                reward = ((abs(reward)*signal)**self.exponent)\
                    *(reward/abs(reward if reward > 0.0 else 1.0))
                reward *= -1.0
            
            else:
                reward = 0.0

            reward_list.append(reward)
            self.previous_soc[i] = current_soc

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward
    
    def reset(self):
        if self.env_metadata is not None:
            self.previous_soc = [
                [b[f'{k}_storage']['initial_soc'] for k in self.storage_keys] 
                for b in self.env_metadata['buildings']
            ]
        
        else:
            pass

class StorageCostReductionReward(RewardFunction):
    def __init__(self, env_metadata: Mapping[str, Any], **kwargs):
        self.electricity_pricing = []
        super().__init__(env_metadata, **kwargs)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []
        electricity_pricing = observations[0]['electricity_pricing']
        self.electricity_pricing.append(observations[0]['electricity_pricing'])
        self.electricity_pricing = sorted(list(set(self.electricity_pricing)))
        electricity_pricing_ix = self.electricity_pricing.index(electricity_pricing)

        for i, (b, o) in enumerate(zip(self.env_metadata['buildings'], observations)):
            reward = 0.0

            for k, v in b['action_metadata'].items():
                if v and 'storage' in k:
                    storage_electricity_consumption = o[f'{k}_electricity_consumption']
                    soc = o[f'{k}_soc']
                    
                    if abs(storage_electricity_consumption) < ZERO_DIVISION_PLACEHOLDER:
                        reward += -(1 - soc)*(len(self.electricity_pricing) - electricity_pricing_ix)

                    elif storage_electricity_consumption < 0.0:
                        reward += electricity_pricing_ix

                    elif 'electrical' not in k\
                        and storage_electricity_consumption > 0.0\
                            and o[f'{k.split("_")[0]}_electricity_consumption'] > storage_electricity_consumption:
                        reward += -self.electricity_pricing.index(electricity_pricing)
                    
                    else:
                        reward += 0.0

                else:
                    pass

            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward

class ComfortReward(RewardFunction):
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for o in observations:
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']
            set_point = o['indoor_dry_bulb_temperature_set_point']
            reward = abs(indoor_dry_bulb_temperature - set_point)*-1.0
            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)/len(reward_list)]

        else:
            reward = reward_list

        return reward
    
class ComfortandConsumptionReductionReward(RewardFunction):
    def __init__(self, env_metadata: Mapping[str, Any], multiplier: float = None):
        self.multiplier = 3.0 if multiplier is None else multiplier
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for o in observations:
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']
            set_point = o['indoor_dry_bulb_temperature_set_point']
            delta = indoor_dry_bulb_temperature - set_point

            if delta < 0.0:
                delta *= self.multiplier
            else:
                pass
            
            reward = abs(delta)*-1.0
            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)/len(reward_list)]

        else:
            reward = reward_list

        return reward

class PeakReductionReward(RewardFunction):
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)
        self.peak = None

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        electricity_consumption = [max(0.0, o['net_electricity_consumption']) for o in observations]

        if self.central_agent:
            electricity_consumption = [sum(electricity_consumption)]
        else:
            pass

        self.peak = [max(p, e) for p, e in zip(self.peak, electricity_consumption)]
        reward = [p - e for p, e in zip(self.peak, electricity_consumption)]

        return reward
    
    def reset(self):
        if self.env_metadata is not None:
            self.peak = [0.0] if self.central_agent else [0.0 for _ in self.env_metadata['buildings']]
        
        else:
            pass