import argparse
import concurrent.futures
from datetime import datetime
import importlib
import inspect
from multiprocessing import cpu_count
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, List, Mapping, Tuple, Union
from citylearn.agents.base import Agent as CityLearnAgent
from citylearn.building import LSTMDynamicsBuilding
from citylearn.citylearn import CityLearnEnv
from citylearn.utilities import read_json, write_json
import pandas as pd
import simplejson as json
from stable_baselines3.common.base_class import BaseAlgorithm as StableBaselines3Agent
from src.utilities import FileHandler

def run_work_order(work_order_filepath, max_workers=None, start_index=None, end_index=None, virtual_environment_path=None, windows_system=None):
    work_order_filepath = Path(work_order_filepath)

    if virtual_environment_path is not None:    
        if windows_system:
            virtual_environment_command = f'"{os.path.join(virtual_environment_path, "Scripts", "Activate.ps1")}"'
        
        else:
            virtual_environment_command = f'source "{os.path.join(virtual_environment_path, "bin", "activate")}"'
    
    else:
        virtual_environment_command = 'echo "No virtual environment"'

    with open(work_order_filepath,mode='r') as f:
        args = f.read()
    
    args = args.strip('\n').split('\n')
    start_index = 0 if start_index is None else start_index
    end_index = len(args) - 1 if end_index is None else end_index
    assert start_index <= end_index, 'start_index must be <= end_index'
    assert start_index < len(args), 'start_index must be < number of jobs'
    args = args[start_index:end_index + 1]
    args = [a for a in args if not a.startswith('#')]
    args = [f'{virtual_environment_command} && {a}' for a in args]
    max_workers = cpu_count() if max_workers is None else max_workers
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f'Will use {max_workers} workers for job.')
        print(f'Pooling {len(args)} jobs to run in parallel...')
        results = [executor.submit(subprocess.run,**{'args':a, 'shell':True}) for a in args]
            
        for future in concurrent.futures.as_completed(results):
            try:
                print(future.result())
            
            except Exception as e:
                print(e)

class CityLearnSimulation:
    @staticmethod
    def simulate(
        simulation_id: str, schema: Path, agent: str, buildings: List[int] = None, random_seed: int = None, reward_function: str = None, active_observations: List[str] = None, 
        active_actions: List[str] = None, inactive_observations: List[str] = None, 
        inactive_actions: List[str] = None, schema_kwargs: Mapping[str, Any] = None, env_kwargs: Mapping[str, Any] = None, 
        agent_kwargs: Mapping[str, Any] = None, reward_function_kwargs: Mapping[str, Any] = None, wrappers: List[str] = None, episodes: int = None, output_directory: Union[str, Path] = None, 
        train_episode_time_steps: List[List[int]] = None, evaluation_episode_time_steps: List[List[int]] = None, central_agent: bool = None, exclude_cooling_storage: bool = None, 
        exclude_heating_storage: bool = None, exclude_dhw_storage: bool = None, exclude_electrical_storage: bool = None, exclude_pv: bool = None, ignore_dynamics: bool = None,
        exclude_power_outage: bool = None,
    ):  
        agent_name = agent
        train_start_timestamp = datetime.utcnow()
        env, agent = CityLearnSimulation.get_agent(
            schema, agent, buildings=buildings, random_seed=random_seed, reward_function=reward_function, active_observations=active_observations, active_actions=active_actions, inactive_observations=inactive_observations, inactive_actions=inactive_actions, schema_kwargs=schema_kwargs, 
            env_kwargs=env_kwargs, agent_kwargs=agent_kwargs, reward_function_kwargs=reward_function_kwargs, wrappers=wrappers, train_episode_time_steps=train_episode_time_steps,
            central_agent=central_agent, exclude_cooling_storage=exclude_cooling_storage, exclude_heating_storage=exclude_heating_storage, exclude_dhw_storage=exclude_dhw_storage, 
            exclude_electrical_storage=exclude_electrical_storage, exclude_pv=exclude_pv, ignore_dynamics=ignore_dynamics, exclude_power_outage=exclude_power_outage
        )
        env_metadata = env.unwrapped.get_metadata()
        env, agent = CityLearnSimulation.learn(env, agent, episodes=episodes)
        train_end_timestamp = datetime.utcnow()
        simulation_summary = {
            'simulation_id': simulation_id,
            'agent': agent_name,
            'buildings': buildings,
            'random_seed': env.unwrapped.random_seed,
            'reward_function': reward_function,
            'central_agent': env.unwrapped.central_agent,
            'env_kwargs': env_kwargs,
            'agent_kwargs': agent_kwargs,
            'reward_function_kwargs': reward_function_kwargs,
            'wrappers': wrappers,
            'central_agent': env.unwrapped.central_agent,
            'exclude_cooling_storage': exclude_cooling_storage,
            'exclude_heating_storage': exclude_heating_storage,
            'exclude_dhw_storage': exclude_dhw_storage,
            'exclude_electrical_storage': exclude_electrical_storage,
            'exclude_pv': exclude_pv,
            'ignore_dynamics': ignore_dynamics,
            'exclude_power_outage': exclude_power_outage,
            'train_episodes': env.unwrapped.episode_tracker.episode,
            'train_episode_time_steps': env.unwrapped.episode_time_steps,
            'train_start_timestamp': train_start_timestamp,
            'train_end_timestamp': train_end_timestamp,
        }
        evaluations = []

        for episode_time_steps in evaluation_episode_time_steps:
            evaluation_start_timestamp = datetime.utcnow()
            env, agent, actions = CityLearnSimulation.evaluate(env, agent, episode_time_steps=episode_time_steps)
            evaluation_end_timestamp = datetime.utcnow()
            evaluation_summary = CityLearnSimulation.get_evaluation_summary(env)
            evaluation_summary['episode_reward_summary'] = evaluation_summary['episode_reward_summary'][-1]
            evaluations.append({
                'evaluation_episode_time_steps': [env.unwrapped.episode_tracker.episode_start_time_step, env.unwrapped.episode_tracker.episode_end_time_step],
                'evaluation_start_timestamp': evaluation_start_timestamp,
                'evaluation_end_timestamp': evaluation_end_timestamp,
                **evaluation_summary,
                'actions': actions,
            })
        
        simulation_summary['evaluations'] = evaluations
        simulation_summary['env_metadata'] = env_metadata
        simulation_summary['train_episode_reward_summary'] = env.unwrapped.episode_rewards[0: -(len(evaluation_episode_time_steps))]
        os.makedirs(output_directory, exist_ok=True)
        filepath = os.path.join(output_directory, f'{simulation_id}.json')
        write_json(filepath, simulation_summary)

    @staticmethod
    def get_evaluation_summary(env: CityLearnEnv) -> dict:
        evaluation = env.evaluate().pivot(index='name', columns='cost_function', values='value')
        data = {
            'evaluation': evaluation.to_dict('index'),
            'episode_reward_summary': env.unwrapped.episode_rewards,
            'episode_rewards': env.unwrapped.rewards,
            'time_series': CityLearnSimulation.get_time_series(env).to_dict('list'),
        }

        return data

    @staticmethod
    def get_time_series(env: CityLearnEnv):
        data_list = []

        for b in env.buildings:
            b: LSTMDynamicsBuilding
            data = pd.DataFrame({
                'bldg_name': b.name,
                'net_electricity_consumption': b.net_electricity_consumption,
                'net_electricity_consumption_cost': b.net_electricity_consumption_cost,
                'net_electricity_consumption_emission': b.net_electricity_consumption_emission,
                'net_electricity_consumption_without_storage': b.net_electricity_consumption_without_storage,
                'net_electricity_consumption_without_storage_and_partial_load': b.net_electricity_consumption_without_storage_and_partial_load,
                'net_electricity_consumption_without_storage_and_partial_load_and_pv': b.net_electricity_consumption_without_storage_and_partial_load_and_pv,
                'indoor_dry_bulb_temperature': b.indoor_dry_bulb_temperature,
                'indoor_dry_bulb_temperature_without_partial_load': b.indoor_dry_bulb_temperature_without_partial_load,
                'indoor_dry_bulb_temperature_set_point': b.energy_simulation.indoor_dry_bulb_temperature_set_point,
                'occupant_count': b.energy_simulation.occupant_count,
                'cooling_cop': b.cooling_device.get_cop(b.weather.outdoor_dry_bulb_temperature, heating=False),
                'cooling_electricity_consumption': b.cooling_electricity_consumption,
                'cooling_demand': b.cooling_demand,
                'cooling_demand_without_partial_load': b.cooling_demand_without_partial_load,
                'dhw_electricity_consumption': b.dhw_electricity_consumption,
                'dhw_demand': b.dhw_demand,
                'dhw_storage_electricity_consumption': b.dhw_storage_electricity_consumption,
                'dhw_storage_soc': b.dhw_storage.soc,
                'electrical_storage_electricity_consumption': b.electrical_storage_electricity_consumption,
                'electrical_storage_soc': b.electrical_storage.soc,
                'non_shiftable_load': b.non_shiftable_load,
                'non_shiftable_load_electricity_consumption': b.non_shiftable_load_electricity_consumption,
                'energy_from_electrical_storage': b.energy_from_electrical_storage,
                'energy_from_dhw_storage': b.energy_from_dhw_storage,
                'energy_to_non_shiftable_load': b.energy_to_non_shiftable_load,
                'energy_from_dhw_device': b.energy_from_dhw_device,
                'energy_from_cooling_device': b.energy_from_cooling_device,
                'energy_to_electrical_storage': b.energy_to_electrical_storage,
                'energy_from_dhw_device_to_dhw_storage': b.energy_from_dhw_device_to_dhw_storage,
                'solar_generation': b.solar_generation,
                'power_outage': b.energy_simulation.power_outage,
                'pricing': b.pricing.electricity_pricing,
                'carbon_intensity': b.carbon_intensity.carbon_intensity
            })
            data['time_step'] = data.index
            data_list.append(data)

        return pd.concat(data_list, ignore_index=True)

    @staticmethod
    def evaluate(env: CityLearnEnv, agent: Union[CityLearnAgent, StableBaselines3Agent], episode_time_steps: List[int] = None) -> Tuple[
        CityLearnEnv, Union[CityLearnAgent, StableBaselines3Agent], List[List[Mapping[str, float]]]
    ]:
        if episode_time_steps is not None:
            env.unwrapped.episode_time_steps = [episode_time_steps]
        
        else:
            pass

        observations = env.reset()
        actions_list = []

        while not env.done:
            if isinstance(agent, CityLearnAgent):
                actions = agent.predict(observations, deterministic=True)
                actions_list.append(env.unwrapped._parse_actions(actions))
            
            elif isinstance(agent, StableBaselines3Agent):
                actions, _ = agent.predict(observations, deterministic=True)
                actions_list.append(env.unwrapped._parse_actions([actions]))

            else:
                raise Exception(f'Unknown agent type: {type(agent)}')

            observations, _, _, _ = env.step(actions)

        return env, agent, actions_list

    @staticmethod
    def learn(env: CityLearnEnv, agent: Union[CityLearnAgent, StableBaselines3Agent], episodes: int) -> Tuple[CityLearnEnv, Union[CityLearnAgent, StableBaselines3Agent]]:
        kwargs = {}

        if isinstance(agent, CityLearnAgent):
            kwargs = {**kwargs, 'episodes': episodes}
            agent.learn(**kwargs)
        
        elif isinstance(agent, StableBaselines3Agent):
            kwargs = {**kwargs, 'total_timesteps': episodes*env.unwrapped.time_steps}
            agent = agent.learn(**kwargs)

        else:
            raise Exception(f'Unknown agent type: {type(agent)}')

        return env, agent

    @staticmethod
    def get_agent(
        schema: Path, agent: str, buildings: List[int] = None, random_seed: int = None, reward_function: str = None, active_observations: List[str] = None, active_actions: List[str] = None, inactive_observations: List[str] = None, inactive_actions: List[str] = None, schema_kwargs: Mapping[str, Any] = None, 
        env_kwargs: Mapping[str, Any] = None, agent_kwargs: Mapping[str, Any] = None, reward_function_kwargs: Mapping[str, Any] = None, train_episode_time_steps: List[List[int]] = None, wrappers: List[str] = None,
        central_agent: bool = None, exclude_cooling_storage: bool = None, exclude_heating_storage: bool = None, exclude_dhw_storage: bool = None, 
        exclude_electrical_storage: bool = None, exclude_pv: bool = None, ignore_dynamics: bool = None, exclude_power_outage: bool = None
    ) -> Tuple[CityLearnEnv, Union[CityLearnAgent, StableBaselines3Agent]]:

        root_directory = os.path.split(schema.absolute())[0]
        schema = read_json(schema)
        schema['root_directory'] = root_directory

        # update general schema settings
        schema_kwargs = {} if schema_kwargs is None else schema_kwargs
        schema = {
            **schema,
            **schema_kwargs,
            'central_agent': central_agent,
            'episode_time_steps': train_episode_time_steps,
        }

        # set active observations and action
        for t, a, b in zip(
            ['observations', 'actions', 'observations', 'actions'], 
            [active_observations, active_actions, inactive_observations, inactive_actions], 
            [True, True, False, False]
        ):
            if a is not None:
                for k in schema[t]:
                    if k in a:
                        schema[t][k]['active'] = b
                    
                    else:
                        schema[t][k]['active'] = not b
            else:
                pass

        # update reward function in schema
        schema['reward_function']['type'] = reward_function if reward_function is not None else schema['reward_function']['type']
        schema['reward_function']['attributes'] = reward_function_kwargs if reward_function_kwargs is not None or reward_function is not None \
            else schema['reward_function']['attributes']
        
        # update buildings
        if buildings is not None:
            valid_buildings = [list(schema['buildings'].keys())[i] for i in buildings]
        else:
            valid_buildings = list(range(len(schema['buildings'])))

        buildings = {}
        exclude_devices = {
            'cooling_storage': exclude_cooling_storage,
            'heating_storage': exclude_heating_storage,
            'dhw_storage': exclude_dhw_storage,
            'electrical_storage': exclude_electrical_storage,
            'pv': exclude_pv,
        }

        for b in valid_buildings:
            for k, v in exclude_devices.items():
                if v is not None and v:
                    schema['buildings'][b][k] = None
                else:
                    pass

            if exclude_power_outage:
                _ = schema['buildings'][b].pop('power_outage', None)

            else:
               pass

            buildings[b] = schema['buildings'][b]

        schema['buildings'] = buildings
        
        # construct env
        env_kwargs = {} if env_kwargs is None else env_kwargs
        env_kwargs['random_seed'] = random_seed if random_seed is not None else env_kwargs.get('random_seed', None)
        env = CityLearnSimulation.env_creator(schema, wrappers=wrappers, **env_kwargs)

        for b in env.unwrapped.buildings:
            if isinstance(b, LSTMDynamicsBuilding):
                b.ignore_dynamics = ignore_dynamics
            else:
                pass

        # construct agent
        agent_kwargs = {} if agent_kwargs is None else agent_kwargs

        if random_seed is not None:
            if isinstance(agent, CityLearnAgent):
                agent_kwargs['random_seed'] = random_seed
        
        elif isinstance(agent, StableBaselines3Agent):
            agent_kwargs['seed'] = random_seed

        else:
            raise Exception(f'Unknown agent type: {type(agent)}')
        
        if 'rbc' in agent_kwargs.keys():
            rbc = agent_kwargs['rbc']
            rbc_kwargs = agent_kwargs.get('rbc_kwargs', {})
            rbc_module = '.'.join(rbc.split('.')[0:-1])
            rbc_name = rbc.split('.')[-1]
            rbc_constructor = getattr(importlib.import_module(rbc_module), rbc_name)
            rbc = rbc_constructor(env=env, **rbc_kwargs)
            agent_kwargs['rbc'] = rbc

        else:
            pass
            
        agent_module = '.'.join(agent.split('.')[0:-1])
        agent_name = agent.split('.')[-1]
        agent_constructor = getattr(importlib.import_module(agent_module), agent_name)
        agent = agent_constructor(env=env, **agent_kwargs)

        return env, agent

    @staticmethod
    def env_creator(schema: Union[dict, Path, str], wrappers: List[str] = None, **kwargs) -> CityLearnEnv:
        env = CityLearnEnv(schema, **kwargs)
        wrappers = [] if wrappers is None else wrappers

        if wrappers is not None:
            for wrapper in wrappers:
                wrapper_module = '.'.join(wrapper.split('.')[0:-1])
                wrapper_name = wrapper.split('.')[-1]
                wrapper_constructor = getattr(importlib.import_module(wrapper_module), wrapper_name)
                env = wrapper_constructor(env)
            
            else:
                pass

        return env

def main():
    parser = argparse.ArgumentParser(prog='citylearn-challenge-2023', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')

    # run work order
    subparser_run_work_order = subparsers.add_parser('run_work_order')
    subparser_run_work_order.add_argument('work_order_filepath', type=Path)
    subparser_run_work_order.add_argument('-m', '--max_workers', dest='max_workers', type=int)
    subparser_run_work_order.add_argument('-s', '--start_index', default=0, dest='start_index', type=int)
    subparser_run_work_order.add_argument('-e', '--end_index', default=None, dest='end_index', type=int)
    subparser_run_work_order.set_defaults(func=run_work_order)

    subparser_general = subparsers.add_parser('general')
    subparser_general.add_argument('-d', '--output_directory', dest='output_directory', type=str, default=FileHandler.DEFAULT_OUTPUT_DIRECTORY)
    general_subparsers = subparser_general.add_subparsers(title='subcommands', required=True, dest='subcommands')
    
    # CityLearn simulation
    subparser_simulate_citylearn = general_subparsers.add_parser('simulate-citylearn')
    subparser_simulate_citylearn.add_argument('agent', type=str)
    subparser_simulate_citylearn.add_argument('-s', '--schema', dest='schema', default=os.path.join(FileHandler.SCHEMA_DIRECTORY, 'schema.json'), type=Path)
    subparser_simulate_citylearn.add_argument('-b', '--buildings', dest='buildings', type=int, nargs='+')
    subparser_simulate_citylearn.add_argument('-x', '--simulation_id', dest='simulation_id', type=str)
    subparser_simulate_citylearn.add_argument('-w', '--reward_function', dest='reward_function', type=str, default=None)
    subparser_simulate_citylearn.add_argument('-ao', '--active_observations', dest='active_observations', type=str, nargs='+')
    subparser_simulate_citylearn.add_argument('-aa', '--active_actions', dest='active_actions', type=str, nargs='+')
    subparser_simulate_citylearn.add_argument('-io', '--inactive_observations', dest='inactive_observations', type=str, nargs='+')
    subparser_simulate_citylearn.add_argument('-ia', '--inactive_actions', dest='inactive_actions', type=str, nargs='+')
    subparser_simulate_citylearn.add_argument('-ke', '--env_kwargs', dest='env_kwargs', type=json.loads, default=None)
    subparser_simulate_citylearn.add_argument('-ka', '--agent_kwargs', dest='agent_kwargs', type=json.loads, default=None)
    subparser_simulate_citylearn.add_argument('-kr', '--reward_function_kwargs', dest='reward_function_kwargs', type=json.loads, default=None)
    subparser_simulate_citylearn.add_argument('-e', '--episodes', dest='episodes', type=int, default=1)
    subparser_simulate_citylearn.add_argument('-tt', '--train_episode_time_steps', dest='train_episode_time_steps', type=int, nargs='+', action='append')
    subparser_simulate_citylearn.add_argument('-te', '--evaluation_episode_time_steps', dest='evaluation_episode_time_steps', type=int, nargs='+', action='append')
    subparser_simulate_citylearn.add_argument('-a', '--wrappers', dest='wrappers', type=str, nargs='+')
    subparser_simulate_citylearn.add_argument('-c', '--central_agent', dest='central_agent', action='store_true')
    subparser_simulate_citylearn.add_argument('-ecs', '--exclude_cooling_storage', dest='exclude_cooling_storage', action='store_true')
    subparser_simulate_citylearn.add_argument('-ehs', '--exclude_heating_storage', dest='exclude_heating_storage', action='store_true')
    subparser_simulate_citylearn.add_argument('-eds', '--exclude_dhw_storage', dest='exclude_dhw_storage', action='store_true')
    subparser_simulate_citylearn.add_argument('-ees', '--exclude_electrical_storage', dest='exclude_electrical_storage', action='store_true')
    subparser_simulate_citylearn.add_argument('-epv', '--exclude_pv', dest='exclude_pv', action='store_true')
    subparser_simulate_citylearn.add_argument('-etd', '--ignore_dynamics', dest='ignore_dynamics', action='store_true')
    subparser_simulate_citylearn.add_argument('-epo', '--exclude_power_outage', dest='exclude_power_outage', action='store_true')
    subparser_simulate_citylearn.add_argument('-r', '--random_seed', dest='random_seed', type=int, default=0)
    subparser_simulate_citylearn.set_defaults(func=CityLearnSimulation.simulate)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())