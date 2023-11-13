# # Baseline
# # -----> No Control
python -m src.simulate general simulate-citylearn src.agent.DummyController -x no_control-b2-no_objective-no_der-0 -b 1 2 -ia dhw_storage electrical_storage cooling_device -tt 0 312 -te 0 312 -te 313 719 -c -eds -ees -epv -epo -r 0
# # 
# # 
# # Baseline
# # -----> No Control with PV
python -m src.simulate general simulate-citylearn src.agent.DummyController -x no_control-b2-no_objective-pv-0 -b 1 2 -ia dhw_storage electrical_storage cooling_device -tt 0 312 -te 0 312 -te 313 719 -c -eds -ees -epo -r 0
# # 
# # 
# # RBC
# # -----> Control DHW
# # ----------> Optimize Emissions
python -m src.simulate general simulate-citylearn src.agent.AustinEnergyEmissionReductionRBC -x rbc-b1-emission-dhw-0 -b 1 2 -ao hour -aa dhw_storage -tt 0 312 -te 0 312 -te 313 719 -c -ees -epv -epo -r 0
# #
# # ----------> Optimize Cost
python -m src.simulate general simulate-citylearn src.agent.AustinEnergyTOURBC -x rbc-b1-cost-dhw-0 -b 1 2 -ao day_type hour -aa dhw_storage -tt 0 312 -te 0 312 -te 313 719 -c -ees -epv -epo -r 0
# # ******************************* END *******************************
# #
# #
# # RBC
# # -----> Control Battery-PV
# # ----------> Optimize Emissions
python -m src.simulate general simulate-citylearn src.agent.AustinEnergyEmissionReductionRBC -x rbc-b1-emission-ess_pv-0 -b 1 2 -ao hour -aa electrical_storage -tt 0 312 -te 0 312 -te 313 719 -c -eds -epo -r 0
# #
# # ----------> Optimize Cost
python -m src.simulate general simulate-citylearn src.agent.AustinEnergyTOURBC -x rbc-b1-cost-ess_pv-0 -b 1 2 -ao day_type hour -aa electrical_storage -tt 0 312 -te 0 312 -te 313 719 -c -eds -epo -r 0
# # ----------> Optimize Peak
python -m src.simulate general simulate-citylearn src.agent.PeakReductionRBC -x rbc-b2-peak-ess_pv-0 -b 1 2 -ao hour -aa electrical_storage -tt 0 312 -te 0 312 -te 313 719 -c -eds -epo -r 0
# # ******************************* END *******************************
# #
# #
# # RBC
# # -----> Control DHW and Battery-PV
# # ----------> Optimize Emissions
python -m src.simulate general simulate-citylearn src.agent.AustinEnergyEmissionReductionRBC -x rbc-b1-emission-dhw_ess_pv-0 -b 1 2 -ao hour -aa dhw_storage electrical_storage -tt 0 312 -te 0 312 -te 313 719 -c -epo -r 0
# #
# # ----------> Optimize Cost
python -m src.simulate general simulate-citylearn src.agent.AustinEnergyTOURBC -x rbc-b1-cost-dhw_ess_pv-0 -b 1 2 -ao day_type hour -aa dhw_storage electrical_storage -tt 0 312 -te 0 312 -te 313 719 -c -epo -r 0
# # ******************************* END *******************************
# #
# #
# # RLC
# # -----> Control DHW
# # ----------> Optimize Emissions
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-emission-dhw-0 -b 1 -w src.reward_function.SignalReward -ao hour day_type dhw_storage_soc carbon_intensity net_electricity_consumption -aa dhw_storage -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"signal\": \"carbon_intensity\", \"exponent\": 1.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -ees -epv -etd -epo -r 0
# #
# # ----------> Optimize Cost
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-cost-dhw-0 -b 1 -w src.reward_function.SignalReward -ao hour day_type dhw_storage_soc electricity_pricing net_electricity_consumption -aa dhw_storage -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"signal\": \"electricity_pricing\", \"exponent\": 1.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -ees -epv -etd -epo -r 0
# # ******************************* END *******************************
# #
# #
# # RLC
# # -----> Control Battery-PV
# # ----------> Optimize Emissions
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-emission-ess_pv-0 -b 1 -w src.reward_function.SignalReward -ao hour day_type electrical_storage_soc carbon_intensity solar_generation net_electricity_consumption -aa electrical_storage -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"signal\": \"carbon_intensity\", \"exponent\": 1.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -eds -etd -epo -r 0
# #
# # ----------> Optimize Cost
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-cost-ess_pv-0 -b 1 -w src.reward_function.SignalReward -ao hour day_type electrical_storage_soc electricity_pricing solar_generation net_electricity_consumption -aa electrical_storage -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"signal\": \"electricity_pricing\", \"exponent\": 1.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -eds -etd -epo -r 0
# #
# # ----------> Optimize Peak
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b2-peak-ess_pv-10 -b 1 2 -w src.reward_function.SignalReward -ao hour day_type electrical_storage_soc solar_generation net_electricity_consumption -aa electrical_storage -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"exponent\": 1.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -eds -etd -epo -r 0
# # ******************************* END *******************************
# #
# #
# # RLC
# # -----> Control DHW and Battery-PV
# # ----------> Optimize Emissions
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-emission-dhw_ess_pv-0 -b 1 -w src.reward_function.SignalReward -ao hour day_type dhw_storage_soc electrical_storage_soc carbon_intensity solar_generation net_electricity_consumption -aa dhw_storage electrical_storage -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"signal\": \"carbon_intensity\", \"exponent\": 1.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -etd -epo -r 0
# #
# # ----------> Optimize Cost
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-cost-dhw_ess_pv-0 -b 1 -w src.reward_function.SignalReward -ao hour day_type dhw_storage_soc electrical_storage_soc electricity_pricing solar_generation net_electricity_consumption -aa dhw_storage electrical_storage -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"signal\": \"electricity_pricing\", \"exponent\": 1.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -etd -epo -r 0
# # # ******************************* END *******************************
# #
# #
# # RLC
# # -----> Control Heat Pump
# # ----------> Optimize Comfort
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-discomfort-hp-0 -b 1 -w src.reward_function.ComfortReward -ao hour day_type outdoor_dry_bulb_temperature indoor_dry_bulb_temperature indoor_dry_bulb_temperature_set_point indoor_dry_bulb_temperature_delta -aa cooling_device -ka "{\"policy\": \"MlpPolicy\"}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -epo -r 0
# ----------> Optimize Comfort and Consumption (Cost and Emission by proxy)
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-discomfort_consumption-hp-0 -b 1 -w src.reward_function.ComfortandConsumptionReductionReward -ao hour day_type outdoor_dry_bulb_temperature indoor_dry_bulb_temperature indoor_dry_bulb_temperature_set_point indoor_dry_bulb_temperature_delta -aa cooling_device -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"multiplier\": 3.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -epo -r 0
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-discomfort_consumption-hp-1 -b 1 -w src.reward_function.ComfortandConsumptionReductionReward -ao hour day_type outdoor_dry_bulb_temperature indoor_dry_bulb_temperature indoor_dry_bulb_temperature_set_point indoor_dry_bulb_temperature_delta -aa cooling_device -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"multiplier\": 6.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -epo -r 0
python -m src.simulate general simulate-citylearn stable_baselines3.SAC -x rlc-b1-discomfort_consumption-hp-2 -b 1 -w src.reward_function.ComfortandConsumptionReductionReward -ao hour day_type outdoor_dry_bulb_temperature indoor_dry_bulb_temperature indoor_dry_bulb_temperature_set_point indoor_dry_bulb_temperature_delta -aa cooling_device -ka "{\"policy\": \"MlpPolicy\"}" -kr "{\"multiplier\": 12.0}" -e 150 -tt 0 312 -te 0 312 -te 313 719 -a citylearn.wrappers.NormalizedObservationWrapper citylearn.wrappers.StableBaselines3Wrapper -c -epo -r 0
# ******************************* END *******************************
