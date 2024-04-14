import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import glob
import os
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go

class FanManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.fan_dfs = {}
        extension = '*.csv'
        files = glob.glob(os.path.join(self.data_dir, extension))
        for file_path in files:
            file_name = os.path.basename(file_path)
            fan_model = file_name.split('.')[0]
            # print(fan_model, file_path)
            df = pd.read_csv(file_path)
            self.fan_dfs[fan_model] = df    

    def find_power(self, fan_model, voltage):
        df = self.fan_dfs[fan_model]
        # print(df)
        # print(f"LALALALA: {voltage}")
        matching_fields = df[df['Control Voltage'] == voltage]
        # print(f"LALALALA: {matching_fields}")
        if not matching_fields.empty:
            # import pdb; pdb.set_trace()
            return matching_fields.iloc[0]['Power']
        return None

    def find_operating_point(self, fan_model, target_pressure, target_airflow):
        df = self.fan_dfs[fan_model]
        best_match = {'voltage': None, 'pressure': None, 'airflow': None, 'distance': np.inf, 'efficiency': None, 'power': None}
        tolerance = 0.025  # 2.5% tolerance

        #First lets check to make sure the requested pressure and airflow is possible for this fan
        max_pressure = df['Air Pressure Pa'].max()
        min_pressure = df['Air Pressure Pa'].min()
        max_airflow = df['Airflow2'].max()
        min_airflow = df['Airflow2'].min()

        if target_pressure < min_pressure or target_pressure > max_pressure or target_airflow < min_airflow or target_airflow > max_airflow:
            return None, None  # Immediately return None if targets are out of range


        #calculate proximity thresholds:
        acceptable_range_factor = 0.10  # 10% acceptable range for airflow and pressure

        voltages = df['Control Voltage'].unique()
        fig = go.Figure()

        # plt.figure(figsize=(10, 6))

        for voltage in voltages:
            subset = df[df['Control Voltage'] == voltage].sort_values('Air Pressure Pa')
            subset_unique = subset.drop_duplicates(subset=['Air Pressure Pa'])

            x = subset_unique['Air Pressure Pa']
            y = subset_unique['Airflow2']
            
            if len(x) > 3:
                cubic_interp = interp1d(x, y, kind='cubic')
                xnew = np.linspace(x.min(), x.max(), 300)
                ynew = cubic_interp(xnew)
                
                # Find the point closest to the target on this curve
                for px, py in zip(xnew, ynew):
                    if (1 - acceptable_range_factor) * target_airflow <= py <= (1 + acceptable_range_factor) * target_airflow and \
                       (1 - acceptable_range_factor) * target_pressure <= px <= (1 + acceptable_range_factor) * target_pressure:
                        distance = euclidean((px, py), (target_pressure, target_airflow))
                        if distance < best_match['distance']:
                            best_match.update({'voltage': voltage, 'pressure': px, 'airflow': py, 'distance': distance})
                
                fig.add_trace(go.Scatter(x=xnew, y=ynew, mode='lines', name=f'{voltage}V - Cubic Interpolation'))
            else:
                return None, None
        if best_match['voltage'] is not None:
            fig.add_trace(go.Scatter(x=[best_match['pressure']], y=[best_match['airflow']], mode='markers', marker=dict(color='black', size=12), name='Target Operating Point'))
            # print(best_match['voltage'])
            best_match['power'] = self.find_power(fan_model, best_match['voltage'])
            best_match['efficiency'] = ((best_match['airflow']/best_match['power'])/3600)*best_match['pressure']*100
            return fig, best_match
        else:
            return None, None
            
    def find_best_fan(self, target_pressure, target_airflow):
        max_efficiency = 0
        fig, operating_point = None, None
        fan_model = None
        test = None
        for fan in self.fan_dfs.keys():
            r_fig, best_match = self.find_operating_point(fan, target_pressure, target_airflow)
            if best_match is not None:
                if best_match['efficiency'] > max_efficiency:
                    max_efficiency = best_match['efficiency']
                    fig, operating_point = r_fig, best_match
                    fan_model = fan
                    test = best_match
        if fan_model is None:
            return None
        print(test['distance'])
        return fig, fan_model, operating_point

    def find_fan_combination(self, target_pressure, target_airflow):
        # Step 1: Check if target pressure is in range for any fan
        pressure_in_range = any(df['Air Pressure Pa'].max() * 1.025 >= target_pressure for df in self.fan_dfs.values())
        if not pressure_in_range:
            return "Error: Target pressure is out of range for all fans."

        # Step 2: Check for airflow and find combinations if needed
        fraction = 1
        while True:
            required_airflow = target_airflow / fraction
            best_fan, best_match, fig = None, None, None
            for fan_model in self.fan_dfs.keys():
                r_fig, match = self.find_operating_point(fan_model, target_pressure, required_airflow)
                if match:
                    best_fan, best_match = fan_model, match
                    fig = r_fig
                    break  # Found at least one fan that can meet the adjusted airflow requirement
            
            if best_fan:
                return fraction, fig, fan_model, best_match
            
            fraction *= 2  # Double the number of fans (halve the required airflow per fan)

        # If we exit the loop, no combination of fans was found
        return None
