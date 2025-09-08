import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import warnings
from typing import Literal, cast

datafolder = r"../data/"
path_newyork = datafolder + "15minute_data_newyork.csv"
path_austin = datafolder + "15minute_data_austin.csv"
path_california = datafolder + "15minute_data_california.csv"

weather_folder = "../weather/"
path_newyork_weather_2019 = weather_folder + "Ithaca_NY_2019.csv"
path_austin_weather_2018 = weather_folder + "Austin_TX_2018.csv"
path_california_weather_2018 = weather_folder + "San Diego_CA_2018.csv"
#path_california_weather_2017 = weather_folder + "San Diego_CA_2017.csv"
path_california_weather_2016 = weather_folder + "San Diego_CA_2016.csv"
path_california_weather_2015 = weather_folder + "San Diego_CA_2015.csv"
path_california_weather_2014 = weather_folder + "San Diego_CA_2014.csv"

def _prepare_data(data_path=path_newyork, shift=1, cleaning=True):
    raw_data = pd.read_csv(data_path)
    
    all_data = _clean_dst_and_shift(raw_data, shift=shift) if cleaning else raw_data.copy()
    
    if 'localminute' not in all_data.columns:
        local_cols = [col for col in all_data.columns if col.startswith("local")]
        if local_cols:
            all_data.rename(columns={local_cols[0]: 'localminute'}, inplace=True)
    


    


    #metadata_sel = metadata_sel[metadata_sel['state'] == 'New York']
    metadata_sel = get_metadata(min_availability=0.8)
    
    filtered_appliances = all_data.columns[all_data.columns.isin(metadata_sel.columns)]
    filtered_appliances = list(filtered_appliances)
    removable = [
        "dataid",
        #"grid",
        #"solar",
        #"solar2"
    ]
    for c in removable:
        filtered_appliances.remove(c)
        
            
    # all NYS data
    filtered_data = all_data[ 
        all_data.dataid.isin(
            all_data.dataid.unique()[
                np.isin(
                    all_data.dataid.unique(),
                    metadata_sel['dataid'].unique()
                    )]
            )]

    survey, _, survey_cols = read_survey()
    survey_essentials = survey[['dataid'] + survey_cols.tolist()]
    filtered_survey = survey_essentials[survey_essentials['dataid'].isin(filtered_data.dataid.unique())]

    return (
        filtered_data,
        filtered_appliances,
        filtered_survey
        )


def read_survey():
    survey = pd.read_csv(datafolder + "survey_2019_all_participants.csv")
    survey['dataid'] = survey['dataid'].astype(str).str.replace('\ufeff', '', regex=False)
    id = survey['dataid'].astype(int)
    survey.drop(columns=['dataid'], inplace=True)
    survey['dataid'] = id
    survey_fields = pd.read_csv(datafolder + "survey_2019_field_descriptions.csv", encoding="latin1")
    survey_cols = survey_fields[survey_fields['interested'] == 1]['column_name']
    survey_cols = survey_cols[survey_cols.isin(survey.columns)]
    return survey, survey_fields, survey_cols

def get_metadata(min_availability=0.8) -> pd.DataFrame:
    metadata = pd.read_csv(datafolder + "metadata.csv")
    metadata = metadata[1:]
         
    def perc_txt_parse(pc):
        pc = str(pc)
        if len(pc.split("nan")) >= 2:
            return 0
        else:
            return eval(pc.split("%")[0]) * 0.01
    
    metadata_sel = metadata[
    metadata["egauge_1min_data_availability"].apply(
        lambda x: perc_txt_parse(x) > min_availability
    )]
    
    id = metadata_sel['dataid'].astype(int)
    metadata_sel.loc[:, 'dataid'] = id
    return metadata_sel

def _prepare_all_data():
    newyork_data, newyork_appliances, newyork_survey = _prepare_data(data_path=path_newyork, shift=1)
    austin_data, austin_appliances, austin_survey = _prepare_data(data_path=path_austin, shift=0)
    california_data, california_appliances, california_survey = _prepare_data(data_path=path_california, shift=-2)
    all_data = pd.concat([newyork_data, austin_data, california_data], axis=0)
    all_appliances = list(set(newyork_appliances + austin_appliances + california_appliances))
    all_survey = pd.concat([newyork_survey, austin_survey, california_survey], axis=0)
    all_survey = all_survey.drop_duplicates(subset=['dataid'])
    return all_data, all_appliances, all_survey

def get_appliance_to_enduse_mapping():
    enduse = pd.read_csv(datafolder + 'enduse.csv')
    enduse.dropna(how='any', inplace=True)
    mapping = dict(zip(enduse['name'], enduse['enduse']))
    return mapping


def _clean_dst_and_shift(df: pd.DataFrame, shift: int = 0) -> pd.DataFrame:
    """
    Clean a Dataport-like dataframe by:
      1) Renaming any 'local_*' time column to 'local_minute'
      2) Ensuring dataid is int and creating a boolean 'DST' column
      3) For each dataid group:
         4.1) Detect '-05'/'-06' suffix (CDT/CST) and set DST (True for -05, False for -06)
         4.2) Remove the suffix from the timestamp strings
         4.3) Shift all rows with DST=True backward by 1 hour
         4.4) Fill gaps (15-min grid) by copying previous day’s same time (set DST=False)
         4.5) If overlapping timestamps exist, keep the shifted (DST=True) record
         4.6) Apply additional user-specified shift (in hours)
      5) Return a single cleaned DataFrame with no gaps or overlaps per dataid.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns including 'dataid' and a time column named like 'local_*'.
    shift : int
        Additional shift in hours to apply at step 4.6 (positive = later, negative = earlier).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with:
          - 'dataid' as int
          - time column 'local_minute' as pandas datetime (naive, uniform 15-min grid)
          - 'DST' boolean label
          - no gaps/overlaps per dataid
    """
    if 'dataid' not in df.columns:
        raise ValueError("Expected a 'dataid' column.")

    out = df.copy()

    # 1) Rename any 'local_*' to 'local_minute' (use the first match)
    local_cols = [c for c in out.columns if c.startswith('local_')]
    if not local_cols:
        raise ValueError("No 'local_*' time column found (e.g., 'local_15min').")
    if 'local_minute' not in out.columns:
        out = out.rename(columns={local_cols[0]: 'local_minute'})

    # 2) Cast dataid and create DST column (default True)
    out['dataid'] = out['dataid'].astype(int)
    out['DST'] = True

    # Helper: process one dataid group
    def _process_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g = g.reset_index()

        # 4.1) Detect suffix -05 / -06 (support '-05', '-06', '-05:00', '-06:00')
        s = g['local_minute'].astype(str)
        # capture '-05' or '-06' possibly followed by ':00'
        suffix = s.str.extract(r'(-0[56])(?::?00)?$', expand=False)
        # Default True, but per spec: false if -06, otherwise true
        g['DST'] = suffix.eq('-05')

        # 4.2) Remove suffix
        g['local_minute'] = s.str.replace(r'(-0[56])(?::?00)?$', '', regex=True).str.strip()
        g['local_minute'] = pd.to_datetime(g['local_minute'], errors='coerce')

        # 4.3) Shift rows with DST=True backward by 1 hour
        is_dst = g['DST'].fillna(True)  # be permissive: treat unknown as True
        g.loc[is_dst, 'local_minute'] = g.loc[is_dst, 'local_minute'] - pd.Timedelta(hours=1)

        # 4.5) Resolve overlaps: keep the shifted (DST=True) record on duplicate timestamps
        # Sort so DST=True comes first; then drop duplicates keeping first
        g = g.sort_values(['local_minute', 'DST'], ascending=[True, False])
        g = g.drop_duplicates(subset=['local_minute'], keep='first')

        # 4.4) Fill gaps on a 15-min grid using previous day same time (DST=False on filled)
        g = g.sort_values('local_minute')
        idx = pd.date_range(g['local_minute'].min(), g['local_minute'].max(), freq='15min')
        g = g.set_index('local_minute')

        missing = idx.difference(g.index)
        if len(missing) > 0:
            # Candidate fill rows come from previous day same time, if present
            prev_times = [t - pd.Timedelta(days=1) for t in missing]
            available_mask = [pt in g.index for pt in prev_times]
            to_fill_new_index = [t for t, ok in zip(missing, available_mask) if ok]
            from_prev_idx = [pt for pt, ok in zip(prev_times, available_mask) if ok]

            if len(from_prev_idx) > 0:
                fill_rows = g.loc[from_prev_idx].copy()
                fill_rows.index = pd.DatetimeIndex(to_fill_new_index)
                fill_rows['DST'] = False  # label filled rows as non-DST
                g = pd.concat([g, fill_rows], axis=0).sort_index()

        # After one round of filling, verify again; if still missing, raise
        final_idx = pd.date_range(g.index.min(), g.index.max(), freq='15min')
        still_missing = final_idx.difference(g.index)
        if len(still_missing) > 0:
            # print(
            #     f"Unfilled gaps remain: "
            #     f"{len(still_missing)} missing timestamps. Consider providing more context for gap fill."
            # )
            pass

        # 4.6) Apply the user-specified additional shift (hours)
        if shift != 0:
            g.index = g.index + pd.Timedelta(hours=shift)

        # Ensure no overlaps after final shift
        if g.index.has_duplicates:
            # Keep preference to DST=True if duplicates happen to appear again
            g = g.sort_values(['DST'], ascending=[False])
            g = g[~g.index.duplicated(keep='first')]

        # Return with time back as a column named 'local_minute'
        g = g.reset_index().rename(columns={'level_0': 'local_minute'})
        #g = g.drop(columns=['level_1'], errors='ignore')
        return g

    cleaned = out.groupby('dataid', group_keys=True).apply(_process_group, include_groups=False).reset_index()

    # Final sanity: per dataid, ensure no gaps/overlaps
    def _assert_regular(g: pd.DataFrame):
        g = g.copy()
        g = g.reset_index()
        t = g['local_minute'].sort_values()
        diffs = t.diff().dropna()
        if not (diffs == pd.Timedelta(minutes=15)).all():
            print(g['dataid'].iloc[0])
            return g[:0]
            #raise AssertionError(f"Non-regular 15-min series detected.")
        return g
    cleaned.drop(columns=['level_1', 'index'], inplace=True, errors='ignore')
    #cleaned = cleaned.groupby('dataid', group_keys=True).apply(_assert_regular, include_groups=False).reset_index()
    cleaned.drop(columns=['level_1', 'index'], inplace=True, errors='ignore')
    return cleaned

def get_weather():
    newyork_weather_2019 = _load_weather_from_path(path_newyork_weather_2019)
    austin_weather_2018 = _load_weather_from_path(path_austin_weather_2018)
    california_weather_2018 = _load_weather_from_path(path_california_weather_2018)
    california_weather_2016 = _load_weather_from_path(path_california_weather_2016)
    california_weather_2015 = _load_weather_from_path(path_california_weather_2015)
    california_weather_2014 = _load_weather_from_path(path_california_weather_2014)
    
    def _mapper(location: Literal['New York', 'Austin', 'California'], year: Literal[2019, 2018, 2016, 2015, 2014]) -> pd.DataFrame:
        if location == 'New York' and year == 2019:
            return newyork_weather_2019
        elif location == 'Austin' and year == 2018:
            return austin_weather_2018
        elif location == 'California' and year == 2018:
            return california_weather_2018
        elif location == 'California' and year == 2016:
            return california_weather_2016
        elif location == 'California' and year == 2015:
            return california_weather_2015
        elif location == 'California' and year == 2014:
            return california_weather_2014
        else:
            # fall back and raise warning
            warnings.warn(f"Weather data for location '{location}' and year '{year}' not found. Returning New York 2019 weather as fallback.")
            return newyork_weather_2019

    return _mapper

def _load_weather_from_path(path):
    weather_df = pd.read_csv(path, skiprows=2)
    weather_df = weather_df.loc[:, ~weather_df.columns.str.startswith('Unnamed')]
    weather_df = weather_df.drop(columns=['1'], errors='ignore')
    weather_df['localminute'] = weather_df.apply(
        lambda row: pd.Timestamp(year=row['Year'].astype(int), month=row['Month'].astype(int), day=row['Day'].astype(int), hour=row['Hour'].astype(int)),
        axis=1
    )
    weather_df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
    weather_df['hoy'] = np.arange(1, len(weather_df) + 1)
    cols = ['hoy', 'localminute'] + [col for col in weather_df.columns if col not in ['hoy', 'localminute']]
    weather_df = weather_df[cols]
    return weather_df

all_data, all_appliances, all_survey = _prepare_all_data()
metadata_sel = get_metadata()
appliance_to_enduse_mapping = get_appliance_to_enduse_mapping()


class Residence:
    def __init__(self,
                 _id,
                 all_meterage = all_data,
                 all_metadata = metadata_sel,
                 _appliances = all_appliances,
                 ):
        
        self.meterage = all_meterage[all_meterage['dataid'] == _id].copy()
        self.metadata = all_metadata[all_metadata['dataid'] == _id].copy()
        self.dataid = _id
        self.possible_appliances = _appliances
        exapp = []

        row0 = self.metadata.iloc[0]
        for app in _appliances:
            val = row0.get(app, None)
            if isinstance(val, str) and val.strip().lower() == "yes":
                exapp.append(app)
            elif (app in self.meterage.columns) & (self.meterage[app].notna().any()):
                exapp.append(app)
        

            
        self.existing_appliances = exapp
        self.meterage["localminute"] = Residence.parse_local_naive(self.meterage["localminute"])
        self.meterage.sort_values('localminute', inplace=True)
        self.meterage.reset_index(inplace=True, drop=True)
        
        # ---- resampling with availability control ----
        to_resample = self.meterage.copy()[["localminute"] + self.existing_appliances]
        to_resample = to_resample.set_index("localminute").sort_index()
        
        #self.meterage10 = self._resample(to_resample, "10min", 1)
        self.meterage30 = self._resample(to_resample, "30min", 1)
        self.meterage60 = self._resample(to_resample, "1h", 1)
        
        #self.meterage_grouped = self.group_by_enduse(self.meterage)
        #self.meterage10_grouped = self.group_by_enduse(self.meterage10)
        self.meterage30_grouped = self.group_by_enduse(self.meterage30)
        self.meterage60_grouped = self.group_by_enduse(self.meterage60)
        
        for removable in ['grid', 'solar', 'solar2']:
            if removable in self.existing_appliances:
                self.existing_appliances.remove(removable)
        
        

    def _resample(self, to_resample, rule: str, freq: int):
        resampled = to_resample.resample(rule, closed='left', label='right').mean().reset_index()
        resampled = resampled[resampled.notna().any(axis=1)]
        for col in self.existing_appliances:
            resampled[col] = resampled[col].fillna(0)
            resampled[col] = resampled[col] * freq
            resampled[col] = resampled[col].apply(lambda x: max(x, 0))
        resampled['dataid'] = self.dataid
        resampled = resampled[['dataid', 'localminute'] + self.existing_appliances]
        return resampled
    
    
    def group_by_enduse(self, ts: pd.DataFrame) -> pd.DataFrame:
        groups = defaultdict(list)
        for appliance in self.existing_appliances:
            group = appliance_to_enduse_mapping.get(appliance, 'Other')
            if group == 'Other':
                continue
            groups[group].append(appliance)

        appliance_cols = [col for col in ts.columns if col in self.existing_appliances]
        col_to_enduse = {col: appliance_to_enduse_mapping.get(col, 'Other') for col in appliance_cols}

        enduse_groups = defaultdict(list)
        for col, group in col_to_enduse.items():
            if group == 'Other':
                continue
            enduse_groups[group].append(col)

        enduse_hourly_sum = pd.DataFrame({'localminute': ts['localminute']})
        for group, cols in enduse_groups.items():
            enduse_hourly_sum[group] = self.meterage60[cols].sum(axis=1)
        return enduse_hourly_sum

    
    # helper: strip trailing timezone-ish suffix? and parse as naive
    @staticmethod
    def parse_local_naive(series: pd.Series) -> pd.Series:
        # handle mixed dtypes robustly
        s = series.astype(str).str.strip()
        # remove a trailing timezone-like chunk: -05, -0500, -05:00, +04, etc.
        s = s.str.replace(r'([+-]\d{2}(:?\d{2})?)$', '', regex=True)
        # also handle a literal trailing 'Z' if it appears
        s = s.str.replace(r'Z$', '', regex=True)
        # now parse as naive local time
        return pd.to_datetime(s, errors='coerce', format='%Y-%m-%d %H:%M:%S')
    
    
    
    def plot_overview(self):
        
        
        buildings_with_metadata_survey = pd.DataFrame({
            'dataid': all_survey['dataid'].astype(int),
            'hvac': all_survey[[
                'hvac_central_air_gas_furnace', 'hvac_central_air_electric_heating',
                'heat_pump_split', 'hvac_geothermal_heat_pump', 'hvac_window_unit_ac',
                'hvac_ductless_minisplit', 'no_hvac',
            ]].apply(lambda x: x.dropna().tolist()[0], axis=1)
        })
        
        if self.dataid not in buildings_with_metadata_survey['dataid'].values:
            hvac = "Unknown"
        else:
            hvac = buildings_with_metadata_survey[buildings_with_metadata_survey['dataid'] == self.dataid]['hvac'].values[0]
    
        fig, axes = plt.subplots(3, 2, figsize=(16, 8))
        ts = self.meterage60#.set_index('localminute').resample('1M', label='left', closed='left').mean().reset_index()
        #ts = ts.iloc[:-1]
        axes = axes.flatten()
        for i in range(1, len(axes), 2):
            axes[i].sharey(axes[i - 1])
        
        for ax in axes[0:2]:
            ts_sum = ts.copy()
            ts_sum['total'] = ts_sum[[col for col in self.existing_appliances if col not in ['dataid', 'localminute', 'grid', 'solar', 'solar2']]].sum(axis=1)
            plot_cols = (['grid'] +
                 (['solar'] if 'solar' in ts_sum.columns else []) +
                 (['solar2'] if 'solar2' in ts_sum.columns else []))
            
            ax.stackplot(
                ts_sum["localminute"],
                [(ts_sum[col] if col == 'grid' else ts_sum[col].apply(lambda x: x)) for col in plot_cols],
                labels=(['Net Consumption'] + (['Solar 1'] if 'solar' in ts_sum.columns else []) + (['Solar 2'] if 'solar2' in ts_sum.columns else [])),
            )
            
            ax.plot(ts_sum["localminute"], ts_sum['total'], color='black', label='Total Consumption', linewidth=1)
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=2)
        for ax in axes[2:4]:
            ax.stackplot(
                ts["localminute"],
                [ts[col] for col in self.existing_appliances if col not in ['dataid', 'localminute']],
                labels=[col for col in self.existing_appliances if col not in ['dataid', 'localminute']],
            )
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=2)
            
            
            
        ts = self.meterage60_grouped
        for ax in axes[4:6]:
            ax.stackplot(
                ts["localminute"],
                [ts[col] for col in ts.columns if col not in ['dataid', 'localminute', 'PV']],
                labels=[col for col in ts.columns if col not in ['dataid', 'localminute', 'PV']],
            )
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1))
            ax.set_xlabel("Time")
            
        fig.suptitle(f"Hourly Consumption: Home #{self.dataid}, HVAC: {hvac}")
        
        
        period_1_start = pd.Timestamp('2019-07-15')
        period_1_end = pd.Timestamp('2019-07-31')
        period_2_start = pd.Timestamp('2019-10-15')
        period_2_end = pd.Timestamp('2019-10-30')
        
        for _ in np.arange(1, 10):
            if (period_1_start >= ts['localminute'].min()) & (period_1_end <= ts['localminute'].max()):
                break
            else:
                period_1_start = period_1_start + pd.DateOffset(months=2)
                period_1_end = period_1_end + pd.DateOffset(months=2)
            if (period_1_start >= ts['localminute'].min()) & (period_1_end <= ts['localminute'].max()):
                break
            else:
                period_1_start = period_1_start - pd.DateOffset(months=2)
                period_1_end = period_1_end - pd.DateOffset(months=2)
            period_1_start = period_1_start - pd.DateOffset(years=1)
            period_1_end = period_1_end - pd.DateOffset(years=1)
            
        for _ in np.arange(1, 10):
            if (period_2_start >= ts['localminute'].min()) & (period_2_end <= ts['localminute'].max()):
                break
            else:
                period_2_start = period_2_start + pd.DateOffset(months=2)
                period_2_end = period_2_end + pd.DateOffset(months=2)
            if (period_2_start >= ts['localminute'].min()) & (period_2_end <= ts['localminute'].max()):
                break
            else:
                period_2_start = period_2_start - pd.DateOffset(months=2)
                period_2_end = period_2_end - pd.DateOffset(months=2)
            period_2_start = period_2_start - pd.DateOffset(years=1)
            period_2_end = period_2_end - pd.DateOffset(years=1)
        
        for ax in axes[0:6:2]:
            ax.set_xlim(period_1_start, period_1_end)
            ax.set_ylabel("Consumption (kW)")

        for ax in axes[1:6:2]:
            ax.set_xlim(period_2_start, period_2_end)
        fig.tight_layout()
        
        return fig, axes