from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt

description = \
  """
  ## Arrival Times & Jobs Data Frame

  *jobs := [arrival time, processing time, start, complete, arrived, pending]*

  - **Id** A unique Id for the job, corresponding to its arrival order
  - **Arrival Time**: When the job arrives to the system
  - **Processing Time**: The time the system takes to process the job once started
  - **Start**: The time when the system starts processing the job.
  - **Complete**: The time when the system completes processing of the job. It is always the case that $Complete = ArrivalTime + ProcessingTime$
  - **Arrived**: How many jobs have arrived up until the time of this arrival, including this one.
  - **Pending**: How many jobs are in the system or "pending" at the time of arrival.
  - **WIP**: How many jobs are currently being actively worked on.
  - **DONE**: How many jobs are currently Complete.
  """


Distribution = Callable[[int], np.ndarray[Any, np.dtype[np.float64]]]
CoinFlip = Callable[[int], np.ndarray[Any, np.dtype[np.int_]]]


def uniform(low: float, high: float, time_unit: float = 1) -> Distribution:
  return lambda nSamples: np.random.uniform(low=low, high=high, size=nSamples)*time_unit


def exponential(scale: float, time_unit: float = 1) -> Distribution:
  return lambda nSamples : np.random.exponential(scale, nSamples)*time_unit


def parametrized_gamma(gamma_shape: float, processing_time: float, time_unit: float = 1) -> Distribution:
  return lambda nSamples : np.random.gamma(gamma_shape, processing_time/gamma_shape, nSamples)*time_unit


def coin_toss(ones_p: float = 0.5) -> CoinFlip:
  return lambda nSamples : np.random.binomial(1, ones_p, nSamples)


def initialize_jobs(n_jobs: int,
                    arrival_distribution: Distribution,
                    processing_distribution: Distribution,
                    success_yield: CoinFlip = coin_toss(1.0)) -> pd.DataFrame:
  inter_arrivals = arrival_distribution(n_jobs+1)
  processing_times = processing_distribution(n_jobs+1)
  successes = success_yield(n_jobs+1)
  # -1 to signal not initialized
  jobs = pd.DataFrame(
    [[idx, 0, round(inter_arrivals[idx]), round(processing_times[idx]), -1, -1, successes[idx], idx+1, -1, -1, -1, -1, -1, -1] for idx in range(0, n_jobs+1)],
    columns=['Id', 'Arrival', 'InterArrival', 'ProcessingTime', 'Start', 'Complete', 
             'Success', 'Arrived', 'Pending', 'WIP', 'DONE', 'QueuingFactor', 'WaitTime', 'LeadTime']
  )
  new_arrivals = [jobs['Arrival'][0]]
  inter_arrivals = jobs['InterArrival'].values
  for idx in range(0, jobs.index.size-1):
    new_arrivals.append(new_arrivals[idx] + inter_arrivals[idx])
  jobs['Arrival'] = new_arrivals
  return jobs


def fill_in_dynamic_values(jobs: pd.DataFrame, k_servers: int, warm_up_fraction: float = 0.0) -> tuple[int, int, pd.DataFrame]:
  """
  ## Fill-In Dynamic values:

  - Elapsed
  - Start
  - Complete
  - Arrived
  - Pending
  - WIP
  - DONE
  """
  # With the Arrival time and processing time known
  n_jobs: int = jobs.index.size  # type: ignore
  for idx in range(0, n_jobs):
    # The time of the simulation is set by the current arrival time
    simNow: int = jobs['Arrival'][idx]  # type: ignore
    # Find the jobs that are pending at simulation time that have already arrived and have not yet completed, 
    # ordered by completion time.
    pending = jobs[(jobs['Complete'] > simNow) & (jobs['Arrival'] <= simNow)].sort_values('Complete')  # type: ignore
    number_pending = pending.index.size  # type: ignore
    # Set the value of # of Pending jobs at this time.
    jobs.loc[idx, 'Pending'] = number_pending
    # Of the jobs pending, find the ones that have started at the current time.
    # find all jobs that have an start time before the completion of this job
    # No need to check for non-initialized b/c the baseline are jobs that already have their complete date populated.
    wip_at_sim_now = pending[pending['Start'] <= simNow]  # type: ignore
    # count how many have started
    count_wip_at_sim_now = wip_at_sim_now.index.size  # type: ignore
    # Set it as the "WIP" value
    jobs.loc[idx, 'WIP'] = count_wip_at_sim_now
    # Count how many have completed at simNow (filtering out '-1' values)
    jobs.loc[idx, 'DONE'] = jobs[(jobs['Complete'] < simNow) & (jobs['Complete'] >= 0)].index.size  # type: ignore
    if (idx == 0):
      # The first row is special
      jobs.loc[idx, 'Start'] = jobs['Arrival'][idx] + 1  # type: ignore
    else:
      # Find the first "free processor" as "now" if there are fewer pending than # of servers,
      # or the time of the first job that will leave a processor free.
      first_free_time: 0 = simNow if (number_pending < k_servers) else pending[-k_servers:]['Complete'].values[0]  # type: ignore
      # Set the start time 1 tick after the first completion:
      jobs.loc[idx, 'Start'] = first_free_time + 1  # type: ignore
      # find the time when it will complete
    jobs.loc[idx, 'Complete'] = jobs['Start'][idx] + jobs['ProcessingTime'][idx]
  jobs['WaitTime'] = jobs['Start']-jobs['Arrival']
  jobs['LeadTime'] = jobs['Complete'] - jobs['Arrival']
  jobs['QueuingFactor'] = jobs['WaitTime']/jobs['ProcessingTime']
  first_job_sample = int(warm_up_fraction*n_jobs)+1
  history_starts = jobs['Arrival'][first_job_sample]
  history_ends = jobs['Complete'].max()
  last_job_sample = int(jobs[jobs['Complete'] == history_ends]['Id'].values[0])+1
  return (history_starts, history_ends, jobs[first_job_sample:last_job_sample])


def event_logs(jobs: pd.DataFrame) -> dict[str, pd.DataFrame]:
  n_jobs = jobs.index.size  # type: ignore
  arrivals = jobs[['Id', 'Arrival']].rename(columns={'Arrival': 'Timestamp'})
  arrivals['Event'] = pd.Series((['ARRIVAL'] * n_jobs))

  starts = jobs[['Id', 'Start']].rename(columns={'Start': 'Timestamp'})
  starts['Event'] = pd.Series((['START'] * n_jobs))

  completes = jobs[['Id', 'Complete']].rename(columns={'Complete': 'Timestamp'})
  completes['Event'] = pd.Series((['COMPLETE'] * n_jobs))

  events = pd.concat(  # type: ignore
    [arrivals, starts, completes]).sort_values('Timestamp').reset_index()[['Id', 'Timestamp', 'Event']]

  return {
    'arrivals': arrivals.sort_values('Timestamp').reset_index()[['Id', 'Timestamp', 'Event']],  # type: ignore
    'starts': starts.sort_values('Timestamp').reset_index()[['Id', 'Timestamp', 'Event']],  # type: ignore
    'completes': completes.sort_values('Timestamp').reset_index()[['Id', 'Timestamp', 'Event']],  # type: ignore
    'all_events': events
  }


def plot_times(report_dimension: str,
               reporting_periods: list[dict[str, Any]], 
               left_tail: float = 0.0,
               right_tail: float = 0.0, hist_from_jobs: bool = True) -> None:
  fig, axes = plt.subplots(ncols=2, nrows=2)
  fig.set_figwidth(12)
  fig.set_figheight(10)
  ax0, ax1 = axes

  reports = [
    {
      'axe': [ax0[0], ax1[0]],
      'period': reporting_periods[0]
    },
    {
      'axe': [ax0[1], ax1[1]],
      'period': reporting_periods[1]
    }
  ]

  for r in reports:
    ax = r['axe']
    top_ax = ax[0]
    bottom_ax = ax[1]
    metrics = r['period']['metrics']
    avg = metrics[report_dimension].mean()
    min_required = metrics[report_dimension].quantile(q=left_tail)
    max_allowed = metrics[report_dimension].quantile(q=1.0-right_tail)
    top_ax.xaxis.set_major_formatter(mp.ticker.EngFormatter(places=1))
    top_ax.yaxis.set_major_formatter(mp.ticker.EngFormatter(places=0))
    top_ax.set_xlabel('Timeline')
    top_ax.set_ylabel(report_dimension)
    top_ax.set_title(f"{report_dimension} {r['period']['name']}")
    metrics[report_dimension].plot(ax=top_ax, color='black', linestyle='-', linewidth=0.5)
    metrics[(metrics[report_dimension] < min_required) | (metrics[report_dimension] > max_allowed)][report_dimension].plot(ax=top_ax, color='black', linestyle='', marker='o', markeredgewidth=0.0, markerfacecolor='red', label='Violations')
    top_ax.plot([r['period']['start_time'], r['period']['end_time']], [avg, avg], label="Average", color="green", linestyle='--')
    top_ax.plot([r['period']['start_time'], r['period']['end_time']], [min_required, min_required], label="p{:.0f}".format(left_tail*100), color='orange', linestyle='--')
    top_ax.plot([r['period']['start_time'], r['period']['end_time']], [max_allowed, max_allowed], label="p{:.0f}".format((1-right_tail)*100), color='orange', linestyle='dotted')
    top_ax.legend(ncols=2)
    #
    jobs = r['period']['jobs'] if hist_from_jobs else metrics
    bottom_ax.hist(jobs[report_dimension], bins='doane', rwidth=0.95, color='lightblue')
    bottom_ax.set_xlabel(report_dimension)
    bottom_ax.set_ylabel("Frequency")
    bottom_ax.xaxis.set_major_formatter(mp.ticker.EngFormatter(places=1))

  plt.show()