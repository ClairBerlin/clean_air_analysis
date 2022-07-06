#!/usr/bin/env python3

# %%
from cmath import isnan
from typing import no_type_check
import click
import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np


def load_samples(sample_file_name):
    """
    Load test samples from a CSV file dumped from the managair database.

    Args:
        sample_file_name (String): The qualified file name of the CSV file to load.

    Returns:
        samples (Pandas data frame): Data frame with DateTime index and co2_ppm values in the single data column. The samples are sorted by date.
    """

    # csv_samples = sample_file.read().rstrip('\n')
    print(f"CSV file: {sample_file_name}")

    samples = pd.read_csv(
        sample_file_name,
        header=None,
        names=[
            "timestamp",
            "co2_ppm",
            "temp_celsius",
            "hum_percent",
            "data_status",
            "node_name",
        ],
    )
    samples["timestamp"] = pd.to_datetime(samples["timestamp"], unit="s")
    samples.set_index("timestamp", inplace=True)
    samples = samples[samples["node_name"] == "c727b2f8-8377-d4cb-0e95-ac03200b8c93"]

    samples.drop("temp_celsius", axis=1, inplace=True)
    samples.drop("hum_percent", axis=1, inplace=True)
    samples.drop("data_status", axis=1, inplace=True)
    samples.drop("node_name", axis=1, inplace=True)
    samples.sort_values("timestamp", inplace=True)

    assert isinstance(
        samples.index[0], pd.Timestamp
    ), "Object must have a datetime-like index."
    return samples


def find_gaps(samples, max_gap_s, timezone):
    """
    Determins gaps in the data frame of nonuniformly spaced samples larger than max_gap

    Args:
        samples (Pandas data frame): date-time index and co2_ppm column
        max_gap_s (Int): Maximum admissible gap between subsequent samples (in seconds)
        timezone (String): Time zone in which to output the gap markers

    Returns:
        zip-list: list of start and stop times identified successive gaps, in the provided time zone.
    """

    # compute time difference between subsequent samples
    sample_timediff = np.diff(samples.index) / np.timedelta64(1, "s")
    # identify intervals with gaps between subsequent samples larger than max_gap
    idx = np.where(
        np.greater(sample_timediff, to_offset(max_gap_s).delta.total_seconds())
    )[0]

    gap_start_indices = (
        samples.index[idx].tz_localize("UTC").tz_convert(timezone).tolist()
    )
    gap_stop_indices = (
        samples.index[idx + 1].tz_localize("UTC").tz_convert(timezone).tolist()
    )

    # Store start and stop indices of large intervals
    gaps = list(zip(gap_start_indices, gap_stop_indices))
    return gaps


def resample_to_uniform_grid(samples, target_rate):
    """
        Resamples the nonuniformly spaced samples to a uniform target_rate

    Args:
        samples (Pandas data frame): date-time index and co2_ppm column
        target_rate (String): Pandas time string.

    Returns:
        Pandas data frame: Resamples values at uniform rate, interpolated where necessary

    See https://towardsdatascience.com/preprocessing-iot-data-linear-resampling-dde750910531
    """
    # First upsample with linear interpolation
    upsampled_samples = samples.resample("1min").mean().interpolate()
    # Then downsample with forward fill.
    return upsampled_samples.resample(target_rate).ffill()


def mark_gaps(samples, gaps):
    """Mark resampled values as None where the original data has large gaps."""
    for start, stop in gaps:
        samples[start:stop] = None
    return samples


def sliceby(samples, freq):
    """Divied the incoming samples data frame into chunks of duration _freq_."""
    grouping = samples.groupby([pd.Grouper(level="timestamp", freq=freq)])
    return dict(list(grouping))


def sliceby_month(samples):
    """Divied the incoming samples data frame into month-sized chunks."""
    return sliceby(samples, freq="MS")


def sliceby_day(samples):
    """Divied the incoming samples data frame into day-sized chunks."""
    return sliceby(samples, freq="D")


def sliceby_hour(samples):
    """Divied the incoming samples data frame into hour-sized chunks."""
    return sliceby(samples, freq="H")


def sliceby_weekday(samples):
    """
    Divide the incoming sample data according to the weekday they belong to.

    The weekdays are indexed from 0 to 6, with 0 = Sunday.
    """
    grouping = samples.groupby(samples.index.weekday)
    return dict(list(grouping))


class Day_Metrics:
    """
    Class to store summary statistics for a given day and to compute derived values.
    """

    max_co2_ppm = None
    mean_co2_ppm = None
    excess_duration_s = None
    mean_excess_co2_ppm = None

    def __init__(self, day, day_duration_s, gap_duration_s):
        self.day = day
        self.day_duration_s = day_duration_s
        self.gap_duration_s = gap_duration_s

    @property
    def has_samples(self):
        return self.gap_duration_s < self.day_duration_s

    @property
    def excess_rate(self):
        if self.has_samples:
            return self.excess_duration_s / (self.day_duration_s - self.gap_duration_s)
        else:
            return None

    @property
    def excess_score(self):
        if self.has_samples:
            return self.mean_excess_co2_ppm * self.excess_rate
        else:
            return None

    def gap_rate(self):
        return 1 - self.gap_duration_s / self.day_duration_s

    def has_data(self):
        return (
            (self.max_co2_ppm is not None)
            and (self.mean_co2_ppm is not None)
            and (self.excess_duration_s is not None)
            and (self.mean_excess_co2_ppm is not None)
            and (self.excess_rate is not None)
            and (self.excess_score is not None)
        )


class Hour_Metrics:
    """
    Class to store summary statistics for a given hour and to compute derived values.
    """

    SECONDS_PER_HOUR = 3600
    MAX_GAP_S = 600  # For hour statistics, only one sample may be amiss.
    is_valid = False
    max_co2_ppm = None
    mean_co2_ppm = None
    excess_duration_s = None
    mean_excess_co2_ppm = None

    def __init__(self, hour, gap_duration_s):
        self.hour = hour
        if gap_duration_s > self.MAX_GAP_S:
            self.gap_duration_s = self.SECONDS_PER_HOUR - 1  # To make computations safe
            self.is_valid = False
        else:
            self.gap_duration_s = gap_duration_s
            self.is_valid = True

    @property
    def excess_rate(self):
        return (
            self.excess_duration_s / (self.SECONDS_PER_HOUR - self.gap_duration_s)
            if self.is_valid
            else None
        )

    @property
    def excess_score(self):
        if self.is_valid:
            return self.mean_excess_co2_ppm * self.excess_rate
        else:
            return None

    def gap_rate(self):
        return 1 - self.gap_duration_s / self.SECONDS_PER_HOUR if self.is_valid else 1

    def has_data(self):
        return (
            self.is_valid
            and (self.max_co2_ppm is not None)
            and (self.mean_co2_ppm is not None)
            and (self.excess_duration_s is not None)
            and (self.mean_excess_co2_ppm is not None)
            and (self.excess_rate is not None)
            and (self.excess_score is not None)
        )


def daily_key_metrics(day, samples, sampling_rate_s, concentration_threshold_ppm):
    """
        Compute key statistics from the samples of a given day

    Args:
        day (Pandas DateTime): The day for which to compute the metrics
        samples (Pandas data frame): datetime index, co2_ppm value column
        sampling_rate_s (Integer): Uniform sampling rate used
        concentration_threshold_ppm (Integer): Threshold for good air quality (e.g., Pettenkofer number)

    Returns:
        Day_Metrics
    """
    day_duration_s = samples.size * sampling_rate_s  # For incomplete days or leap days
    gap_samples = samples[
        samples["co2_ppm"].isna()
    ]  # Exclude samples marked as missing
    gap_duration_s = gap_samples.size * sampling_rate_s

    metrics = Day_Metrics(
        day=day, day_duration_s=day_duration_s, gap_duration_s=gap_duration_s
    )

    if gap_duration_s < day_duration_s:
        metrics.max_co2_ppm = samples["co2_ppm"].max()
        metrics.mean_co2_ppm = samples["co2_ppm"].mean()
        excess_co2_ppm = samples[
            samples["co2_ppm"] >= concentration_threshold_ppm
        ].copy()
        excess_co2_ppm["co2_ppm"] = excess_co2_ppm["co2_ppm"].subtract(
            concentration_threshold_ppm
        )
        metrics.excess_duration_s = excess_co2_ppm.size * sampling_rate_s
        if metrics.excess_duration_s == 0:
            metrics.mean_excess_co2_ppm = 0
        else:
            metrics.mean_excess_co2_ppm = excess_co2_ppm["co2_ppm"].mean()
    return metrics


def prepare_daily_metrics(samples, sampling_rate_s, concentration_threshold_ppm):
    """
    Compute daily metrics for a month's samples; convert metrics into a new data frame.

    TODO: Streamline conversion without the need for the interim Day_Metrics object
    """
    daily_samples = sliceby_day(samples)
    # Use dict comprehension to create a metrics object for each day of the incoming
    # month-samples.
    day_metrics_dict = {
        day: daily_key_metrics(
            day=day,
            samples=samples,
            sampling_rate_s=sampling_rate_s,
            concentration_threshold_ppm=concentration_threshold_ppm,
        )
        for (day, samples) in daily_samples.items()
    }
    # As further processing is simpler when using a data frame, convert the dict of
    # Day_Metrics objects into a new data frame. Construct the data frame from a list.
    daily_metrics_list = [
        {
            "day": m.day,
            "day_duration_s": m.day_duration_s,
            "gap_duration_s": m.gap_duration_s,
            "max_co2_ppm": m.max_co2_ppm,
            "mean_co2_ppm": m.mean_co2_ppm,
            "excess_duration_s": m.excess_duration_s,
            "mean_excess_co2": m.mean_excess_co2_ppm,
            "excess_rate": m.excess_rate,
            "excess_score": m.excess_score,
        }
        for (day, m) in day_metrics_dict.items()
    ]
    month_metrics = pd.DataFrame(daily_metrics_list)
    month_metrics.set_index("day", inplace=True)
    return month_metrics


def hourly_key_metrics(hour, samples, sampling_rate_s, concentration_threshold_ppm):
    """
        Compute key statistics from the samples of a given hour

    Args:
        hour (Pandas DateTime): The hour for which to compute the metrics
        samples (Pandas data frame): datetime index, co2_ppm value column
        sampling_rate_s (Integer): Uniform sampling rate used
        concentration_threshold_ppm (Integer): Threshold for good air quality (e.g., Pettenkofer number)

    Returns:
        Hour_Metrics
    """
    # Require full hours. Disregard incomplete hours at the start or end of a sampling
    # interval.
    if samples.size * sampling_rate_s != 3600:
        return None
    gap_samples = samples[samples["co2_ppm"].isna()]
    gap_duration_s = gap_samples.size * sampling_rate_s

    metrics = Hour_Metrics(hour=hour, gap_duration_s=gap_duration_s)
    # Tolerate a single missing sample only
    if gap_duration_s <= 600 and gap_samples.size <= 1:
        metrics.max_co2_ppm = samples["co2_ppm"].max()
        metrics.mean_co2_ppm = samples["co2_ppm"].mean()
        excess_co2_ppm = samples[
            samples["co2_ppm"] >= concentration_threshold_ppm
        ].copy()
        excess_co2_ppm["co2_ppm"] = excess_co2_ppm["co2_ppm"].subtract(
            concentration_threshold_ppm
        )
        metrics.excess_duration_s = excess_co2_ppm.size * sampling_rate_s
        if metrics.excess_duration_s == 0:
            metrics.mean_excess_co2_ppm = 0
        else:
            metrics.mean_excess_co2_ppm = excess_co2_ppm["co2_ppm"].mean()
    return metrics


def prepare_hourly_metrics(samples, sampling_rate_s, concentration_threshold_ppm):
    """
    Compute hourly metrics for a month's samples; convert metrics into a new data frame.

    TODO: Streamline conversion without the need for the interim Hour_Metrics object
    """
    hourly_samples = sliceby_hour(samples)
    # Use dict comprehension to create a metrics object for each hour of the incoming
    # month-samples.
    hourly_metrics_dict = {
        hour: hourly_key_metrics(
            hour=hour,
            samples=samples,
            sampling_rate_s=sampling_rate_s,
            concentration_threshold_ppm=concentration_threshold_ppm,
        )
        for (hour, samples) in hourly_samples.items()
    }
    # As further processing is simpler when using a data frame, convert the dict of
    # Hour_Metrics objects into a new data frame. Construct the data frame from a list.
    hourly_metrics_list = [
        {
            "hour": m.hour,
            "gap_duration_s": m.gap_duration_s,
            "max_co2_ppm": m.max_co2_ppm,
            "mean_co2_ppm": m.mean_co2_ppm,
            "excess_duration_s": m.excess_duration_s,
            "mean_excess_co2": m.mean_excess_co2_ppm,
            "excess_rate": m.excess_rate,
            "excess_score": m.excess_score,
        }
        for (hour, m) in hourly_metrics_dict.items()
    ]
    hourly_metrics = pd.DataFrame(hourly_metrics_list)
    hourly_metrics.set_index("hour", inplace=True)
    return hourly_metrics


def weekday_histogram(hourly_metrics):
    """
        Compute a histogram across weekdays of the provided input

    Args:
        hourly_metrics (Pandas data frame): date-time index with hour resolution, metrics columns

    Returns:
        Weekday-Dict: hourly sumary statistics. Weekdays are indexed starting at 0, with 0 = Sunday
    """
    purged_hourly_metrics = hourly_metrics[
        hourly_metrics["mean_co2_ppm"].isna() == False
    ]
    # TODO: Check for hours with too much missing data to exclude from histogram?
    # purged_hourly_count = purged_hourly_metrics.groupby(
    #     purged_hourly_metrics.index.hour
    # ).count()
    # na_hourly_metrics = hourly_metrics[hourly_metrics["mean_co2_ppm"].isna() == True]
    # na_hourly_count = na_hourly_metrics.groupby(na_hourly_metrics.index.hour).count()

    weekday_metrics = sliceby_weekday(purged_hourly_metrics)
    return {
        weekday: m.groupby(m.index.hour).mean()
        for (weekday, m) in weekday_metrics.items()
    }


def clean_air_medal(daily_metrics):
    """
    Determines from the daily summary statistics if the clean-air-medal should be awarded for the given month.

    Args:
        daily_metrics (Pandas data frame): Data frame with daily metrics for a given month, for which

    Returns:
        Boolean: If the clean-air-medal is awarded for the given month or not
    """

    BAD_AIR_THRESHOLD_PPM = 2000
    EXCESS_SCORE_THRESHOLD = 150

    # Wenn alle Maximalwerte unter 2000 ppm waren, nie eine rote Ampel auftrat (d.h. der Wert lag
    # auch nicht an einem Tag über 30% der Zeit über dem Referenzwert) und weniger als 30% Gelbe-Ampel-Tagesbewertungen, wird die Frischluft-Medaille vergeben

    if (
        daily_metrics[
            daily_metrics["max_co2_ppm"] >= BAD_AIR_THRESHOLD_PPM
        ].max_co2_ppm.count()
        > 0
    ):
        # If the CO2 concentration exceeds the BAD_AIR_THRESHOLD even once during the
        # entire month, the clean air medal cannot be awarded.
        return False
    elif (
        daily_metrics[
            daily_metrics["excess_score"] >= EXCESS_SCORE_THRESHOLD
        ].excess_score.count()
        > 0
    ):
        # If CO2-concentration exceeds the clean-air threshold on average by more than
        # EXCESS_SCORE_THRESHOLD, the clean air medal cannot be awarded.
        return False
    elif daily_metrics[
        daily_metrics["excess_score"] >= 0
    ].excess_score.count() >= 0.3 * len(daily_metrics):
        # If the CO2-concentration exceeds the clean air threshold on more than 30% of the days of the given month, the clean air medal cannot be awarded.
        return False
    else:
        return True


@click.command()
@click.argument("sample_file_name")
@click.argument("month")
def analyze_samples(sample_file_name, month):
    """Analyze CO2 sensor time series imported from a CSV file.

    Args:
        sample_file_name (_String_): The CSV file that contains the sample time series.
        month (YYYY-MM): The month to analyze.
    """

    max_gap = "30min"  # maximum gap between successive samples to tolerate
    target_rate = "10min"  # uniform sampling frequency to transform the data to
    timezone = "Europe/Berlin"

    # For testing, load samples from a file. In the online implementation, query samples
    # from the database for a given installation and over a specified time frame.
    samples = load_samples(sample_file_name)

    gaps = find_gaps(samples, max_gap, timezone)

    samples.index = samples.index.tz_localize("UTC").tz_convert(timezone)

    # Most samples are nonuniformly spaced because of transmission delays and clock
    # skew. To simplify processing, resample these samples on a uniform grid at the
    # target_rate. This might lead to some noise amplification for stretches of sparse
    # original samples. If the gaps between subsequent samples are too large,
    # resampling will yield mostly noise; therefore, we exclude these stretches and
    # insert NaN-values instead.
    uniform_samples = resample_to_uniform_grid(samples, target_rate)
    uniform_marked_samples = mark_gaps(uniform_samples, gaps)

    # Take a single month for all day-based analyses, as this is what we would retrieve
    # from the DB
    monthly_samples = sliceby_month(uniform_marked_samples)
    month = pd.Timestamp(month).tz_localize("Europe/Berlin")
    month_samples = monthly_samples[month]

    # Pandas data frame of daily metrics for the given month; i.e. summary statistics
    # for each day of the selected month. With 10min sampling rate, the statistics are
    # computed over 24*6 = 144 samples a day, except for leap days.
    daily_metrics = prepare_daily_metrics(
        samples=month_samples,
        sampling_rate_s=600,
        concentration_threshold_ppm=1000,
    )
    print(daily_metrics.head())

    # Pandas data frame of hourly metrics for the given month; i.e., summary statistics
    # for each hour of the selected month. With 10 min sampling rate, the statistics
    # are computed over 6 samples per hour (which is not that much).
    hourly_metrics = prepare_hourly_metrics(
        samples=month_samples,
        sampling_rate_s=600,
        concentration_threshold_ppm=1000,
    )
    print(hourly_metrics.head())

    # Compute mean values for each day of the week (Mo, Tu, etc.) for each hour of the
    # day. The resulting histograms can be used to indicate which hours are the most
    # critical for the selected room. The indexing starts at Sunday (index 0).
    weekday_hist = weekday_histogram(hourly_metrics)
    print(weekday_hist)

    award_clean_air_medal = clean_air_medal(daily_metrics)
    print("Clean-air-medal awarded: {}".format(award_clean_air_medal))


if __name__ == "__main__":
    analyze_samples()
