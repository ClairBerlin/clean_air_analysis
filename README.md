# Clean Air Analysis

Stand-alone Python script to analyse CO2 concentration measurement data to determine the air quality of the location measured.

The present script is the stand-alone development version of data analysis algorithms to be included in the COMo-Berlin _managair_ backend application for managing and analysing CO2-concentration date collected from remotely-read sensors installed in various rooms.

## Algorithms

The algorithms implemented here are developed by [HTW Berlin](https://www.htw-berlin.de/forschung/online-forschungskatalog/projekte/projekt/?eid=3099) and have been simplified and adapted by the COMo development team for simplicity, speed and usage as part of the COMo platform.

## Script

To launch the standalone development script, set up a Python virtual environment:
```python
> . venv/bin/activate
```

Then, run the script on the provided sample data for some month (say, Dec. 2021):
```python
> ./scripts/analyze.py clairchen-rot-stagin-samples2.csv 2021-12
```
