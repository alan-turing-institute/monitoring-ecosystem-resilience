#!/usr/bin/env python

"""
Store all coordaintes in one place.

In future could think about keeping a record when we run a donwload with 
the date of download and a commit hash. 

"""

import pandas as pd

# initialise a DataFrame to store coordinates
coordinates = pd.DataFrame(columns=['continent', 'country', 'type', 'latitude', 'longitude'])

# sudan
coordinates.append({'continent': 'Africa', 'country': 'Sudan', 'type': 'spots',      'latitude': 11.58, 'longitude': 27.94})
coordinates.append({'continent': 'Africa', 'country': 'Sudan', 'type': 'labyrinths', 'latitude': 11.12, 'longitude': 28.37})
coordinates.append({'continent': 'Africa', 'country': 'Sudan', 'type': 'gaps',       'latitude': 10.96, 'longitude': 28.20})

# niger
coordinates.append({'continent': 'Africa', 'country': 'Niger', 'type': 'tiger bush', 'latitude': 13.12, 'longitude': 2.59})
coordinates.append({'continent': 'Africa', 'country': 'Niger', 'type': 'tiger bush', 'latitude': 13.17, 'longitude': 1.58})

# senegal
coordinates.append({'continent': 'Africa', 'country': 'Senegal', 'type': 'labyrinths', 'latitude': 15.20, 'longitude': -15.20})
coordinates.append({'continent': 'Africa', 'country': 'Senegal', 'type': 'labyrinths', 'latitude': 15.09, 'longitude': -15.04})
coordinates.append({'continent': 'Africa', 'country': 'Senegal', 'type': 'gaps',       'latitude': 15.80, 'longitude': -14.36})
coordinates.append({'continent': 'Africa', 'country': 'Senegal', 'type': 'gaps',       'latitude': 15.11, 'longitude': -14.53})

# zambia
coordinates.append({'continent': 'Africa', 'country': 'Zambia', 'type': 'gaps', 'latitude': -15.34, 'longitude': 22.22})

# kenya
coordinates.append({'continent': 'Africa', 'country': 'Kenya', 'type': 'spots', 'latitude': 0.43, 'longitude': 40.30})

# somalia
coordinates.append({'continent': 'Africa', 'country': 'Somalia', 'type': 'labyrinths', 'latitude': 8.09, 'longitude': 47.44})

# australia
coordinates.append({'continent': 'Australia', 'country': 'Australia', 'type': 'gaps', 'latitude': -15.71, 'longitude': 133.10})

# usa
coordinates.append({'continent': 'America', 'country': 'USA', 'type': 'tiger bush', 'latitude': 26.82, 'longitude': -112.86})
coordinates.append({'continent': 'America', 'country': 'USA', 'type': 'labyrinths', 'latitude': 31.05, 'longitude': -103.09})
