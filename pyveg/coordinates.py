#!/usr/bin/env python

"""
Store all coordaintes in one place.

To chose a location for download, copy the location id into the relevant
line in `config.py`

In future think about:
 - keeping a record when we run a donwload with the date of download
   and a commit hash.
 - having an interface to the `config.py` file which allows the user
   to specify groups of coordinates to download in series/parallel.
   e.g. user specifies that they want to download all "spots"
   locations in Africa.

"""

import pandas as pd

# initialise a DataFrame to store coordinate
coordinate_store = pd.DataFrame(columns=['continent', 'country', 'type', 'latitude', 'longitude'])

# rename index
coordinate_store.index.names = ['id']

# --------------------------------------------------------------------------------
# sudan
coordinate_store.loc['00'] = {'continent': 'Africa', 'country': 'Sudan', 'type': 'spots',      'latitude': 11.58, 'longitude': 27.94}
coordinate_store.loc['01'] = {'continent': 'Africa', 'country': 'Sudan', 'type': 'labyrinths', 'latitude': 11.12, 'longitude': 28.37}
coordinate_store.loc['02'] = {'continent': 'Africa', 'country': 'Sudan', 'type': 'gaps',       'latitude': 10.96, 'longitude': 28.20}

# niger
coordinate_store.loc['03'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'tiger bush', 'latitude': 13.12, 'longitude': 2.59}
coordinate_store.loc['04'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'tiger bush', 'latitude': 13.17, 'longitude': 1.58}

# senegal
coordinate_store.loc['05'] = {'continent': 'Africa', 'country': 'Senegal', 'type': 'labyrinths', 'latitude': 15.20, 'longitude': -15.20}
coordinate_store.loc['06'] = {'continent': 'Africa', 'country': 'Senegal', 'type': 'labyrinths', 'latitude': 15.09, 'longitude': -15.04}
coordinate_store.loc['07'] = {'continent': 'Africa', 'country': 'Senegal', 'type': 'gaps',       'latitude': 15.80, 'longitude': -14.36}
coordinate_store.loc['08'] = {'continent': 'Africa', 'country': 'Senegal', 'type': 'gaps',       'latitude': 15.11, 'longitude': -14.53}

# zambia
coordinate_store.loc['09'] = {'continent': 'Africa', 'country': 'Zambia', 'type': 'gaps', 'latitude': -15.34, 'longitude': 22.22}

# kenya
coordinate_store.loc['10'] = {'continent': 'Africa', 'country': 'Kenya', 'type': 'spots', 'latitude': 0.43, 'longitude': 40.30}

# somalia
coordinate_store.loc['11'] = {'continent': 'Africa', 'country': 'Somalia', 'type': 'labyrinths', 'latitude': 8.09, 'longitude': 47.44}

# australia
coordinate_store.loc['12'] = {'continent': 'Australia', 'country': 'Australia', 'type': 'gaps', 'latitude': -15.71, 'longitude': 133.10}

# usa
coordinate_store.loc['13'] = {'continent': 'America', 'country': 'USA', 'type': 'tiger bush', 'latitude': 26.82, 'longitude': -112.86}
coordinate_store.loc['14'] = {'continent': 'America', 'country': 'USA', 'type': 'labyrinths', 'latitude': 31.05, 'longitude': -103.09}
# --------------------------------------------------------------------------------

# declining locations in West Niger
coordinate_store.loc['15'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'declining', 'latitude': 14.91, 'longitude':  -0.66}
coordinate_store.loc['16'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'declining', 'latitude': 15.03, 'longitude': -0.87}
coordinate_store.loc['17'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'declining', 'latitude': 15.23, 'longitude': -0.97}
coordinate_store.loc['18'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'declining', 'latitude': 15.34, 'longitude': -1.15}
coordinate_store.loc['19'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'declining', 'latitude': 15.14, 'longitude':  -1.16}
coordinate_store.loc['20'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'declining', 'latitude': 14.85, 'longitude': -1.43}
coordinate_store.loc['21'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'declining', 'latitude': 14.97, 'longitude': -1.12}
coordinate_store.loc['22'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'declining', 'latitude': 15.14, 'longitude': -1.56}
coordinate_store.loc['23'] = {'continent': 'Africa', 'country': 'Niger', 'type': 'declining', 'latitude': 15.02, 'longitude': -1.35}

# location with recent drought in Namibia (Issue #283)
coordinate_store.loc['24'] = {'continent': 'Africa', 'country': 'Namibia', 'type': 'declining', 'latitude': 12.76, 'longitude': -18.05}


# hardcode a check to make sure we don't overwrite any rows
assert( len(coordinate_store) == 25 )
