import datetime

import ipywidgets as ipw
from ipyleaflet import Map, TileLayer, basemaps, WidgetControl


# MApping of esri wayback ids with corresponding dates
DATE_TIMEID_MAPPING = [('2016-01-13', 3515),
                       ('2017-01-11', 577),
                       ('2018-01-08', 13161),
                       ('2019-01-09', 6036),
                       ('2020-01-08', 23001),
                       ('2021-01-13', 1049),
                       ('2022-01-12', 42663),
                       ('2023-01-11', 11475)]


################################
## Google webmap
Google = Map(basemap=basemaps.OpenStreetMap.Mapnik,
             center=(0,0),
             scroll_wheel_zoom=True)
_google_tl = TileLayer(url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}")
Google.add(_google_tl)

#################################
# ESRI with wayback functionality
# Date slider widget
_wayback_slider = ipw.SelectionSlider(
    options=DATE_TIMEID_MAPPING,
    value=3515,
    description='Date',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True
)
_time_control = WidgetControl(widget=_wayback_slider, position='topright')

# Prepare map, tilelayer and draw it
EsriWayback = Map(center=(0, 0),
                  scroll_wheel_zoom=True)
_wayback_tl = TileLayer(url='https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/wmts/1.0.0/default028mm/mapserver/tile/3515/{z}/{y}/{x}',
                        max_zoom=25)
EsriWayback.add_layer(_wayback_tl)
EsriWayback.add_control(_time_control)


def on_date_change(*args):
    _wayback_tl.url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/wmts/1.0.0/default028mm/mapserver/tile/%d/{z}/{y}/{x}' % _wayback_slider.value
    _wayback_tl.redraw()

_wayback_slider.observe(on_date_change, 'value')

#########################################
# Planet monthly basemap
class PlanetBasemap:
    """A class to create and display Planet basemaps with time navigation functionality using ipyleaflet.

    Args:
        frequency (str): The frequency of the basemap updates. Either 'monthly' or 'quarterly'.
        begin (datetime.datetime): The start date for the basemap time range.
        end (datetime.datetime): The end date for the basemap time range.
        api_key (str): The API key for accessing Planet basemaps.

    Attributes:
        frequency (str): The frequency of the basemap updates. Either 'monthly' or 'quarterly'.
        begin (datetime.datetime): The start date for the basemap time range.
        end (datetime.datetime): The end date for the basemap time range.
        api_key (str): The API key for accessing Planet basemaps.
        map (ipyleaflet.Map): The ipyleaflet map instance.

    Example:
        >>> import datetime
        >>> from nrt.validate.webmaps import PlanetBasemap
        >>> planet = PlanetBasemap(frequency='monthly',
        ...                        begin=datetime.datetime(2017, 1, 1),
        ...                        end=datetime.datetime.now(),
        ...                        api_key='your_api_key')
        >>> display(planet.map)
    """
    def __init__(self, frequency='quarterly', begin=datetime.datetime(2017, 1, 1), end=datetime.datetime.now(), api_key=''):
        self.frequency = frequency
        self.begin = begin
        self.end = end
        self.api_key = api_key
        self.map = Map(center=(0, 0), scroll_wheel_zoom=True)
        self._create_time_slider()
        self._add_tile_layer(self.slider.value)

    def _create_time_slider(self):
        date_range = self._generate_date_range()
        self.slider = ipw.SelectionSlider(
            options=[(self._format_label(date), date) for date in date_range],
            value=date_range[0],
            description='Date',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )
        self.slider.observe(self._on_date_change, 'value')

        button_back = ipw.Button(icon='step-backward',
                                 layout=ipw.Layout(width='10%'), style={'button_color': 'transparent'})
        button_next = ipw.Button(icon='step-forward',
                                 layout=ipw.Layout(width='10%'), style={'button_color': 'transparent'})

        button_back.on_click(self._on_prev_button_click)
        button_next.on_click(self._on_next_button_click)

        button_box = ipw.HBox([button_back, self.slider, button_next])

        buttons_control = WidgetControl(widget=button_box, position='topright')
        self.map.add_control(buttons_control)

    def _generate_date_range(self):
        date_range = []
        current_date = self.begin

        while current_date <= self.end:
            date_range.append(current_date)
            if self.frequency == 'monthly':
                # Move to the first day of the next month
                next_month = current_date.month + 1 if current_date.month < 12 else 1
                next_year = current_date.year if current_date.month < 12 else current_date.year + 1
                current_date = current_date.replace(year=next_year, month=next_month, day=1)
            else:  # quarterly
                # Move to the first day of the next quarter
                next_month = current_date.month + 3 if current_date.month <= 9 else 1
                next_year = current_date.year if current_date.month <= 9 else current_date.year + 1
                current_date = current_date.replace(year=next_year, month=next_month, day=1)

        return date_range

    def _format_label(self, date):
        if self.frequency == 'monthly':
            return date.strftime("%Y-%m")
        else:
            quarter = (date.month - 1) // 3 + 1
            return f"{date.year}-Q{quarter}"

    def _format_date(self, date):
        if self.frequency == 'monthly':
            return date.strftime("%Y_%m")
        else:
            quarter = (date.month - 1) // 3 + 1
            return f"{date.year}q{quarter}"

    def _add_tile_layer(self, date):
        formatted_date = self._format_date(date)
        tile_url = self._generate_tile_url(formatted_date)
        self.planet_tl = TileLayer(url=tile_url)
        self.map.add_layer(self.planet_tl)

    def _generate_tile_url(self, formatted_date):
        base_url = "https://tiles.planet.com/basemaps/v1/planet-tiles"
        item_type = "monthly" if self.frequency == "monthly" else "quarterly"
        return f"{base_url}/global_{item_type}_{formatted_date}_mosaic/gmap/{{z}}/{{x}}/{{y}}.png?api_key={self.api_key}"

    def _on_date_change(self, change):
        self.map.remove_layer(self.planet_tl)
        self._add_tile_layer(change['new'])

    def _on_prev_button_click(self, b):
        current_index = self.slider.options.index((self._format_label(self.slider.value), self.slider.value))
        if current_index > 0:
            self.slider.value = self.slider.options[current_index - 1][1]

    def _on_next_button_click(self, b):
        current_index = self.slider.options.index((self._format_label(self.slider.value), self.slider.value))
        if current_index < len(self.slider.options) - 1:
            self.slider.value = self.slider.options[current_index + 1][1]

