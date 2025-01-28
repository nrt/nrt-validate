import datetime
import xml.etree.ElementTree as ET
import re

import ipywidgets as ipw
from ipyleaflet import Map, TileLayer, WidgetControl, basemaps
import requests


class GoogleBasemap:
    """
    A class to create and display a Google basemap using ipyleaflet.

    Attributes:
        map (ipyleaflet.Map): The ipyleaflet map instance.

    Example:
        >>> from nrt.validate.webmaps import GoogleBasemap
        >>> google_map = GoogleBasemap()
        >>> display(google_map.map)
    """
    def __init__(self):
        self.map = Map(basemap=basemaps.OpenStreetMap.Mapnik,
                       center=(0, 0),
                       scroll_wheel_zoom=True)
        self._add_google_layer()

    def _add_google_layer(self):
        google_tl = TileLayer(url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}")
        self.map.add(google_tl)


class EsriWaybackBasemap:
    """
    A class to create and display an Esri Wayback basemap with time navigation functionality using ipyleaflet.

    Attributes:
        map (ipyleaflet.Map): The ipyleaflet map instance.
        DATE_TIMEID_MAPPING (list): A list of tuples containing date and ID mappings.

    Example:
        >>> from nrt.validate.webmaps import EsriWaybackBasemap
        >>> esri_wayback_map = EsriWaybackBasemap()
        >>> display(esri_wayback_map.map)
    """
    def __init__(self):
        self.map = Map(center=(0, 0), scroll_wheel_zoom=True)
        self.DATE_TIMEID_MAPPING = self._fetch_date_timeid_mapping()
        self._create_time_slider()
        self._add_tile_layer(self.DATE_TIMEID_MAPPING[0][2])

    def _fetch_date_timeid_mapping(self):
        url = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/mapserver/wmts/1.0.0/wmtscapabilities.xml"
        response = requests.get(url)
        tree = ET.ElementTree(ET.fromstring(response.content))
        root = tree.getroot()

        ns = {
            'opengis': 'https://www.opengis.net/wmts/1.0',
            'ows': 'https://www.opengis.net/ows/1.1'
        }

        date_timeid_mapping = []
        content = root.find('opengis:Contents', ns)
        for layer in content.findall('opengis:Layer', ns):
            name = layer.find('ows:Title', ns).text
            date = re.findall(r'(\d{4}-\d{2}-\d{2})', name)[0]
            idx = layer.find('ows:Identifier', ns).text
            template = layer.find('opengis:ResourceURL', ns).attrib['template']
            time_id = re.findall(r'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/{TileMatrixSet}/MapServer/tile/(\d+)/{TileMatrix}/{TileRow}/{TileCol}', template)[0]
            date_timeid_mapping.append((date, idx, int(time_id)))

        date_timeid_mapping = sorted(date_timeid_mapping)

        # Filter to get R01 and R08 basemap versions per year
        filtered_mapping = []
        years = set(datetime.datetime.strptime(date, "%Y-%m-%d").year for date, _, _ in date_timeid_mapping)
        for year in sorted(years):
            year_entries = [entry for entry in date_timeid_mapping if int(entry[0][:4]) == year]
            r01_entries = [entry for entry in year_entries if 'R01' in entry[1]]
            r08_entries = [entry for entry in year_entries if 'R08' in entry[1]]
            if r01_entries:
                filtered_mapping.append(r01_entries[0])
            if r08_entries:
                filtered_mapping.append(r08_entries[0])

        return filtered_mapping

    def _create_time_slider(self):
        self.slider = ipw.SelectionSlider(
            options=[(date, time_id) for date, _, time_id in self.DATE_TIMEID_MAPPING],
            value=self.DATE_TIMEID_MAPPING[0][2],
            description='Date',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True
        )
        self.slider.observe(self._on_date_change, 'value')
        time_control = WidgetControl(widget=self.slider, position='topright')
        self.map.add_control(time_control)

    def _add_tile_layer(self, date_value):
        tile_url = f'https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/wmts/1.0.0/default028mm/mapserver/tile/{date_value}/{{z}}/{{y}}/{{x}}'
        self.esri_tl = TileLayer(url=tile_url, max_zoom=25)
        self.map.add_layer(self.esri_tl)

    def _on_date_change(self, change):
        self.esri_tl.url = f'https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/wmts/1.0.0/default028mm/mapserver/tile/{change["new"]}/{{z}}/{{y}}/{{x}}'
        self.esri_tl.redraw()


class PlanetBasemap:
    """
    A class to create and display Planet basemaps with time navigation functionality using ipyleaflet.

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
        >>> planet = PlanetBasemap(frequency='monthly', begin=datetime.datetime(2017, 1, 1), end=datetime.datetime.now(), api_key='your_api_key')
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

        button_back = ipw.Button(icon='step-backward', layout=ipw.Layout(width='30px', height='30px'), style={'button_color': 'transparent'})
        button_next = ipw.Button(icon='step-forward', layout=ipw.Layout(width='30px', height='30px'), style={'button_color': 'transparent'})
        button_back.on_click(self._on_prev_button_click)
        button_next.on_click(self._on_next_button_click)
        button_box = ipw.HBox([button_back, self.slider, button_next], layout=ipw.Layout(align_items='center'))
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
        formatted_date = self._format_date(change['new'])
        self.planet_tl.url = self._generate_tile_url(formatted_date)
        self.planet_tl.redraw()

    def _on_prev_button_click(self, b):
        current_index = self.slider.options.index((self._format_label(self.slider.value), self.slider.value))
        if current_index > 0:
            self.slider.value = self.slider.options[current_index - 1][1]

    def _on_next_button_click(self, b):
        current_index = self.slider.options.index((self._format_label(self.slider.value), self.slider.value))
        if current_index < len(self.slider.options) - 1:
            self.slider.value = self.slider.options[current_index + 1][1]
