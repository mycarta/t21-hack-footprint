import panel as pn
import numpy as np
from scipy import signal
import utils
import plotting
import holoviews as hv
import hvplot.xarray  # noqa

import param


class FilterDesigner(param.Parameterized):
    
    filter_ = param.Parameter()
    
    def __init__(self, time_cube, **params):    
        self.ds = utils.time_cube_to_xarray(time_cube)
        self.param.filter_.default = np.zeros((max(time_cube.shape[:2]),max(time_cube.shape[:2])))
        super().__init__(**params)
        self._init_options()

    def _init_options(self):
        self.time_slice_slider = pn.widgets.IntSlider(
            start=0, end=len(self.ds.time_slice) -1,  width=100
        )
        self.time_slice_deep_slider = pn.widgets.IntSlider(
            start=0, end=len(self.ds.time_slice) - 1, 
            value=len(self.ds.time_slice) - 5, width = 100
        )

    def options_pane(self):
        options = pn.Column(
            pn.pane.Markdown("##Filter Slice"),
            self.time_slice_slider,
            pn.pane.Markdown("##Deep Slice"),
            self.time_slice_deep_slider,
        )
        return options

    @pn.depends("time_slice_slider.value")
    def _shallow_slice_graph(self):
        data = self.ds.isel(time_slice=self.time_slice_slider.value)

        # gauss_kernel = utils.scipy_gaussian_2D(11)
        # filter00 = signal.fftconvolve(data.values, gauss_kernel, mode='same')
        # filter_ = utils.normalise(filter00)

        # print(data.values.shape)
        # print(filter_.shape)
        # filtered = utils.apply_filter(data.values, filter_)
        
        base_image = hv.Image(
            data,
            group = 'shallow'
        ).opts(cmap="Greys")

        # filtered_image = hv.Image(
        #     filtered,
        #     group = 'shallow'
        # ).opts(cmap="Greys")

        return base_image# + filtered

    @pn.depends("time_slice_deep_slider.value")
    def _deep_slice_graph(self):
        return hv.Image(
            self.ds.isel(time_slice=self.time_slice_deep_slider.value),
            group = 'deep'
        ).opts(cmap="Greys")

    @pn.depends("time_slice_slider.value")
    def _filter_graph(self):
        spectrum = plotting.view_spectrum(self.ds.isel(time_slice=self.time_slice_slider.value).amplitude.values).opts(plot=dict(shared_axes=False))
        filter_ =  hv.Image(
            self.filter_
        ).opts(alpha=0.1)
        return spectrum * filter_

    def main_graphs(self):
        _filter = hv.DynamicMap(self._filter_graph)

        shallow_slice = hv.DynamicMap(self._shallow_slice_graph)

        deep_slice = hv.DynamicMap(self._deep_slice_graph)

        graphs = pn.Column(
            _filter,
            shallow_slice,
            deep_slice
        )
        return graphs

    def app(self):
        graphs = self.main_graphs()
        options = self.options_pane()
        template = pn.template.MaterialTemplate(
            title="Filter Designer",
            logo="https://raw.githubusercontent.com/CEREGE-CL/CEREGE-CL.github.io/main/logo.png",
            favicon="https://raw.githubusercontent.com/CEREGE-CL/CEREGE-CL.github.io/main/logo.png",
            header_background="#42a5f5",
        )
        # template.sidebar.append(self.file_pane)
        template.sidebar.append(options)
        template.main.append(graphs)
        # template.main.append(self.save_button)
        return template


if "bokeh_app" in __name__:
    penobscot = np.load("images_and_data/penobscot.npy")
    fd = FilterDesigner(penobscot)
    fd.app().servable("Filter Designer")
