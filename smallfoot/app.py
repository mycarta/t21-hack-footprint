import panel as pn
import numpy as np
import io
from scipy import signal
import utils
import plotting
import holoviews as hv
from holoviews.selection import link_selections
import hvplot.xarray  # noqa

import param

from holoviews import opts

opts.defaults(
    opts.Image(
        tools=["hover", "lasso_select", "box_select"],  # Default = hover
    )
)


class FilterDesigner(param.Parameterized):

    filter_ = param.Parameter()
    selection = param.Parameter()

    def __init__(self, time_cube, **params):
        self.ds = utils.time_cube_to_xarray(time_cube)
        self.param.filter_.default = np.zeros(
            (max(time_cube.shape[:2]), max(time_cube.shape[:2]))
        )
        self.param.selection.default = link_selections.instance(unselected_alpha=0.4)
        super().__init__(**params)
        self._init_options()

    def _init_options(self):
        self.time_slice_slider = pn.widgets.IntSlider(
            start=0, end=len(self.ds.time_slice) - 1, width=100
        )
        self.time_slice_deep_slider = pn.widgets.IntSlider(
            start=0,
            end=len(self.ds.time_slice) - 1,
            value=len(self.ds.time_slice) - 5,
            width=100,
        )
        self.save_button = pn.widgets.FileDownload(
            label="\u21A7 Save",
            align="start",
            button_type="success",
            width=100,
            filename="filtered_cube.npy",
        )
        self.save_button.callback = self.save

    def options_pane(self):
        options = pn.Column(
            pn.pane.Markdown("##Filter Slice"),
            self.time_slice_slider,
            pn.pane.Markdown("##Deep Slice"),
            self.time_slice_deep_slider,
            pn.pane.Markdown("##Save"),
            self.save_button,
        )
        return options

    @pn.depends("time_slice_slider.value")
    def _spectrum_graph(self):
        spectrum = plotting.view_spectrum(
            self.ds.isel(time_slice=self.time_slice_slider.value).amplitude.values
        ).opts(plot=dict(shared_axes=False))
        return spectrum

    @pn.depends("selection.selection_expr")
    def _update_filter(self):
        if self.selection.selection_expr is not None:
            hvds = hv.Dataset(
                (
                    np.linspace(-0.5, 0.5, self.filter_.shape[1]),
                    np.linspace(-0.5, 0.5, self.filter_.shape[0]),
                    np.zeros(self.filter_.shape),
                ),
                ["x", "y"],
                "val",
            )
            hvds = hv.Dataset(hvds.dframe())
            hvds.data["val"].loc[
                hvds.select(self.selection.selection_expr).data.index
            ] = 1
            data = hvds["val"].reshape(self.filter_.shape).copy().T[::-1]

            gauss_kernel = utils.scipy_gaussian_2D(11)
            filter00 = signal.fftconvolve(data, gauss_kernel, mode="same")
            filter00 = utils.normalise(filter00)

            self.filter_ = self.filter_ + filter00

        filter_ = hv.Image(self.filter_, group="filter")
        return filter_

    def _make_graphs_slice(self, time_slice, groupname=None):
        data = self.ds.isel(time_slice=time_slice).amplitude.values

        data_sq, slc = utils.pad_next_square_size(data)
        filtered = utils.apply_filter(data_sq, self.filter_)
        filtered = utils.reverse_padding(data, filtered, slc)

        base_image = hv.Image(data, group=groupname).opts(cmap="Greys", title="Default")

        filtered_image = hv.Image(filtered, group=groupname).opts(
            cmap="Greys", title="Filtered"
        )

        differences = hv.Image(data - filtered, group="shallow").opts(
            cmap="Greys", title="Differences"
        )

        return base_image + filtered_image + differences

    @pn.depends("time_slice_slider.value", "filter_")
    def _shallow_slice_graph(self):
        return self._make_graphs_slice(self.time_slice_slider.value, "Shallow")

    @pn.depends("time_slice_deep_slider.value", "filter_")
    def _deep_slice_graph(self):
        return self._make_graphs_slice(self.time_slice_deep_slider.value, "Deep")

    def main_graphs(self):
        _filter = self.selection(hv.DynamicMap(self._spectrum_graph))
        filter_graphs = _filter + hv.DynamicMap(self._update_filter)
        filter_graphs = filter_graphs.opts(plot=dict(shared_axes=False))

        shallow_slice = hv.DynamicMap(self._shallow_slice_graph)

        deep_slice = hv.DynamicMap(self._deep_slice_graph)

        graphs = pn.Column(
            filter_graphs,
            pn.pane.Markdown("##Shallow"),
            shallow_slice,
            pn.pane.Markdown("##Deep"),
            deep_slice,
        )
        return graphs

    def save(self, event=None):
        results = utils.apply_filter_vector_dask(
            self.filter_, np.array(self.ds.amplitude)
        ).compute()
        output = io.BytesIO()
        np.save(output, results)
        return output

    def app(self):
        graphs = self.main_graphs()
        options = self.options_pane()
        template = pn.template.MaterialTemplate(
            title="Filter Designer",
            # logo="https://raw.githubusercontent.com/CEREGE-CL/CEREGE-CL.github.io/main/logo.png",
            # favicon="https://raw.githubusercontent.com/CEREGE-CL/CEREGE-CL.github.io/main/logo.png",
            header_background="#42a5f5",
        )
        template.sidebar.append(options)
        template.main.append(graphs)
        return template


if "bokeh_app" in __name__:
    penobscot = np.load("images_and_data/penobscot.npy")
    fd = FilterDesigner(penobscot)
    fd.app().servable("Filter Designer")
