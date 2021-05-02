import panel as pn
import numpy as np
import io
import os
from scipy import signal
import utils
import plotting
import holoviews as hv
from holoviews.selection import link_selections
import hvplot.xarray  # noqa
from scipy.ndimage import gaussian_filter

import param

from holoviews import opts

opts.defaults(
    opts.Image(
        tools=["hover", "box_select"],  # Default = hover
    )
)


class FilterDesigner(param.Parameterized):

    filter_ = param.Parameter()
    selection = param.Parameter()

    def __init__(self, listfile, **params):
        time_cube=np.load(listfile[0])[:-1,:,:]
        self.ds = utils.time_cube_to_xarray(time_cube)
        self.param.filter_.default = np.zeros(
            (max(time_cube.shape[:2]), max(time_cube.shape[:2]))
        )
        self.param.selection.default = link_selections.instance(unselected_alpha=0.4)
        super().__init__(**params)
        listname = [w.split('/')[-1] for w in listfile]
        self._init_options(listname)

    def _init_options(self, listname):
        #listname = [w.split('/')[-1] for w in listfile]
        self.select_file = pn.widgets.Select(name='List available:', options=listname, size=1,width=230)
        self.time_slice_slider = pn.widgets.IntSlider(
            start=0, end=len(self.ds.time_slice) - 1, width=200
        )
        self.time_slice_deep_slider = pn.widgets.IntSlider(
            start=0,
            end=len(self.ds.time_slice) - 1,
            value=len(self.ds.time_slice) - 5,
            width=200,
        )
        self.reset_button=pn.widgets.Button(name='Reset Pick', button_type='warning', width=100)
        self.load_button=pn.widgets.Button(name='Load', button_type='danger', width=100)
        self.save_button = pn.widgets.FileDownload(
            label="\u21A7 Save",
            align="start",
            button_type="success",
            width=100,
            filename=self.select_file.value.replace('.npy','_')+"filtered_cube.npy",
        )
        self.save_button.callback = self.save
        self.reset_button.on_click(self.reset)
        self.load_button.on_click(self._update_self)
        #self.reset_button.callback = self.reset
    def options_pane(self):
        options = pn.Column(
            pn.pane.Markdown("##Select File"),
            self.select_file,
            self.load_button,
            pn.pane.Markdown(""),
            pn.pane.Markdown("##Filter Slice"),
            self.time_slice_slider,
            pn.pane.Markdown("##Deep Slice"),
            self.time_slice_deep_slider,
            pn.pane.Markdown(""),
            self.reset_button,
            pn.pane.Markdown("##Save"),
            self.save_button,
        )
        return options



#     @pn.depends("time_slice_slider.value")
#     def _spectrum_graph(self):
#         spectrum = plotting.view_spectrum(
#             self.ds.isel(time_slice=self.time_slice_slider.value).amplitude.values
#         ).opts(plot=dict(shared_axes=False))
#         return spectrum
    
    
    #@pn.depends("select_file.value")

    
    @pn.depends("time_slice_slider.value")
    def _spectrum_graph(self):
        print('spectrum:',self.select_file.value,self.ds.isel(time_slice=self.time_slice_slider.value).amplitude.values.shape,self.filter_.shape)
        return hv.Image(np.log(self.ds.isel(time_slice=self.time_slice_slider.value).spec_amp.values +
                               1)).opts(cmap="cubehelix", title="FFT Spectrum") #, plot=dict(shared_axes=False))


    @pn.depends("selection.selection_expr")
    def _update_filter(self):
        print('update_filter:',self.select_file.value,self.ds.isel(time_slice=self.time_slice_deep_slider.value).amplitude.values.shape,self.filter_.shape)
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

            gauss_kernel = utils.scipy_gaussian_2D(int(self.filter_.shape[1]/40))
            filter00 = signal.fftconvolve(data, gauss_kernel, mode="same")
            filter00 = utils.normalise(filter00)

            self.filter_ = self.filter_ + filter00

        filter_ = hv.Image(self.filter_, group="filter")
        return filter_

    def _make_graphs_slice(self, time_slice, groupname=None):
        print('make_graphs_slice:',self.select_file.value,self.ds.isel(time_slice=self.time_slice_deep_slider.value).amplitude.values.shape,self.filter_.shape)
        data = self.ds.isel(time_slice=time_slice).amplitude.values

        data_sq, slc = utils.pad_next_square_size(data)
        filtered = utils.apply_filter_pytorch(data_sq, self.filter_)
        filtered = utils.reverse_padding(data, filtered, slc)

        base_image = hv.Image(data, group=groupname).opts(cmap="Greys", title="Default")

        filtered_image = hv.Image(filtered, group=groupname).opts(
            cmap="Greys", title="Filtered"
        )

        differences = hv.Image(data - filtered, group="shallow").opts(
            cmap="Greys", title="Differences", clim=(data.min()/2.0,data.max()/2.0)
        )

        return base_image + filtered_image + differences

    @pn.depends("time_slice_slider.value", "filter_")
    def _shallow_slice_graph(self):
        print('shallow_slice_graph:',self.select_file.value,self.ds.isel(time_slice=self.time_slice_slider.value).amplitude.values.shape,self.filter_.shape)
        return self._make_graphs_slice(self.time_slice_slider.value, "Shallow")

    @pn.depends("time_slice_deep_slider.value", "filter_")
    def _deep_slice_graph(self):
        print('deep_slice_graph:',self.select_file.value,self.ds.isel(time_slice=self.time_slice_deep_slider.value).amplitude.values.shape,self.filter_.shape)
        return self._make_graphs_slice(self.time_slice_deep_slider.value, "Deep")

    def main_graphs(self):
        self=self._update_self()
        print(self.ds.isel(time_slice=self.time_slice_deep_slider.value).amplitude.values.shape)
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
        print('save:',self.select_file.value,self.ds.isel(time_slice=self.time_slice_deep_slider.value).amplitude.values.shape,self.filter_.shape)
        #results = utils.apply_filter_vector_dask(
        print(self.filter_.shape,np.array(self.ds.amplitude).shape)
        results = utils.apply_filter_vector_pytorch(
            self.filter_, np.array(self.ds.amplitude)
        ).astype('float32')#.compute()
        
        output = io.BytesIO()
        np.save(output, results)
        print(results.max(), results.shape)
        #print(output.getvalue())
        output.seek(0)
        return output

    def reset(self, event=None):
        print('reset:',self.select_file.value,self.ds.isel(time_slice=self.time_slice_deep_slider.value).amplitude.values.shape,self.filter_.shape)
        #self.selection.selection_expr = None
        #self.filter_ = self.filter_*0.0
        time_cube=np.load('.//images_and_data//'+self.select_file.value)[:-1,:,:]
        print('new shape:',time_cube.shape)
        self.ds = utils.time_cube_to_xarray(time_cube)
        
        self.filter_ = np.zeros(
            (max(time_cube.shape[:2]), max(time_cube.shape[:2]))
        )
        self.selection.selection_expr = None
        #_filter = self.selection(hv.DynamicMap(self._spectrum_graph))
        #filter_graphs = _filter + hv.DynamicMap(self._update_filter)
        #filter_graphs = filter_graphs.opts(plot=dict(shared_axes=False))
  
        return self

    def _update_self(self, event=None):
        #print('old shape',self.ds.shape)
        print('change file:',self.select_file.value,self.ds.isel(time_slice=self.time_slice_deep_slider.value).amplitude.values.shape,self.filter_.shape)
        time_cube=np.load('.//images_and_data//'+self.select_file.value)[:-1,:,:]
        #time_cube=np.load('.//images_and_data//'+'penobscot.npy')[:-1,:,:]
        print('new shape:',time_cube.shape)
        self.ds = utils.time_cube_to_xarray(time_cube)
        self.time_slice_slider.end=len(self.ds.time_slice) - 1
        self.time_slice_slider.value=0
        self.time_slice_deep_slider.value =len(self.ds.time_slice) - 5 
        self.time_slice_deep_slider.end = len(self.ds.time_slice) - 1
        self.filter_ = np.zeros(
            (max(time_cube.shape[:2]), max(time_cube.shape[:2]))
        ) 
        self.selection.selection_expr = None

        return self

    def app(self):
        graphs = self.main_graphs()
        options = self.options_pane()
        template = pn.template.MaterialTemplate(
            title="F.R.I.D.A. Seismic Footprint Filter Designer App",
            logo="https://images.squarespace-cdn.com/content/58a4b31dbebafb6777c575b4/1549829488328-IZMTRHP7SLI9P9Z7MUSW/website_logo_head.png?content-type=image%2Fpng",
            # logo="https://raw.githubusercontent.com/CEREGE-CL/CEREGE-CL.github.io/main/logo.png",
            # favicon="https://raw.githubusercontent.com/CEREGE-CL/CEREGE-CL.github.io/main/logo.png",
            header_background="#42a5f5",
        )
        template.sidebar.append(options)
        template.main.append(graphs)
        return template


if "bokeh_app" in __name__:
#    seismic = np.load("images_and_data/penobscot.npy")
#     seismic = np.load("images_and_data/F3_original_subvolume_IL230-430_XL475-675_T200-1800.npy")
    listfile = []
    listname = []
    for root, dirs, files in os.walk(r'./images_and_data/'):
        for file in files:
            file2=file.lower()
            if file2.endswith(".sgy") or file2.endswith(".npy"):
               listfile.append(os.path.join(root, file))
               listname.append(file)
    fd = FilterDesigner(listfile)
    fd.app().servable("Filter Designer")
