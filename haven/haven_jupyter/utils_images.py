
from .. import haven_utils
from .. import haven_results as hr
from .. import haven_utils as hu 
from .. import haven_share as hd

import os
import pprint, json
import copy
import pprint
import pandas as pd 

try:
    import ast
    from ipywidgets import Button, HBox, VBox
    from ipywidgets import widgets

    from IPython.display import display
    from IPython.core.display import Javascript, display, HTML
    from IPython.display import FileLink, FileLinks
    from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
except:
    print('widgets not available...')


def images_tab(self, output):
    tfigsize = widgets.Text(
        value=str(self.vars.get('figsize', '(10,5)')),
        description='figsize:',
        disabled=False
            )
    llegend_list = widgets.Text(
        value=str(self.vars.get('legend_list', '[model]')),
        description='legend_list:',
        disabled=False
            )
    
    t_n_images = widgets.Text(
        value=str(self.vars.get('n_images', '5')),
        description='n_images:',
        disabled=False
            )

    t_n_exps = widgets.Text(
        value=str(self.vars.get('n_exps', '3')),
        description='n_exps:',
        disabled=False
            )
    t_dirname = widgets.Text(
        value=str(self.vars.get('dirname', 'images')),
        description='dirname:',
        disabled=False
            )
    bdownload = widgets.Button(description="Download Images", 
                                layout=self.layout_button)
    bdownload_out = widgets.Output(layout=self.layout_button)
    brefresh = widgets.Button(description="Display Images")
    button = widgets.VBox([widgets.HBox([brefresh, bdownload, bdownload_out]),
            widgets.HBox([t_n_exps, t_n_images]),
            widgets.HBox([tfigsize, llegend_list, ]),
            widgets.HBox([t_dirname, ]),
                        ])

    output_plot = widgets.Output()

    with output:
        display(button)
        display(output_plot)

    def on_clicked(b):
        output_plot.clear_output()
        with output_plot:
            self.update_rm()
        
        
            w, h = tfigsize.value.strip('(').strip(')').split(',')
            self.vars['figsize'] = (int(w), int(h))
            self.vars['legend_list'] = hu.get_list_from_str(llegend_list.value)
            self.vars['n_images'] = int(t_n_images.value)
            self.vars['n_exps'] = int(t_n_exps.value)
            self.vars['dirname'] = t_dirname.value
            self.rm_original.fig_image_list =  self.rm.get_images(legend_list=self.vars['legend_list'], 
                    n_images=self.vars['n_images'],
                    n_exps=self.vars['n_exps'],
                    figsize=self.vars['figsize'],
                    dirname=self.vars['dirname'])
            show_inline_matplotlib_plots()
            
    brefresh.on_click(on_clicked)

    
    

    def on_download_clicked(b):
        fname = 'images'
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        pp = PdfPages(fname)
        for fig in self.rm_original.fig_image_list:
            fig.savefig(pp, format='pdf')
        pp.close()

        bdownload_out.clear_output()
        
        with bdownload_out:
            display(FileLink(fname, result_html_prefix="Download: "))

    bdownload.on_click(on_download_clicked)