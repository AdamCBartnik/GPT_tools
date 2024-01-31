import ipywidgets as widgets
import matplotlib.pyplot as plt
import glob, os, yaml, copy
import pandas as pd
import numpy as np
from GPT_tools.SnappingCursor import SnappingCursor
import time


class front_gui:
    def __init__(self, xopt_file, pop_directory):
        
        self.default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.color_dict = {}
        self.default_legend_dict = {}
        self.legend_dict = {}
        
        self.xopt_file = xopt_file
        self.pop_directory = pop_directory
        self.colorbar_instance = None
        
        self.mouse_event_handler_1 = None
        self.mouse_event_handler_2 = None

        with open(self.xopt_file, 'r') as fid:
            self.xopt_file =  yaml.safe_load(fid)     

        obj = list(self.xopt_file['vocs']['objectives'].keys())
        vars = list(self.xopt_file['vocs']['variables'].keys())
        cons = list(self.xopt_file['vocs']['constraints'].keys())
        
        self.params_from_xopt = obj + vars + cons

        file_list = glob.glob(os.path.join(self.pop_directory, '*.csv'))
        file_list.sort(key=lambda x: os.path.getmtime(x))
        file_list.reverse()
        
        file_list = [os.path.basename(ff) for ff in file_list]
        for ii,ff in enumerate(file_list):
            self.default_legend_dict[ff] = f'file {ii+1}'
            self.legend_dict[ff] = ''
                    
        # Outermost container object
        self.gui = widgets.VBox(layout={'border': '1px solid grey'})

        # Layouts
        layout_150px = widgets.Layout(width='150px',height='30px')
        layout_20px = widgets.Layout(width='20px',height='30px')
        label_layout = layout_150px

        # Make plotting region
        dpi = 120
        plot_width = 600
        plot_height = 500

        plt.ioff() # turn off interactive mode so figure doesn't show
        self.fig, self.ax = plt.subplots(layout='constrained', dpi=dpi, figsize=[plot_width/dpi, plot_height/dpi]) # layout = constrained, tight
        plt.ion() 

        self.fig.canvas.toolbar_position = 'right'
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.resizable = False

        fig_hbox = widgets.HBox([self.fig.canvas], layout=widgets.Layout(width='650px'))

        # Make input region
        input_hbox = widgets.HBox()
        file_vbox = widgets.VBox()
        
        self.file_select = widgets.SelectMultiple(options=file_list, value=[file_list[0]], disabled=False, layout=widgets.Layout(width='500px', height='300px'))
        self.active_file = widgets.Dropdown(options=[file_list[0]], disabled=False, layout=widgets.Layout(width='250px', height='30px'))
        self.active_color = widgets.ColorPicker(concise=True, description='', value=self.default_color_list[0], disabled=False, layout=widgets.Layout(width='50px', height='30px'))

        self.legend_checkbox = widgets.Checkbox(value=False,description='Legend: ',disabled=False,indent=False, layout=widgets.Layout(width='70px', height='30px'))
        self.legend_str = widgets.Text(value='',placeholder='Legend name',description='',disabled=False, layout=widgets.Layout(width='110px', height='30px'))
        
        active_file_hbox = widgets.HBox()
        active_file_hbox.children += (self.active_file, )
        active_file_hbox.children += (self.active_color, )
        active_file_hbox.children += (self.legend_checkbox, ) 
        active_file_hbox.children += (self.legend_str, ) 
                                            
        file_vbox.children += (widgets.Label('Select files (hold Ctrl for multiple)'), )
        file_vbox.children += (self.file_select, )
        file_vbox.children += (active_file_hbox, )
        
        self.x_select = widgets.Dropdown(options=['Temp'],value='Temp',description='x :',disabled=False,layout=widgets.Layout(width='250px',height='30px'))
        self.y_select = widgets.Dropdown(options=['Temp'],value='Temp',description='y :',disabled=False,layout=widgets.Layout(width='250px',height='30px'))
        self.c_select = widgets.Dropdown(options=['Temp'],value='Temp',description='color :',disabled=False,layout=widgets.Layout(width='250px',height='30px'))

        scale_list = [str(xx) for xx in np.arange(-15,16,3)]
        self.x_scale = widgets.Dropdown(options=scale_list,value='0',description='scale : 10^', disabled=False,layout=widgets.Layout(width='140px',height='30px'))
        self.x_units = widgets.Text(value='',placeholder='Enter units',disabled=False,layout=widgets.Layout(width='110px',height='30px'))
        self.x_label = widgets.Text(value='',placeholder='Enter label',description='Label : ', disabled=False,layout=widgets.Layout(width='250px',height='30px'))
        self.x_min = widgets.Text(value='',placeholder='x min',description='Limits : ', disabled=False,layout=widgets.Layout(width='150px',height='30px'))
        self.x_max = widgets.Text(value='',placeholder='x max',disabled=False,layout=widgets.Layout(width='70px',height='30px'))
        
        self.y_scale = widgets.Dropdown(options=scale_list,value='0',description='scale : 10^', disabled=False,layout=widgets.Layout(width='140px',height='30px'))
        self.y_units = widgets.Text(value='',placeholder='Enter units',disabled=False,layout=widgets.Layout(width='110px',height='30px'))
        self.y_label = widgets.Text(value='',placeholder='Enter label',description='Label : ', disabled=False,layout=widgets.Layout(width='250px',height='30px'))
        self.y_min = widgets.Text(value='',placeholder='y min',description='Limits : ', disabled=False,layout=widgets.Layout(width='150px',height='30px'))
        self.y_max = widgets.Text(value='',placeholder='y max',disabled=False,layout=widgets.Layout(width='70px',height='30px'))
        
        self.c_scale = widgets.Dropdown(options=scale_list,value='0',description='scale : 10^', disabled=False,layout=widgets.Layout(width='140px',height='30px'))
        self.c_units = widgets.Text(value='',placeholder='Enter units',disabled=False,layout=widgets.Layout(width='110px',height='30px'))
        self.c_label = widgets.Text(value='',placeholder='Enter label',description='Label : ', disabled=False,layout=widgets.Layout(width='250px',height='30px'))
        self.c_min = widgets.Text(value='',placeholder='c min',description='Limits : ', disabled=False,layout=widgets.Layout(width='150px',height='30px'))
        self.c_max = widgets.Text(value='',placeholder='c max',disabled=False,layout=widgets.Layout(width='70px',height='30px'))
        
        x_scale_hbox = widgets.HBox()
        x_scale_hbox.children += (self.x_scale, )
        x_scale_hbox.children += (self.x_units, )
        
        x_limits_hbox = widgets.HBox()
        x_limits_hbox.children += (self.x_min, )
        x_limits_hbox.children += (self.x_max, )
        
        y_scale_hbox = widgets.HBox()
        y_scale_hbox.children += (self.y_scale, )
        y_scale_hbox.children += (self.y_units, )
        
        y_limits_hbox = widgets.HBox()
        y_limits_hbox.children += (self.y_min, )
        y_limits_hbox.children += (self.y_max, )
        
        c_scale_hbox = widgets.HBox()
        c_scale_hbox.children += (self.c_scale, )
        c_scale_hbox.children += (self.c_units, )
        
        c_limits_hbox = widgets.HBox()
        c_limits_hbox.children += (self.c_min, )
        c_limits_hbox.children += (self.c_max, )
        
        var_select_vbox = widgets.VBox()
        var_select_vbox.children += (self.x_select, )
        var_select_vbox.children += (x_scale_hbox, )
        var_select_vbox.children += (self.x_label, )
        var_select_vbox.children += (x_limits_hbox, )
        var_select_vbox.children += (widgets.VBox(layout=widgets.Layout(width='20px',height='30px')), )
        var_select_vbox.children += (self.y_select, )
        var_select_vbox.children += (y_scale_hbox, )
        var_select_vbox.children += (self.y_label, )
        var_select_vbox.children += (y_limits_hbox, )
        var_select_vbox.children += (widgets.VBox(layout=widgets.Layout(width='20px',height='30px')), )
        var_select_vbox.children += (self.c_select, )   
        var_select_vbox.children += (c_scale_hbox, )
        var_select_vbox.children += (self.c_label, )
        var_select_vbox.children += (c_limits_hbox, )

        input_hbox.children += (file_vbox, )
        input_hbox.children += (var_select_vbox, )

        # Settings region
        self.settings_box = widgets.Textarea(value='Click a point to see settings',placeholder='',description='',disabled=False, layout=widgets.Layout(width='1350px', height='150px'))

        # Assemble GUI
        gui_top = widgets.HBox(layout={'border': '1px solid grey'})
        gui_top.children +=  (input_hbox, )
        gui_top.children +=  (fig_hbox, )

        self.gui.children += (gui_top, )
        self.gui.children += (self.settings_box, )

        self.pop_list = []
        self.settings = {}
        
        self.dummy = 0
        
        self.make_gui()
        
    def make_gui(self):
        self.load_and_plot_on_value_change(None)
        display(self.gui)
        
        self.x_select.observe(self.reset_units_and_plot_on_value_change, names='value')
        self.y_select.observe(self.reset_units_and_plot_on_value_change, names='value')
        self.c_select.observe(self.reset_units_and_plot_on_value_change, names='value')
        self.file_select.observe(self.load_and_plot_on_value_change, names='value')
        self.x_scale.observe(self.plot_on_value_change, names='value')
        self.x_units.observe(self.plot_on_value_change, names='value')
        self.x_label.observe(self.plot_on_value_change, names='value')
        self.y_scale.observe(self.plot_on_value_change, names='value')
        self.y_units.observe(self.plot_on_value_change, names='value')
        self.y_label.observe(self.plot_on_value_change, names='value')
        self.c_scale.observe(self.plot_on_value_change, names='value')
        self.c_units.observe(self.plot_on_value_change, names='value')
        self.c_label.observe(self.plot_on_value_change, names='value')
        
        self.x_min.observe(self.plot_on_value_change, names='value')
        self.x_max.observe(self.plot_on_value_change, names='value')
        self.y_min.observe(self.plot_on_value_change, names='value')
        self.y_max.observe(self.plot_on_value_change, names='value')
        self.c_min.observe(self.plot_on_value_change, names='value')
        self.c_max.observe(self.plot_on_value_change, names='value')
        self.legend_checkbox.observe(self.plot_on_value_change, names='value')
        
        self.active_file.observe(self.active_file_change, names='value')
        
        self.active_color.observe(self.active_color_change, names='value')
        self.legend_str.observe(self.legend_str_change, names='value')
        
    def on_click(self, event):
                
        which_line = self.snap_cursor.which_line
        which_point = self.snap_cursor.data_index
        
        pop = self.pop_list[which_line]
        pop_index = np.array(pop.index)
            
        all_settings = pop.to_dict('index')[pop_index[which_point]]
        
        pop_filename = os.path.join(self.pop_directory, self.file_select.value[which_line])
        
        wanted_keys = {**self.xopt_file['vocs']['variables'], **self.xopt_file['vocs']['constants']}.keys()
        self.settings = dict((k, all_settings[k]) for k in wanted_keys if k in all_settings)
        self.settings_box.value = f'index = {pop_index[which_point]} in {pop_filename}\n{self.x_select.value} = {all_settings[self.x_select.value]:.7g}\n{self.y_select.value} = {all_settings[self.y_select.value]:.7g}\nsettings = {self.settings}'
    
    def reset_units(self, owner):
        if (owner == self.x_select):
            self.x_scale.value = '0'
            self.x_units.value = ''
            self.x_label.value = ''
        
        if (owner == self.y_select):
            self.y_scale.value = '0'
            self.y_units.value = ''
            self.y_label.value = ''
        
        if (owner == self.c_select):
            self.c_scale.value = '0'
            self.c_units.value = ''
            self.c_label.value = ''
    
    def next_default_color(self):
        c = self.default_color_list[0]
        self.default_color_list = self.default_color_list[1:] + [c]
        return c
    
    def update_active_file_list(self):   
        self.active_file.unobserve_all(name='value')
        
        self.active_file.index = 0
        self.active_file.options = self.file_select.value
        
        for file_key in self.active_file.options:
            if file_key not in self.color_dict:
                self.color_dict[file_key] = self.next_default_color()
                
        self.active_file.observe(self.active_file_change, names='value')
    
    def update_active_file_params(self):
        self.active_color.unobserve_all(name='value')
        self.legend_str.unobserve_all(name='value')

        file_key = self.active_file.value
        self.active_color.value = self.color_dict[file_key]
        self.legend_str.value = self.legend_dict[file_key]
        
        self.active_color.observe(self.active_color_change, names='value')
        self.legend_str.observe(self.legend_str_change, names='value')
    
    def load_files(self):
        pop_filenames = list(self.file_select.value)
        
        self.pop_list = []
        
        old_x = self.x_select.value
        old_y = self.y_select.value
        old_c = self.c_select.value
        
        for f in pop_filenames:
            print(os.path.join(self.pop_directory, f))
            self.pop_list += [pd.read_csv(os.path.join(self.pop_directory, f), index_col="xopt_index")]
        
        dropdown_items = self.params_from_xopt
        for pop in self.pop_list:
            dropdown_items += list(pop.columns[1:])
        
        dropdown_items = list(dict.fromkeys(dropdown_items)) # remove duplicates
        
        if (old_x in dropdown_items):
            self.x_select.options = dropdown_items
            self.x_select.value = old_x
        else:
            self.x_select.index = 0
            self.x_select.options = dropdown_items
            self.x_select.value = dropdown_items[0]
        
        if (old_x in dropdown_items):
            self.y_select.options = dropdown_items
            self.y_select.value = old_y
        else:
            self.y_select.index = 0
            self.y_select.options = dropdown_items
            self.y_select.value = dropdown_items[1]
    
        if (old_c in dropdown_items):
            self.c_select.options = ['None'] + dropdown_items
            self.c_select.value = old_c
        else:
            self.c_select.index = 0
            self.c_select.options = ['None'] + dropdown_items
            self.c_select.value = 'None'
 
    
    def make_plot(self):   
        
        if (self.colorbar_instance is not None):
            self.colorbar_instance.remove()
            self.colorbar_instance = None
        
        self.ax.cla()  
        if (self.c_select.value != 'None'):
            cmin = np.min([np.min(pop[self.c_select.value]*10**(-float(self.c_scale.value))) for pop in self.pop_list])
            cmax = np.max([np.max(pop[self.c_select.value]*10**(-float(self.c_scale.value))) for pop in self.pop_list])
            if (cmin >= cmax*(1.0 - 1.0e-14)):  # What were we thinking?!
                cmin = 0.9 * cmin
                cmax = 1.1 * cmax

            if (len(self.c_min.value) > 0):
                cmin = float(self.c_min.value)
            if (len(self.c_max.value) > 0):
                cmax = float(self.c_max.value)
        sc = []
        pl = []
        
        pop_filenames = list(self.file_select.value)
            
        for ii, pop in enumerate(self.pop_list):
            pop_filename = pop_filenames[ii]
            
            x = pop[self.x_select.value] * 10**(-float(self.x_scale.value))
            y = pop[self.y_select.value] * 10**(-float(self.y_scale.value))
            
            if (self.c_select.value != 'None'):
                c = pop[self.c_select.value]* 10**(-float(self.c_scale.value))
            
            not_nan = np.logical_not(np.isnan(x))
            
            if (self.c_select.value != 'None'):
                line_handle = self.ax.scatter(x[not_nan], y[not_nan], 10, c=c[not_nan], vmin=cmin, vmax=cmax, cmap='jet', marker='.')
                sc.append(line_handle)
            else:
                legend_str = self.default_legend_dict[pop_filename]
                if len(self.legend_dict[pop_filename])>0:
                    legend_str = self.legend_dict[pop_filename]
                line_handle, = self.ax.plot(x[not_nan], y[not_nan], '.', color=self.color_dict[pop_filename], label=legend_str) 
                pl.append(line_handle)
                        
        if (self.legend_checkbox.value):
            self.ax.legend()
        
        if (len(pl)>0):
            snap_cursor = SnappingCursor(self.fig, self.ax, pl)
            self.mouse_event_handler_1 = self.fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
            self.mouse_event_handler_2 = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.snap_cursor = snap_cursor
               
        if (len(sc)>0):            
            self.colorbar_instance = plt.colorbar(sc[-1], ax=self.ax)
            
            if (len(self.c_label.value) == 0):
                clabel_str = self.c_select.value
            else:
                clabel_str = self.c_label.value
            
            if (len(self.c_units.value)>0):
                clabel_str += f' ({self.c_units.value})'
            self.colorbar_instance.set_label(clabel_str)
            
            if (self.mouse_event_handler_1 is not None):
                self.fig.canvas.mpl_disconnect(self.mouse_event_handler_1)
            if (self.mouse_event_handler_2 is not None):
                self.fig.canvas.mpl_disconnect(self.mouse_event_handler_2)
            self.snap_cursor = []
        
        if (len(self.x_min.value) > 0):
            self.ax.set_xlim(left=float(self.x_min.value))
        if (len(self.x_max.value) > 0):
            self.ax.set_xlim(right=float(self.x_max.value))
        
        if (len(self.y_min.value) > 0):
            self.ax.set_ylim(bottom=float(self.y_min.value))
        if (len(self.y_max.value) > 0):
            self.ax.set_ylim(top=float(self.y_max.value))
        
        if (len(self.x_label.value) == 0):
            xlabel_str = self.x_select.value
        else:
            xlabel_str = self.x_label.value
            
        if (len(self.x_units.value)>0):
            xlabel_str += f' ({self.x_units.value})'
            
        if (len(self.y_label.value) == 0):
            ylabel_str = self.y_select.value
        else:
            ylabel_str = self.y_label.value
            
        if (len(self.y_units.value)>0):
            ylabel_str += f' ({self.y_units.value})'
        
        self.ax.set_xlabel(xlabel_str)
        self.ax.set_ylabel(ylabel_str)
                
    def reset_units_and_plot_on_value_change(self, change):
        self.reset_units(change['owner'])
        self.make_plot()
        
    def plot_on_value_change(self, change):
        self.make_plot()
        
    def load_and_plot_on_value_change(self, change):
        self.load_files()
        self.update_active_file_list()
        self.update_active_file_params()
        self.make_plot()
        
    def active_file_change(self, change):
        self.update_active_file_params()
        self.make_plot()

    def active_color_change(self, change):
        self.color_dict[self.active_file.value] = self.active_color.value
        self.make_plot()
        
    def legend_str_change(self, change):
        self.legend_dict[self.active_file.value] = self.legend_str.value
        self.make_plot()