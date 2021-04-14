try:
    import ast
    from ipywidgets import Button, HBox, VBox
    from ipywidgets import widgets

    from IPython.display import display
    from IPython.core.display import Javascript, display, HTML
    from IPython.display import FileLink, FileLinks
    from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
except Exception:
    print("widgets not available...")


def autofix():
    pass


WIDTH = "200px"


class Display:
    def __init__(self, heading, options, value, db_vars, var):
        h = widgets.Label(
            value="Select Rows:",
            layout=widgets.Layout(width=WIDTH),
        )
        sm = widgets.SelectMultiple(options=db.rm.exp_params, value=list(db.vars.get("latex_rows")))
        return

    def update():
        pass


class Text:
    def __init__(self, header, default, type, db_vars, var):
        org_value = db_vars.get(var)

        if org_value is None or org_value == "":
            value = default
        else:
            value = str(org_value)
        self.type = type
        self.header = widgets.Label(
            value=header,
            layout=widgets.Layout(width=WIDTH),
        )
        self.text = widgets.Text(
            value=value,
            disabled=False,
            layout=widgets.Layout(width=WIDTH),
        )
        self.db_vars = db_vars
        self.var = var

    def get_widget(self):
        return widgets.VBox([self.header, self.text])

    def update(self):
        self.db_vars[self.var] = str(self.text.value)
        value = self.db_vars[self.var]
        if self.type == "tuple":
            w, h = value.strip("(").strip(")").split(",")
            value = (int(w), int(h))
        if self.type == "int":
            value = int(value)
        if self.type == "str":
            value = str(value)

        return value


class SelectMultiple:
    def __init__(self, header, options, db_vars, var, select_all=False):
        org_value = db_vars.get(var)

        if org_value is None:
            if len(options) > 0:
                if select_all:
                    value = options
                else:
                    value = [options[0]]
            else:
                value = []
        else:
            value = [v for v in org_value if v in options]
            if len(value) == 0 and len(options) > 0:
                value = [options[0]]

        self.header = widgets.Label(
            value=header,
            layout=widgets.Layout(width=WIDTH),
        )
        self.select_multiple = widgets.SelectMultiple(options=options, value=value, layout=widgets.Layout(width=WIDTH))
        self.db_vars = db_vars
        self.var = var

    def get_widget(self):
        return widgets.VBox([self.header, self.select_multiple])

    def update(self):
        self.db_vars[self.var] = list(self.select_multiple.value)
        return self.db_vars[self.var]


class Dropdown:
    def __init__(self, header, options, db_vars, var):
        org_value = db_vars.get(var)

        if org_value is None or org_value not in options:
            if len(options) > 0:
                value = options[0]
            else:
                value = None
        else:
            value = org_value

        self.header = widgets.Label(
            value=header,
            layout=widgets.Layout(width=WIDTH),
        )
        self.dropdown = widgets.Dropdown(
            options=options,
            value=value,
            disabled=False,
            layout=widgets.Layout(width=WIDTH),
        )
        self.db_vars = db_vars
        self.var = var

    def get_widget(self):
        return widgets.VBox([self.header, self.dropdown])

    def update(self):
        self.db_vars[self.var] = self.dropdown.value
        value = self.db_vars[self.var]
        if value == "None":
            return None
        return value


def create_button():
    pass


def create_download():
    pass


def create_text_box():
    pass
