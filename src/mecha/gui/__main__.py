from .app import build_app
import panel as pn

pn.serve(build_app(), port=5006, show=True, title="MECHA GUI")
