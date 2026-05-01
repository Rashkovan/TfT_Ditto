# TfT_Ditto
Project for Tools for Though S26

The flow of the architercture: 

models.py -- AnnotatedVariable and VariableResult dataclasses

parser.py -- Two-pass regex + AST parser for # !Ditto: <tag> annotations

inference.py -- Prior sampling via eval() + SVI posterior path via importlib + Pyro Predictive

main.py -- FastAPI with /upload (POST) and /session/{id} (GET), in-memory session state

static/index.html -- Single-page app, Chart.js loaded from CDN

static/app.js -- KDE in ~20 lines (Silverman bandwidth + Gaussian kernel), Chart.js rendering, drag-drop upload, single/compare grid

static/style.css -- Dark-first, CSS variables, card grid, two-column compare layout with divider

example_model.py -- Ready-to-upload demo with 4 annotated priors

requirements.txt -- fastapi, uvicorn[standard], python-multipart, pyro-ppl>=1.8, torch
