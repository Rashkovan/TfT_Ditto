# The Ditto Manual

Ditto is a DAG visualizer for [Pyro](https://pyro.ai/) probabilistic models. You annotate your Pyro source with a few lightweight comments, point Ditto at the file, and Ditto returns a browser-based dashboard that shows the dependency graph of your model with prior and posterior plots attached to each node. From the dashboard you can also drop in new model files, edit priors live, and load alternative datasets without restarting the server.

This manual is written for researchers who already know Python and the basics of Pyro. It walks through installation, annotation, the CLI, configuration, and common pitfalls.

---

## 1. Installation

### Requirements

- Python 3.9 or later. (Ditto uses `ast.unparse`, which arrived in 3.9.)
- An OS Tk install if you want the native "Open File…" button. On macOS and Windows this is bundled with Python; on Linux you may need to install `python3-tk` (`sudo apt install python3-tk`).

### Install from source

```bash
git clone <your-repo-url> ditto
cd ditto
pip install -e .
```

The console script is named `ditto-viz`. After install you should be able to run:

```bash
ditto-viz --help
```

### Optional dependencies

- `pandas` — required if you plan to load `.csv` data files from the UI. Already a transitive dependency of most Pyro environments, but `pip install pandas` if you don't have it.
- `pygraphviz` — only used by the static `--export` path for hierarchical layout. Skip if you don't care about static PNG export.

---

## 2. Annotating Your Pyro Model

Ditto only sees variables that you mark with a `# !Ditto:` comment placed on the line directly above an assignment or a bare `pyro.sample(...)` expression statement. Three tags are recognised: `prior`, `latent`, and `observed`.

### The three tags

- `prior` — a variable sampled from its declared distribution. The
  hover/click panel shows just a prior plot.
- `latent` — a latent random variable to be inferred via SVI. The visualization on the right shows the prior and the posterior stacked vertically. Ditto auto-builds the guide; you do *not* declare one yourself.
- `observed` — a likelihood / conditioning site. The panel shows the posterior predictive distribution with the actual observed data concatenated in.

You can combine tags by separating with a comma: `# !Ditto: prior, latent` makes the variable both a prior (which gets sampled directly) and a latent (which gets a posterior). The display tag (used for the node colour) is chosen by priority: `latent > observed > prior`.

### Two annotation styles

#### Assignment-style

Put the tag above an assignment whose right-hand side is a Pyro/PyTorch distribution:

```python
import pyro.distributions as dist

# !Ditto: prior
mu = dist.Normal(0., 1.)

# !Ditto: prior
sigma = dist.HalfNormal(1.)

# !Ditto: observed
y = dist.Normal(mu, sigma)
```

This style is convenient for top-of-file definitions you'll reference inside a `model` function.

#### Bare `pyro.sample(...)` style

Put the tag above a `pyro.sample(...)` expression statement, typically inside your `model` function. The Pyro site name (the string-literal first argument) becomes the variable name in the DAG:

```python
def model(x, obs=None):
    # !Ditto: latent
    alpha = pyro.sample("alpha", dist.Normal(0., 1.))

    # !Ditto: latent
    beta = pyro.sample("beta", dist.Normal(0., 1.))

    # !Ditto: observed
    pyro.sample("obs", dist.Normal(alpha + beta * x, 0.1), obs=obs)
```

The bare form is required for `observed` sites — Pyro's `obs=...` conditioning lives on the `pyro.sample` call itself, so there's nothing to assign to.

—-

## 3. Required Components in Your Source File

Your annotated `.py` file must define two top-level callables:

### `model(*args, **kwargs)`

A standard Pyro model: a Python callable that uses `pyro.sample` to draw from priors, applies any deterministic transforms, and conditions on data with `obs=`. Required when any `latent` or `observed` variables are annotated. It is the callable Ditto traces, runs SVI on, and uses for posterior prediction.

### `get_data() -> (args, kwargs)`

Returns the positional and keyword arguments to pass to `model`. Required whenever `model` is invoked (i.e. whenever any `latent` or `observed` variable is annotated). Example:

```python
import torch

def get_data():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.1, 3.9, 6.2])
    return (x,), {"obs": y}
```

If `get_data` is missing, Ditto raises a clear `AttributeError` instead of the Python default `module has no attribute 'get_data'` message.

You no longer need to write a guide. Ditto auto-creates `AutoNormal` for bare-distribution latents or `AutoNormalizingFlow` (with `block_autoregressive`) when any latent uses `pyro.sample(...)`. The discrete latent sites are blocked from the autoguide automatically via `poutine.block`.

---

## 4. Starting the App

### Launch with a file

```bash
ditto-viz path/to/your_model.py
```

This runs the full pipeline (parse, build the graph, sample priors, run SVI, sample the posterior) and then opens the Dash app at http://localhost:8050.

### Launch without a file

```bash
ditto-viz
```

The Dash app comes up immediately with an empty canvas and a prompt to drop a `.py` file or click "Open File…". Use this when you want to explore multiple models in a single session, or when you don't yet have a
model file to point at.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `ditto.yaml` | YAML configuration file. Missing files are tolerated; built-in defaults are used. |
| `--port N` | from config (8050) | Port for the Dash server. Overrides `server.port` in the YAML. |
| `--export PATH` | off | Render a static PNG of the DAG to `PATH` instead of (or in addition to) launching the app. Requires a filepath. |
| `--dry-run` | off | Run parse + build + inference and print a summary, then exit without launching the server. Requires a filepath. |

`--export` and `--dry-run` both require an explicit filepath; calling them with no positional argument is an error (the empty-canvas mode has no graph to export or summarise).

---

## 5. Using the App

The window is split into three regions:

- The toolbar across the top: the title, "Open File…" button, "Refresh" button, and a status message line.
- The DAG panel on the left: a draggable, zoomable Cytoscape graph.
- The side panel on the right: hover/click info, a prior editor, and a data uploader.

### Loading a `.py` file

#### Drag and drop

Drag a `.py` file from your file manager onto the dashed "Drop a .py file here" box that sits over the DAG panel. Ditto will:

1. Decode the upload and write it to a temp file.
2. Run the full pipeline (parse, build, sample, run SVI, sample posteriors).
3. Render the DAG with thumbnails attached to each node.

If parsing or inference fails, the status message at the top shows the error and the canvas stays in its previous state. The drag-drop overlay shrinks to a thin strip after a successful load so it doesn't obscure the graph; you can still drop another file on it to swap models.

#### "Open File…" button

Click "Open File…" in the toolbar to open your operating system's native file dialog (powered by `tkinter.filedialog.askopenfilename`). Pick a `.py` file and Ditto runs the same load pipeline as drag-drop.

If `tkinter` is not available (e.g. on a headless server, or because `python3-tk` is not installed), the status line shows a friendly "File dialog unavailable" message and the rest of the app keeps working; use drag-drop instead.

### Navigating the DAG

- **Pan**: click-drag empty space.
- **Zoom**: scroll wheel or pinch.
- **Drag a node**: click-drag a node to reposition it. Layout recomputation is disabled (`autoRefreshLayout=False`) so manual positions stick.

### Hovering for distribution plots

Move the mouse over a node and the side panel updates with the node's KDE/histogram. `prior` nodes show one plot; `latent` nodes show two (prior on top, posterior below); `observed` nodes show the posterior predictive (with the actual observed values mixed in).

The plots are pre-computed as base64 PNGs at app startup; the hover callback is a pure dict lookup, so latency is minimal even for large graphs.

### Editing a prior
*Currently not supported*
Click (don't just hover) a node tagged `prior` or `latent`. The side panel adds an "Edit prior" textarea pre-filled with the variable's current expression and an "Apply Prior" button. To change the prior:

1. Edit the expression in the textarea. Anything that parses as a Python expression is accepted (e.g. `dist.Normal(0., 5.)` or `dist.Beta(2., 3.)`).
2. Click "Apply Prior". Ditto:
   - Validates the expression with `ast.parse(expr, mode="eval")`. A red error message appears if the expression doesn't parse.
   - Stores the new expression on the in-memory `AnnotatedVariable`.
   - Re-samples that variable's prior using the existing inference engine.
   - Rebuilds only that node's prior thumbnail.
3. Click "Refresh" to re-run the full SVI + posterior sampling pipeline so the change propagates into the posterior plots.

The edit is held in memory only — your source file is untouched.

### Loading a data file
*Currently not supported*

### Refresh
*Currently not supported*

---

## 6. Configuration

Ditto loads `ditto.yaml` from the working directory (overridable with `--config`). Every key has a built-in default, so missing files and partial files are both fine — your YAML is layered on top of the defaults.

The full schema:

```yaml
inference:
  svi_steps: 2000          # number of SVI iterations
  learning_rate: 0.01      # passed to pyro.optim.Adam
  num_samples: 1000        # draws per variable for prior + posterior

visualization:
  kde_points: 200          # x-axis resolution for KDE curves
  histogram_bins: 40       # bins for histogram fallback
  figure_size: [4, 3]      # per-thumbnail figsize in inches
  prior_color: "#4C72B0"   # blue
  observed_color: "#DD8452"# orange
  latent_color: "#55A868"  # green

server:
  port: 8050               # Dash server port
  debug: false             # currently unused; debug=False is forced

export:
  path: null               # default --export target (CLI flag overrides)
  format: png              # extension chosen by output_path
  dpi: 150                 # static export DPI
```

The `--port` CLI flag overrides `server.port`. The `export.*` keys are read only when `--export` is passed.
