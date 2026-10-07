"""Tool for step A1: label Thai patches by eye. Started from notebooks/label_thai_ui.ipynb.

One picture shows the reference examples on the left and the patch to label on the right; the answer buttons
sit under it. Every click is saved at once to outputs/manual_labels/<labeler>.csv with the columns
patch_id, manual_label, labeler, time, peeked. `peeked` is True when the WorldCover label of that patch was
shown (by the "show WorldCover label" button) before the answer.

The logic lives here, not in the notebook, so a change reaches everybody after a kernel restart without
having to reload the notebook.

Usage (in the notebook):  import label_tool; label_tool.start("your_name")
"""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display

import config as c

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

LABEL_DIR = c.OUTPUT_DIR / "manual_labels"
COLUMNS = ["patch_id", "manual_label", "labeler", "time", "peeked"]
TRUE = [c.BANDS.index(b) for b in ("B4", "B3", "B2")]
FALSE = [c.BANDS.index(b) for b in ("B8", "B4", "B3")]
KEYS = {str(i + 1): name for i, name in enumerate(c.CLASSES)}
KEYS["0"] = "unsure"

ui = {}   # the widgets of the last session, so that tests can press the buttons


def start(labeler, group="all", fig_width=15.5, fig_aspect=2.0):
    """Show the labelling interface. group: "all" (200 patches) or 1, 2, 3 (60 own + 20 shared patches)."""
    out_file = LABEL_DIR / f"{labeler}.csv"
    sample = pd.read_csv(LABEL_DIR / "label_sample.csv")
    chosen = sample if group == "all" else sample[sample["group"].isin([0, group])]
    mine = chosen.sort_values("order")["patch_id"].tolist()

    X = np.load(c.PATCH_DIR / "thailand_X_L2A.npy", mmap_mode="r")   # (9216, 12, 64, 64); row = patch_id
    ref = plt.imread(LABEL_DIR / "reference_compact.png")           # reference examples, drawn left of every patch
    # WorldCover label of every patch; shown only after the reveal button
    wc = pd.read_csv(c.PATCH_DIR / "thailand_patches.csv", keep_default_na=False).set_index("patch_id")[["label", "purity"]]

    rows = pd.read_csv(out_file).to_dict("records") if out_file.exists() else []
    for r in rows:
        r.setdefault("peeked", False)
    done = {r["patch_id"] for r in rows}
    todo = [p for p in mine if p not in done]
    peeked = {r["patch_id"] for r in rows if r["peeked"]}

    def save():
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out_file, index=False)

    def show(patch_id, position, total):
        patch = X[patch_id].astype("float32")
        height = fig_width / fig_aspect
        ref_w = ref.shape[1] / ref.shape[0] * height / fig_width     # width of the reference, as a fraction of the picture
        side = 0.42 * height / fig_width                             # width of a square patch, as a fraction of the picture
        x0 = ref_w + (1 - ref_w - side) / 2
        fig = plt.figure(figsize=(fig_width, height))
        left_ax = fig.add_axes([0, 0, ref_w, 1])
        left_ax.imshow(ref)
        left_ax.axis("off")
        true_ax = fig.add_axes([x0, 0.53, side, 0.42])
        false_ax = fig.add_axes([x0, 0.04, side, 0.42])
        true_ax.imshow(np.clip(patch[TRUE].transpose(1, 2, 0) / 3000, 0, 1), interpolation="bilinear")
        true_ax.set_title(f"TO LABEL: patch {patch_id} ({position}/{total}){chr(10)}true colour (B4, B3, B2)", fontsize=11)
        false_ax.imshow(np.clip(patch[FALSE].transpose(1, 2, 0) / 5000, 0, 1), interpolation="bilinear")
        false_ax.set_title("false colour (B8, B4, B3)", fontsize=11)
        for ax in (true_ax, false_ax):
            ax.set_xticks([])
            ax.set_yticks([])
        display(fig)      # draw it now, not when the cell ends
        plt.close(fig)

    if widgets is None:
        print("ipywidgets is not installed (pip install ipywidgets, then restart the kernel).")
        return

    picture = widgets.Output()
    hint = widgets.HTML()
    status = widgets.HTML()

    def render():
        hint.value = ""
        with picture:
            clear_output(wait=True)
            if todo:
                show(todo[0], len(rows) + 1, len(mine))
            else:
                print("All done. Commit your file in outputs/manual_labels/.")
        status.value = (f"<b>{len(rows)} of {len(mine)}</b> patches labelled by {labeler}; saved to {out_file}; "
                        f"{len(peeked)} of them with the WorldCover label shown")

    def answer(label):
        if not todo:
            return
        patch_id = todo.pop(0)
        rows.append({"patch_id": patch_id, "manual_label": label, "labeler": labeler,
                     "time": datetime.now().isoformat(timespec="seconds"), "peeked": patch_id in peeked})
        save()
        render()

    def undo(_):
        if rows:
            todo.insert(0, rows.pop()["patch_id"])
            save()
            render()

    def reveal(_):
        if not todo:
            return
        peeked.add(todo[0])
        info = wc.loc[todo[0]]
        hint.value = (f"WorldCover says: <b>{info['label']}</b> (purity {info['purity']:.0%}). It is only about 75% accurate, "
                      f"so do not simply copy it. This answer will be marked as <i>peeked</i>.")

    size = widgets.Layout(width="150px", height="42px")
    buttons = []
    for key, label in KEYS.items():
        button = widgets.Button(description=f"{key}  {label}", layout=size)
        button.on_click(lambda _, label=label: answer(label))
        buttons.append(button)
    undo_button = widgets.Button(description="undo", button_style="warning", layout=size)
    undo_button.on_click(undo)
    reveal_button = widgets.Button(description="show WorldCover label", button_style="info",
                                   layout=widgets.Layout(width="190px", height="42px"))
    reveal_button.on_click(reveal)
    buttons += [undo_button, reveal_button]

    render()
    display(widgets.VBox([picture, widgets.HBox(buttons, layout=widgets.Layout(flex_flow="row wrap")), hint, status]))
    ui.update(picture=picture, hint=hint, status=status, buttons=buttons, out_file=out_file)
