# Three-body project (data part)

State layout (12-dim): [x1,y1,x2,y2,x3,y3, vx1,vy1,vx2,vy2,vx3,vy3]; G=1, masses 1, E0=-0.5.

## Setup
    python -m venv venv
    venv\Scripts\Activate.ps1        # Windows   (Mac/Linux: source venv/bin/activate)
    pip install -r requirements.txt

## Use the shared dataset (do NOT regenerate)
    python check_data.py             # must print the EXPECTED numbers
    python make_figures.py           # report figures -> figs/*.pdf, prints Table values

`data/threebody_raw.npz` is the official dataset (chaos => regenerating gives different data).
`make_data.py` only exists for documentation/reproducibility and refuses to overwrite it.

## In your own code
    from preprocess import *
    trajs, dt, E0 = load_dataset()
    tr, va, te = split_trajectories(trajs)      # same split for everybody (seed 0)
    tr = subsample(tr, stride)                  # network step = stride * 0.01
    sq = Squasher().fit(tr); z = [sq.transform(t) for t in tr]
    X, Y = make_windows(z, seq_len=50)          # RNN input/target windows
