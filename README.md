# Student Performance Analyzer (NumPy)

A beginner-friendly data analysis project built with **pure NumPy**. It simulates marks of 30 students across 5 subjects and analyses them without a single `for` loop doing the heavy lifting.

## Features
- Subject-wise mean, median, std deviation, highest, lowest, pass %
- Z-score normalisation using broadcasting
- Vectorised grading with `np.select`
- Student ranking with `np.argsort`
- Most correlated subjects using `np.corrcoef`
- Outlier detection and CSV report export

## NumPy concepts used
`axis` operations - broadcasting - boolean masking - `np.select` - `np.argsort` - `np.unique` - `np.corrcoef` - `np.savetxt`

## Run it
```bash
git clone https://github.com/<your-username>/numpy-student-analyzer.git
cd numpy-student-analyzer
pip install -r requirements.txt
python main.py
```

## Sample output
```
[2] Top 5 students
  #1  Student 10 | 83.60% | Grade A
  ...
```

## Future improvements
- Load real marks from a CSV file
- Add charts with Matplotlib
- Build a Streamlit dashboard

Made by <your name> | [LinkedIn](https://linkedin.com/in/<your-id>)
