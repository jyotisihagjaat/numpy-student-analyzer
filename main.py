"""Student Performance Analyzer - a NumPy project.

Generates marks for 30 students in 5 subjects and analyses them using
pure NumPy: statistics, z-score normalisation, grading, ranking,
pass percentage and subject correlation.
"""
import numpy as np

SUBJECTS = ["Maths", "Physics", "Chemistry", "English", "Computer"]
PASS_MARK = 40
rng = np.random.default_rng(42)  # fixed seed -> same results every run


def generate_marks(n_students=30):
    """Random marks (35-99) -> shape (students, subjects)."""
    return rng.integers(35, 100, size=(n_students, len(SUBJECTS)))


def subject_stats(marks):
    """Column-wise statistics using the axis argument."""
    return {
        "Mean": marks.mean(axis=0),
        "Median": np.median(marks, axis=0),
        "Std Dev": marks.std(axis=0),
        "Highest": marks.max(axis=0),
        "Lowest": marks.min(axis=0),
        "Pass %": (marks >= PASS_MARK).mean(axis=0) * 100,
    }


def z_score(marks):
    """Normalise each subject: (x - mean) / std  (uses broadcasting)."""
    return (marks - marks.mean(axis=0)) / marks.std(axis=0)


def assign_grades(percent):
    """Vectorised grading with np.select (no loops)."""
    conditions = [percent >= 90, percent >= 75, percent >= 60, percent >= 45]
    return np.select(conditions, ["A+", "A", "B", "C"], default="D")


def top_correlated_pair(marks):
    """Find the two subjects whose marks are most correlated."""
    corr = np.corrcoef(marks, rowvar=False)
    np.fill_diagonal(corr, -np.inf)  # ignore self-correlation
    i, j = np.unravel_index(np.argmax(corr), corr.shape)
    return SUBJECTS[i], SUBJECTS[j], corr[i, j]


def main():
    marks = generate_marks()
    total = marks.sum(axis=1)
    percent = total / (len(SUBJECTS) * 100) * 100
    grades = assign_grades(percent)
    rank_order = np.argsort(-percent)  # best student first

    print("=" * 62)
    print("        STUDENT PERFORMANCE ANALYZER (NumPy)")
    print("=" * 62)

    print("\n[1] Subject-wise statistics")
    print(f"{'':10}" + "".join(f"{s:>11}" for s in SUBJECTS))
    for name, values in subject_stats(marks).items():
        print(f"{name:10}" + "".join(f"{v:11.2f}" for v in values))

    print("\n[2] Top 5 students")
    for rank, idx in enumerate(rank_order[:5], start=1):
        print(f"  #{rank}  Student {idx + 1:02d} | {percent[idx]:.2f}% | Grade {grades[idx]}")

    print("\n[3] Grade distribution")
    labels, counts = np.unique(grades, return_counts=True)
    for g, c in zip(labels, counts):
        print(f"  {g:3} : {'#' * c} ({c})")

    a, b, r = top_correlated_pair(marks)
    print(f"\n[4] Most correlated subjects: {a} & {b} (r = {r:.2f})")

    z = z_score(marks)
    print(f"\n[5] Outliers (|z| > 2): {int((np.abs(z) > 2).sum())} marks found")

    print(f"\n[6] Class average: {percent.mean():.2f}% | "
          f"Toppers (>=90%): {int((percent >= 90).sum())}")

    # Save the report
    rows = np.column_stack([
        np.arange(1, len(marks) + 1), marks, total,
        np.round(percent, 2), grades,
    ])
    header = "Student," + ",".join(SUBJECTS) + ",Total,Percent,Grade"
    np.savetxt("results.csv", rows, delimiter=",", header=header, fmt="%s", comments="")
    print("\nReport saved to results.csv")


if __name__ == "__main__":
    main()
