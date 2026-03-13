"""
Thesis pairs for ArtCon classification.
Each pair groups a topic with its pro/con thesis statements.
Classification will be mutually exclusive per pair: support / oppose / neutral.
"""

THESIS_PAIRS = [
    {
        "key":     "current_possibility",
        "topic":   "whether artificial consciousness is currently possible",
        "pro":     "Artificial consciousness is currently possible.",
        "con":     "Currently, artificial consciousness is not possible.",
    },
    {
        "key":     "future_possibility",
        "topic":   "whether artificial consciousness will ever be possible",
        "pro":     "Artificial consciousness will be possible in the future.",
        "con":     "In principle, artificial consciousness is not possible.",
    },
    {
        "key":     "functionalism",
        "topic":   "whether functionalism about consciousness is correct",
        "pro":     "Functionalism about consciousness is correct.",
        "con":     "Functionalism about consciousness is false.",
    },
    {
        "key":     "computational_functionalism",
        "topic":   "whether computational functionalism about consciousness is correct",
        "pro":     "Computational functionalism about consciousness is correct.",
        "con":     "Computational functionalism about consciousness is false.",
    },
    {
        "key":     "biology",
        "topic":   "whether biology is necessary for consciousness",
        "pro":     "Biology is necessary for consciousness.",
        "con":     "Biology is not necessary for consciousness.",
    },
]

# Flat list of all individual theses (for backwards compatibility)
THESES = [t for pair in THESIS_PAIRS for t in (pair["pro"], pair["con"])]
